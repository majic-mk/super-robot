from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Tuple

from .experiment_jobs import E1Job, E1Result, ResultStatus
from .manifest import ManifestCase, ManifestSource
from .metrics import best_answer_f1, token_id_f1
from .repair_semantics import repaired_segment_token_count
from .cacheblend_closed_loop_runtime import CacheBlendRuntimeCapabilities


DEFAULT_INSTRUCTION = (
    "You will be asked a question after reading several passages. "
    "Answer directly using only the passages.\nPassages:\n"
)


@dataclass(frozen=True)
class GenerationMeasurement:
    token_ids: Tuple[int, ...]
    text: str
    output_hash: str
    ttft_ms: float
    total_host_ms: float
    total_gpu_ms: float
    logits_trace: Tuple[Any, ...]


@dataclass
class CanonicalSourceKV:
    source_id: str
    layers: Sequence[Tuple[Any, Any]]
    digest: str


class CacheBlendCaseRuntime:
    """Case-scoped bridge to the pinned CacheBlend/vLLM 0.4.1 stack.

    Heavy dependencies are supplied by the server worker so importing the
    ProbeKV package remains possible on a CPU-only development machine.

    This class intentionally remains the CB1-CB3/H1 case runner.  It performs
    complete ``generate()`` calls and must not be substituted for the v5
    online runtime merely because both use the same CacheBlend repair patch.
    """

    paper_evidence = False
    runtime_mode = "h1_case_runner"

    @staticmethod
    def capabilities() -> CacheBlendRuntimeCapabilities:
        return CacheBlendRuntimeCapabilities(
            backend_name="cacheblend_case_runner",
            async_source_loading=False,
            layer_resumable_prefill=False,
            scheduler_feedback=False,
            boundary_conditioned_profiles=False,
            canonical_sources_read_only=True,
            cuda_event_timing=True,
        )

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        sampling_params_type: Any,
        case: ManifestCase,
        provenance: Mapping[str, str],
        max_new_tokens: int = 64,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        case.validate()
        if case.split == "test":
            raise ValueError("server pilot cannot open a locked test case")
        if case.split != "pilot":
            raise ValueError("CacheBlend H1 worker accepts pilot cases only")
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params_type = sampling_params_type
        self.case = case
        self.provenance = dict(provenance)
        self.max_new_tokens = int(max_new_tokens)
        self.instruction = instruction
        self.outer_model, self.model = self._model_handles()
        self.layers = self.model.layers
        self.torch = self._torch()
        self.sources: Dict[str, CanonicalSourceKV] = {}
        self._manifest_sources: Dict[str, ManifestSource] = {}
        self.prefix_kv: Sequence[Tuple[Any, Any]] = ()
        self.suffix_kv: Sequence[Tuple[Any, Any]] = ()
        self._closed = False
        try:
            self._initialize_case()
        except Exception:
            self.close()
            raise

    def _initialize_case(self) -> None:
        case = self.case
        self.prefix_ids = self._prefix_ids(
            case.current_context, self.instruction
        )
        encoded_segment = tuple(
            int(value)
            for value in self.tokenizer.encode(
                case.segment_text, add_special_tokens=False
            )
        )
        if encoded_segment != case.segment_token_ids:
            raise ValueError(
                "manifest C tokens do not match the frozen tokenizer revision"
            )
        self.segment_ids = encoded_segment
        self.suffix_ids = tuple(
            int(value)
            for value in self.tokenizer.encode(
                "%s\n\nQuestion: %s\nAnswer briefly:"
                % (case.current_suffix_context, case.question),
                add_special_tokens=False,
            )
        )
        if not self.suffix_ids:
            raise ValueError("mandatory suffix S must contain tokens")
        self.current_ids = self.prefix_ids + self.segment_ids + self.suffix_ids
        self.full = self._generate(
            self.current_ids, self.max_new_tokens, check=False, capture_logits=True
        )
        self.full_answer_f1 = best_answer_f1(self.full.text, case.answers)
        self.prefix_kv = self._collect_component(self.prefix_ids)
        # S is only a shape-correct placeholder. Every S token is mandatory
        # dense and overwrites these tensors from the first reuse layer onward.
        self.suffix_kv = self._collect_component(self.suffix_ids)
        self.sources = {
            source.source_id: self._build_source(source)
            for source in case.sources
        }
        self._manifest_sources = {
            source.source_id: source for source in case.sources
        }
        self._case_digest = hashlib.sha256(
            json.dumps(
                case.to_row(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __enter__(self) -> "CacheBlendCaseRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release every model-global reference retained by the patched stack."""
        if self._closed:
            return
        metadata = self.model.cache_fuse_metadata
        metadata["check"] = False
        metadata["collect"] = False
        metadata["capture_logits"] = False
        metadata["imp_indices"] = None
        metadata["attn_bias"] = None
        metadata["selected_segment_indices"] = None
        self.model.old_kvs = []
        self.outer_model.probekv_logits_trace = []
        for layer in self.layers:
            if hasattr(layer.self_attn, "hack_kv"):
                layer.self_attn.hack_kv = None
        self.sources.clear()
        self.prefix_kv = ()
        self.suffix_kv = ()
        self._closed = True

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required on the CacheBlend server") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CacheBlend server runtime requires CUDA")
        return torch

    def _model_handles(self) -> Tuple[Any, Any]:
        try:
            outer = (
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            )
            inner = outer.model
            inner.cache_fuse_metadata
            inner.layers
        except AttributeError as error:
            raise RuntimeError(
                "pinned CacheBlend model handles do not match vLLM 0.4.1"
            ) from error
        return outer, inner

    def _prefix_ids(self, context: str, instruction: str) -> Tuple[int, ...]:
        body = tuple(
            int(value)
            for value in self.tokenizer.encode(
                instruction + context, add_special_tokens=False
            )
        )
        bos = getattr(self.tokenizer, "bos_token_id", None)
        return ((int(bos),) if bos is not None else ()) + body

    def _source_prefix_ids(
        self, source: ManifestSource, instruction: str = DEFAULT_INSTRUCTION
    ) -> Tuple[int, ...]:
        return self._prefix_ids(source.historical_context, instruction)

    def _generate(
        self,
        prompt_token_ids: Sequence[int],
        max_tokens: int,
        check: bool,
        capture_logits: bool,
    ) -> GenerationMeasurement:
        metadata = self.model.cache_fuse_metadata
        metadata["check"] = bool(check)
        metadata["collect"] = False
        metadata["capture_logits"] = bool(capture_logits)
        self.outer_model.probekv_logits_trace = []
        params = self.sampling_params_type(
            temperature=0,
            max_tokens=max_tokens,
        )
        start_event = self.torch.cuda.Event(enable_timing=True)
        end_event = self.torch.cuda.Event(enable_timing=True)
        self.torch.cuda.synchronize()
        host_start = time.perf_counter()
        start_event.record()
        try:
            try:
                outputs = self.llm.generate(
                    prompt_token_ids=[list(prompt_token_ids)],
                    sampling_params=params,
                    use_tqdm=False,
                )
            except TypeError:
                outputs = self.llm.generate(
                    prompts=None,
                    prompt_token_ids=[list(prompt_token_ids)],
                    sampling_params=params,
                    use_tqdm=False,
                )
            end_event.record()
            self.torch.cuda.synchronize()
        finally:
            metadata["capture_logits"] = False
        host_ms = (time.perf_counter() - host_start) * 1000.0
        gpu_ms = float(start_event.elapsed_time(end_event))
        request = outputs[0]
        completion = request.outputs[0]
        output_ids = tuple(int(value) for value in completion.token_ids)
        output_text = str(completion.text)
        output_hash = hashlib.sha256(
            json.dumps(
                {"token_ids": output_ids, "text": output_text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        metrics = request.metrics
        ttft_ms = (
            float(metrics.first_token_time - metrics.first_scheduled_time)
            * 1000.0
        )
        logits = tuple(self.outer_model.probekv_logits_trace)
        return GenerationMeasurement(
            output_ids,
            output_text,
            output_hash,
            ttft_ms,
            host_ms,
            gpu_ms,
            logits,
        )

    def _collect_component(
        self, prompt_token_ids: Sequence[int]
    ) -> Tuple[Tuple[Any, Any], ...]:
        metadata = self.model.cache_fuse_metadata
        metadata["check"] = False
        metadata["collect"] = True
        metadata["capture_logits"] = False
        params = self.sampling_params_type(temperature=0, max_tokens=1)
        try:
            try:
                self.llm.generate(
                    prompt_token_ids=[list(prompt_token_ids)],
                    sampling_params=params,
                    use_tqdm=False,
                )
            except TypeError:
                self.llm.generate(
                    prompts=None,
                    prompt_token_ids=[list(prompt_token_ids)],
                    sampling_params=params,
                    use_tqdm=False,
                )
            collected = []
            for layer in self.layers:
                key, value = layer.self_attn.hack_kv
                if key.shape[0] != len(prompt_token_ids):
                    raise RuntimeError(
                        "collected KV length %d does not match prompt tokens %d"
                        % (key.shape[0], len(prompt_token_ids))
                    )
                collected.append(
                    (key.detach().clone(), value.detach().clone())
                )
        finally:
            metadata["collect"] = False
        return tuple(collected)

    def _build_source(self, source: ManifestSource) -> CanonicalSourceKV:
        historical_prefix = self._source_prefix_ids(source, self.instruction)
        combined = historical_prefix + self.segment_ids
        source_prefill = self._collect_component(combined)
        segment_start = len(historical_prefix)
        assembled = []
        for source_key, source_value in source_prefill:
            c_key = source_key[segment_start:].detach().clone()
            c_value = source_value[segment_start:].detach().clone()
            if c_key.shape[0] != len(self.segment_ids):
                raise RuntimeError("canonical Source C slice has an invalid shape")
            assembled.append((c_key, c_value))
        digest = self._kv_digest(assembled)
        return CanonicalSourceKV(source.source_id, tuple(assembled), digest)

    def _kv_digest(self, layers: Sequence[Tuple[Any, Any]]) -> str:
        digest = hashlib.sha256()
        for key, value in layers:
            for tensor in (key, value):
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                raw = (
                    tensor.detach()
                    .contiguous()
                    .view(self.torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                digest.update(raw)
        return digest.hexdigest()

    @staticmethod
    def _relative_l2(
        observed: Sequence[Any], reference: Sequence[Any], limit: int = 32
    ) -> float:
        count = min(limit, len(observed), len(reference))
        if count <= 0:
            raise RuntimeError("logit instrumentation produced an empty trace")
        numerator = 0.0
        denominator = 0.0
        for index in range(count):
            left = observed[index]
            right = reference[index]
            if tuple(left.shape) != tuple(right.shape):
                raise RuntimeError("cached/full logit trace shapes differ")
            difference = left - right
            numerator += float((difference * difference).sum().item())
            denominator += float((right * right).sum().item())
        return math.sqrt(numerator / max(denominator, 1e-30))

    def stage_selected_source(
        self,
        source_id: str,
        reuse_layer: int,
        repair_ratio: float,
    ) -> str:
        """Stage one locked Source for the unchanged CacheBlend repair path.

        This is the reusable repair primitive shared by the H1 worker and the
        future A800 online engine. It does not perform Source selection,
        loading/scheduling, or final admission.
        """
        if self._closed:
            raise RuntimeError("CacheBlend case runtime is closed")
        if source_id not in self.sources:
            raise ValueError("unknown selected Source")
        if not 1 <= reuse_layer <= len(self.layers):
            raise ValueError("reuse_layer must be 1-based")
        if not 0.0 <= repair_ratio <= 1.0:
            raise ValueError("repair_ratio must be in [0, 1]")
        canonical = self.sources[source_id]
        self.model.old_kvs = []
        for layer_index, (c_key, c_value) in enumerate(canonical.layers):
            p_key, p_value = self.prefix_kv[layer_index]
            s_key, s_value = self.suffix_kv[layer_index]
            self.model.old_kvs.append(
                [
                    self.torch.cat((p_key, c_key, s_key), dim=0),
                    self.torch.cat((p_value, c_value, s_value), dim=0),
                ]
            )
        metadata = self.model.cache_fuse_metadata
        metadata["check_layers"] = [reuse_layer - 1]
        metadata["recomp_ratio"] = repair_ratio
        metadata["suffix_len"] = len(self.suffix_ids)
        metadata["segment_start"] = len(self.prefix_ids)
        metadata["segment_len"] = len(self.segment_ids)
        return canonical.digest

    def execute_selected_repair(
        self,
        source_id: str,
        reuse_layer: int,
        repair_ratio: float,
        capture_logits: bool = True,
    ) -> GenerationMeasurement:
        self.stage_selected_source(
            source_id,
            reuse_layer,
            repair_ratio,
        )
        return self._generate(
            self.current_ids,
            self.max_new_tokens,
            check=True,
            capture_logits=capture_logits,
        )

    @staticmethod
    def _median_measurement(
        measurements: Sequence[GenerationMeasurement],
    ) -> GenerationMeasurement:
        if not measurements:
            raise ValueError("at least one measurement is required")
        final = measurements[-1]
        if any(
            measurement.token_ids != final.token_ids
            for measurement in measurements
        ):
            raise RuntimeError(
                "deterministic timing repetitions produced different outputs"
            )
        return replace(
            final,
            ttft_ms=statistics.median(
                measurement.ttft_ms for measurement in measurements
            ),
            total_host_ms=statistics.median(
                measurement.total_host_ms for measurement in measurements
            ),
            total_gpu_ms=statistics.median(
                measurement.total_gpu_ms for measurement in measurements
            ),
        )

    def run_source_jobs(
        self,
        source_id: str,
        jobs: Sequence[E1Job],
        attempt_by_job: Mapping[str, int],
        timing_warmup_runs: int = 0,
        timing_measurement_runs: int = 1,
    ) -> Tuple[E1Result, ...]:
        if self._closed:
            raise RuntimeError("CacheBlend case runtime is closed")
        if timing_warmup_runs < 0 or timing_measurement_runs < 1:
            raise ValueError("invalid timing repetition counts")
        if source_id not in self.sources:
            raise ValueError("job references an unknown Source")
        canonical = self.sources[source_id]
        manifest_source = self._manifest_sources[source_id]
        digest_before = canonical.digest
        pending_rows = []
        for job in sorted(
            jobs, key=lambda item: (item.reuse_layer, item.repair_ratio)
        ):
            if job.case_id != self.case.case_id or job.source_id != source_id:
                raise ValueError("job group does not match case/source runtime")
            if (
                job.case_digest != self._case_digest
                or job.content_hash != self.case.content_hash
                or job.model_signature != self.case.model_signature
                or job.source_context_id != manifest_source.context_id
                or job.segment_tokens != len(self.segment_ids)
            ):
                raise ValueError(
                    "job provenance does not bind to the loaded case/source"
                )
            for _ in range(timing_warmup_runs):
                self.execute_selected_repair(
                    canonical.source_id,
                    job.reuse_layer,
                    job.repair_ratio,
                    capture_logits=False,
                )
            measurements = []
            for _ in range(timing_measurement_runs):
                measurements.append(
                    self.execute_selected_repair(
                        canonical.source_id,
                        job.reuse_layer,
                        job.repair_ratio,
                        capture_logits=True,
                    )
                )
            generation = self._median_measurement(measurements)
            metadata = self.model.cache_fuse_metadata
            selected = int(metadata["selected_segment_tokens"])
            expected = repaired_segment_token_count(
                len(self.segment_ids), job.repair_ratio
            )
            if selected != expected:
                raise RuntimeError("CacheBlend selected an invalid C token count")
            effective = selected / float(len(self.segment_ids))
            selected_indices = tuple(
                int(value)
                for value in metadata["selected_segment_indices"]
                .detach()
                .cpu()
                .tolist()
            )
            answer_f1 = best_answer_f1(generation.text, self.case.answers)
            output_f1 = token_id_f1(generation.token_ids, self.full.token_ids)
            pending_rows.append(
                E1Result(
                    job_id=job.job_id,
                    attempt=attempt_by_job.get(job.job_id, 0),
                    status=ResultStatus.COMPLETED,
                    quality_score=answer_f1,
                    task_score_drop=self.full_answer_f1 - answer_f1,
                    token_f1=output_f1,
                    repair_latency_ms=generation.ttft_ms,
                    full_remaining_ms=self.full.ttft_ms,
                    requested_ratio=job.repair_ratio,
                    eligible_segment_tokens=len(self.segment_ids),
                    selected_segment_tokens=selected,
                    effective_ratio=effective,
                    mandatory_suffix_tokens=len(self.suffix_ids),
                    prefix_tokens=len(self.prefix_ids),
                    selected_segment_indices=selected_indices,
                    reuse_start_layer=job.reuse_layer,
                    repair_gpu_ms=generation.total_gpu_ms,
                    repair_host_ms=generation.total_host_ms,
                    full_remaining_gpu_ms=self.full.total_gpu_ms,
                    full_remaining_host_ms=self.full.total_host_ms,
                    source_digest_before=digest_before,
                    source_digest_after=digest_before,
                    output_token_ids=generation.token_ids,
                    output_hash=generation.output_hash,
                    output_text=generation.text,
                    full_output_token_ids=self.full.token_ids,
                    full_output_hash=self.full.output_hash,
                    output_ids_exact_full=(
                        generation.token_ids == self.full.token_ids
                    ),
                    logit_relative_l2=self._relative_l2(
                        generation.logits_trace, self.full.logits_trace
                    ),
                    logit_trace_mode="matched_greedy_prefix",
                    logit_positions_compared=min(
                        32,
                        len(generation.logits_trace),
                        len(self.full.logits_trace),
                    ),
                    source_k_representation="pre_rope",
                    rope_alignment_mode="cacheblend_current_org_pos",
                    causal_mask_mode="absolute_query_positions",
                    full_timing_scope="full_prefill_total_proxy",
                    repair_timing_scope=(
                        "ttft_is_prefill_to_first_token;"
                        "gpu_and_host_include_full_decode"
                    ),
                    timing_warmup_runs=timing_warmup_runs,
                    timing_measurement_runs=timing_measurement_runs,
                    code_commit=self.provenance["code_commit"],
                    environment_hash=self.provenance["environment_hash"],
                    model_revision=self.provenance["model_revision"],
                    cacheblend_commit=self.provenance["cacheblend_commit"],
                    cacheblend_patch_sha256=self.provenance[
                        "cacheblend_patch_sha256"
                    ],
                    cacheblend_tree=self.provenance["cacheblend_tree"],
                    vllm_version=self.provenance["vllm_version"],
                    torch_version=self.provenance["torch_version"],
                    cuda_version=self.provenance["cuda_version"],
                    gpu_uuid=self.provenance["gpu_uuid"],
                    finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    evidence_class="server_pilot",
                    paper_evidence=False,
                )
            )
        digest_after = self._kv_digest(canonical.layers)
        if digest_after != digest_before:
            raise RuntimeError("canonical full-prefill Source was mutated")
        self.model.old_kvs = []
        return tuple(
            E1Result(
                **{
                    **row.__dict__,
                    "source_digest_after": digest_after,
                }
            )
            for row in pending_rows
        )
