from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Tuple

from .cacheblend_v6_online_engine import TorchLayerwiseSourceLoader
from .experiment_jobs import E1Job, E1Result, ResultStatus
from .manifest import ManifestCase, ManifestSource
from .metrics import best_answer_f1, token_id_f1
from .repair_semantics import repaired_segment_token_count
from .v6_a800_executor import (
    GenerationTrace,
    RealCacheBlendA800Executor,
    RuntimeFixture,
    aggregate_relative_l2,
)


DEFAULT_INSTRUCTION = (
    "You will be asked a question after reading several passages. "
    "Answer directly using only the passages.\nPassages:\n"
)


def compose_manifest_prompt_regions(
    case: ManifestCase,
    base_prefix: Sequence[int],
    segment: Sequence[int],
    base_suffix: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Restore the complete parent document around a v7 canonical Segment."""
    prefix = tuple(int(value) for value in base_prefix)
    middle = tuple(int(value) for value in segment)
    suffix = tuple(int(value) for value in base_suffix)
    if case.protocol_version in {7, 8}:
        prefix += tuple(case.canonical_parent_left_token_ids)
        suffix = tuple(case.canonical_parent_right_token_ids) + suffix
    return prefix, middle, suffix


def manifest_segment_token_ids(case: ManifestCase, tokenizer: Any) -> Tuple[int, ...]:
    """Use exact frozen IDs in v7; preserve the legacy v6 text audit."""
    if case.protocol_version in {7, 8}:
        return tuple(int(value) for value in case.segment_token_ids)
    segment = tuple(
        int(value)
        for value in tokenizer.encode(case.segment_text, add_special_tokens=False)
    )
    if segment != case.segment_token_ids:
        raise ValueError("manifest C tokens differ from the frozen tokenizer")
    return segment


class V6H1CorrectnessError(RuntimeError):
    """Hard gate: a v6 r=1 path failed to reproduce dense execution."""


def stable_cacheblend_repair_positions(
    segment_positions: Sequence[int],
    drift_scores: Sequence[float],
    ratio: float,
    rounding_policy: str = "floor",
) -> Tuple[int, ...]:
    """Apply CacheBlend's stable largest-V-drift policy inside one C only."""

    positions = tuple(int(value) for value in segment_positions)
    scores = tuple(float(value) for value in drift_scores)
    if len(positions) != len(scores) or not positions:
        raise ValueError("drift scores must cover the complete Segment")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError("Segment positions must be sorted and unique")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("V-drift scores must be finite")
    count = repaired_segment_token_count(
        len(positions), float(ratio), rounding_policy
    )
    ranking = sorted(range(len(positions)), key=lambda index: (-scores[index], index))
    return tuple(sorted(positions[index] for index in ranking[:count]))


@dataclass(frozen=True)
class V6H1CaseFixture:
    runtime: RuntimeFixture
    prefix_tokens: int
    segment_tokens: int
    suffix_tokens: int
    source_ids: Tuple[str, ...]
    case_digest: str


class V6H1CaseRuntime:
    """Real-case H1 labels on the qualified layer-resumable v6 data plane.

    H1 intentionally fixes one Source and one boundary at a time. It does not
    run the Source selector or final admission. Unlike the legacy case runner,
    every request remains dense through ``reuse_layer - 1`` and commits the
    canonical Source immediately before ``reuse_layer``.
    """

    paper_evidence = False
    runtime_mode = "v6_resumable_h1_case_runner"

    def __init__(
        self,
        executor: RealCacheBlendA800Executor,
        case: ManifestCase,
        provenance: Mapping[str, str],
        *,
        max_new_tokens: int = 64,
        instruction: str = DEFAULT_INSTRUCTION,
        repair_rounding_policy: str = "floor",
        allowed_splits: Sequence[str] = ("pilot",),
    ) -> None:
        case.validate()
        if case.split not in set(str(value) for value in allowed_splits):
            raise ValueError("case split is outside the runtime evidence partition")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.executor = executor
        self.case = case
        self.provenance = dict(provenance)
        self.max_new_tokens = int(max_new_tokens)
        self.instruction = instruction
        if repair_rounding_policy not in {"floor", "ceil"}:
            raise ValueError("repair rounding policy must be floor or ceil")
        self.repair_rounding_policy = repair_rounding_policy
        self.fixture = self._build_fixture()
        self.source_index = {
            source_id: index for index, source_id in enumerate(self.fixture.source_ids)
        }
        eos = executor.tokenizer.eos_token_id
        self.stop_token_ids = () if eos is None else (int(eos),)
        self.full = executor._dense_generate(
            self.fixture.runtime,
            self.max_new_tokens,
            stop_token_ids=self.stop_token_ids,
        )
        self.full_text = executor.tokenizer.decode(
            self.full.token_ids, skip_special_tokens=True
        )
        self.full_answer_f1 = best_answer_f1(self.full_text, case.answers)

    def _prefix_ids(self, context: str) -> Tuple[int, ...]:
        tokenizer = self.executor.tokenizer
        body = tuple(int(value) for value in tokenizer.encode(
            self.instruction + context, add_special_tokens=False
        ))
        bos = tokenizer.bos_token_id
        return ((int(bos),) if bos is not None else ()) + body

    def _collect_layers(self, token_ids: Sequence[int]) -> Tuple[Tuple[Any, Any], ...]:
        executor = self.executor
        metadata = executor.inner_model.cache_fuse_metadata
        metadata["check"] = False
        metadata["collect"] = True
        metadata["probekv_resumable"] = False
        executor.inner_model.old_kvs = [
            [None, None] for _ in range(executor.model_spec.num_layers)
        ]
        try:
            # Canonical Source creation must observe an exact dense full
            # prefill.  Going through LLM.generate while native Prefix Cache
            # is enabled can return only the uncached suffix to the patched
            # K/V hook, especially because all RAG cases share an instruction
            # prefix.  The executor's direct model path uses an explicit
            # qualification block table and therefore cannot silently turn
            # Source materialization into a cached-prefix partial prefill.
            prompt = tuple(int(value) for value in token_ids)
            fixture = RuntimeFixture(prompt, (), (), (), ())
            tensors = executor._prepare(
                fixture,
                (),
                is_prompt=True,
                reuse=False,
                request_id="canonical-dense-exact-collect",
            )
            input_ids, positions, attention = tensors[:3]
            with executor.torch.inference_mode():
                executor.outer_model(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=executor.kv_caches,
                    attn_metadata=attention,
                )
            result = []
            for layer in executor.inner_model.layers:
                key, value = layer.self_attn.hack_kv
                if key.shape[0] != len(prompt) or value.shape[0] != len(prompt):
                    raise RuntimeError("collected KV does not cover the prompt")
                result.append((key.detach().cpu().clone(), value.detach().cpu().clone()))
            return tuple(result)
        finally:
            metadata["collect"] = False

    def _build_fixture(self) -> V6H1CaseFixture:
        tokenizer = self.executor.tokenizer
        case = self.case
        prefix = self._prefix_ids(case.current_context)
        segment = manifest_segment_token_ids(case, tokenizer)
        suffix = tuple(int(value) for value in tokenizer.encode(
            "%s\n\nQuestion: %s\nAnswer briefly:"
            % (case.current_suffix_context, case.question),
            add_special_tokens=False,
        ))
        prefix, segment, suffix = compose_manifest_prompt_regions(
            case, prefix, segment, suffix
        )
        if not suffix:
            raise ValueError("mandatory suffix S must contain tokens")
        prompt = prefix + segment + suffix
        if len(prompt) + self.max_new_tokens > self.executor.runner.model_config.max_model_len:
            raise ValueError("case exceeds the configured model context")
        start = len(prefix)
        positions = tuple(range(start, start + len(segment)))
        current_all = self._collect_layers(prompt)
        current_layers = tuple(
            (key[start:start + len(segment)], value[start:start + len(segment)])
            for key, value in current_all
        )
        variants = []
        for source in case.sources:
            historical_prefix = self._prefix_ids(source.historical_context)
            if case.protocol_version in {7, 8}:
                historical_prefix += tuple(case.canonical_parent_left_token_ids)
            combined = historical_prefix + segment
            if len(combined) + 1 > self.executor.runner.model_config.max_model_len:
                raise ValueError("historical Source exceeds the configured model context")
            collected = self._collect_layers(combined)
            source_start = len(historical_prefix)
            variants.append(tuple(
                (
                    key[source_start:source_start + len(segment)].clone(),
                    value[source_start:source_start + len(segment)].clone(),
                )
                for key, value in collected
            ))
        row = json.dumps(
            case.to_row(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return V6H1CaseFixture(
            runtime=RuntimeFixture(
                prompt_ids=prompt,
                segment_positions=(positions,),
                canonical_variants=(tuple(variants),),
                selection_variants=(tuple(
                    tuple(key.clone() for key, _ in layers)
                    for layers in variants
                ),),
                current_layers=(current_layers,),
                canonical_variant_digests=(tuple(
                    TorchLayerwiseSourceLoader._digest(
                        self.executor.torch, layers
                    )
                    for layers in variants
                ),),
                selection_state_separate_backing_verified=True,
            ),
            prefix_tokens=len(prefix),
            segment_tokens=len(segment),
            suffix_tokens=len(suffix),
            source_ids=tuple(source.source_id for source in case.sources),
            case_digest=hashlib.sha256(row).hexdigest(),
        )

    def _validate_job(self, job: E1Job, source: ManifestSource) -> None:
        expected = (
            job.case_id == self.case.case_id,
            job.case_digest == self.fixture.case_digest,
            job.content_hash == self.case.content_hash,
            job.model_signature == self.case.model_signature,
            job.source_context_id == source.context_id,
            job.segment_tokens == self.fixture.segment_tokens,
        )
        if not all(expected):
            raise ValueError("job provenance does not bind to the v6 case fixture")
        if not 2 <= job.reuse_layer <= self.executor.model_spec.num_layers:
            raise ValueError("v6 H1 boundary must leave at least one dense layer")

    def _repair_positions(self, source_index: int, layer: int, ratio: float) -> Tuple[int, ...]:
        current_value = self.fixture.runtime.current_layers[0][layer - 1][1]
        source_value = self.fixture.runtime.canonical_variants[0][source_index][layer - 1][1]
        dimensions = tuple(range(1, current_value.ndim))
        scores = (
            (current_value - source_value).float().square().sum(dim=dimensions)
            .cpu().tolist()
        )
        return stable_cacheblend_repair_positions(
            self.fixture.runtime.segment_positions[0],
            scores,
            ratio,
            self.repair_rounding_policy,
        )

    def _generate(
        self, source_index: int, reuse_layer: int, ratio: float, *, teacher: bool
    ) -> GenerationTrace:
        repair = self._repair_positions(source_index, reuse_layer, ratio)
        return self.executor._reuse_generate(
            self.fixture.runtime,
            ratio=ratio,
            token_count=(len(self.full.token_ids) if teacher else self.max_new_tokens),
            probe_layer=reuse_layer - 1,
            winner_variant=source_index,
            teacher_tokens=(self.full.token_ids if teacher else ()),
            boundary_by_segment={0: reuse_layer},
            repair_positions_by_segment={0: repair},
            model_signature=self.case.model_signature,
            stop_token_ids=self.stop_token_ids,
        )

    def run_group(
        self,
        source_id: str,
        jobs: Sequence[E1Job],
        attempts: Mapping[str, int],
    ) -> Tuple[E1Result, ...]:
        if source_id not in self.source_index or not jobs:
            raise ValueError("unknown Source or empty H1 group")
        source = next(item for item in self.case.sources if item.source_id == source_id)
        ordered = tuple(sorted(jobs, key=lambda item: item.repair_ratio))
        layers = {job.reuse_layer for job in ordered}
        if len(layers) != 1:
            raise ValueError("one v6 H1 group must use one boundary")
        for job in ordered:
            self._validate_job(job, source)
        source_index = self.source_index[source_id]
        r1_job = next((job for job in ordered if job.repair_ratio == 1.0), None)
        if r1_job is None:
            raise ValueError("v6 H1 group must contain the r=1 hard gate")
        r1_greedy = self._generate(source_index, r1_job.reuse_layer, 1.0, teacher=False)
        if r1_greedy.token_ids != self.full.token_ids:
            raise V6H1CorrectnessError(
                "r=1 resumable token IDs differ from dense reference"
            )
        r1_teacher = self._generate(source_index, r1_job.reuse_layer, 1.0, teacher=True)
        r1_l2 = aggregate_relative_l2(r1_teacher.logits, self.full.logits)
        if r1_l2 > 1e-4:
            raise V6H1CorrectnessError(
                "r=1 resumable path differs from dense reference"
            )
        if (
            not r1_greedy.source_digests_unchanged
            or not r1_greedy.absolute_union_mask_verified
            or (
                self.executor.protocol_version in {7, 8}
                and not r1_greedy.artifact_digests_unchanged
            )
        ):
            raise V6H1CorrectnessError("r=1 Source or causal-mask integrity failed")

        results = []
        for job in ordered:
            if job.repair_ratio == 1.0:
                greedy, teacher, relative = r1_greedy, r1_teacher, r1_l2
            else:
                greedy = self._generate(
                    source_index, job.reuse_layer, job.repair_ratio, teacher=False
                )
                teacher = self._generate(
                    source_index, job.reuse_layer, job.repair_ratio, teacher=True
                )
                relative = aggregate_relative_l2(teacher.logits, self.full.logits)
            if (
                not greedy.source_digests_unchanged
                or not greedy.absolute_union_mask_verified
                or (
                    self.executor.protocol_version in {7, 8}
                    and not greedy.artifact_digests_unchanged
                )
            ):
                raise RuntimeError("v6 Source or causal-mask integrity failed")
            selected = self._repair_positions(
                source_index, job.reuse_layer, job.repair_ratio
            )
            text = self.executor.tokenizer.decode(
                greedy.token_ids, skip_special_tokens=True
            )
            output_hash = hashlib.sha256(json.dumps(
                {"token_ids": list(greedy.token_ids), "text": text},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            full_hash = hashlib.sha256(json.dumps(
                {"token_ids": list(self.full.token_ids), "text": self.full_text},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            canonical = self.fixture.runtime.canonical_variants[0][source_index]
            digest = TorchLayerwiseSourceLoader._digest(self.executor.torch, canonical)
            row = E1Result(
                job_id=job.job_id,
                attempt=attempts.get(job.job_id, 0),
                status=ResultStatus.COMPLETED,
                quality_score=best_answer_f1(text, self.case.answers),
                task_score_drop=self.full_answer_f1 - best_answer_f1(text, self.case.answers),
                token_f1=token_id_f1(greedy.token_ids, self.full.token_ids),
                repair_latency_ms=greedy.host_ms,
                full_remaining_ms=self.full.host_ms,
                requested_ratio=job.repair_ratio,
                eligible_segment_tokens=self.fixture.segment_tokens,
                selected_segment_tokens=len(selected),
                effective_ratio=len(selected) / float(self.fixture.segment_tokens),
                mandatory_suffix_tokens=self.fixture.suffix_tokens,
                prefix_tokens=self.fixture.prefix_tokens,
                selected_segment_indices=selected,
                reuse_start_layer=job.reuse_layer,
                repair_gpu_ms=greedy.gpu_ms,
                repair_host_ms=greedy.host_ms,
                full_remaining_gpu_ms=self.full.gpu_ms,
                full_remaining_host_ms=self.full.host_ms,
                source_digest_before=digest,
                source_digest_after=digest,
                output_token_ids=greedy.token_ids,
                output_hash=output_hash,
                output_text=text,
                full_output_token_ids=self.full.token_ids,
                full_output_hash=full_hash,
                output_ids_exact_full=greedy.token_ids == self.full.token_ids,
                logit_relative_l2=relative,
                logit_trace_mode="teacher_forced_dense_reference",
                logit_positions_compared=len(self.full.logits),
                source_k_representation="pre_rope",
                rope_alignment_mode="cacheblend_current_absolute_positions",
                causal_mask_mode="v6_absolute_staggered_union",
                full_timing_scope="dense_prefill_to_generation_endpoint",
                repair_timing_scope="v6_resumable_prefill_to_generation_endpoint",
                code_commit=self.provenance["code_commit"],
                environment_hash=self.provenance["environment_hash"],
                model_revision=self.provenance["model_revision"],
                cacheblend_commit=self.provenance["cacheblend_commit"],
                cacheblend_patch_sha256=self.provenance["cacheblend_patch_sha256"],
                cacheblend_tree=self.provenance["cacheblend_tree"],
                vllm_version=self.provenance["vllm_version"],
                torch_version=self.provenance["torch_version"],
                cuda_version=self.provenance["cuda_version"],
                gpu_uuid=self.provenance["gpu_uuid"],
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                evidence_class="server_pilot",
                paper_evidence=False,
            )
            row.validate()
            results.append(row)
        return tuple(results)


class V7H1CaseRuntime(V6H1CaseRuntime):
    """Single-Artifact v7 H1 sentinel labels with conservative ceil repair."""

    runtime_mode = "v7_single_artifact_h1_case_runner"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["repair_rounding_policy"] = "ceil"
        super().__init__(*args, **kwargs)
        if self.executor.protocol_version != 7:
            raise ValueError("v7 H1 runtime requires a protocol-v7 executor")
        if self.case.protocol_version != 7 or not self.case.reuse_content_key:
            raise ValueError("v7 H1 requires canonical Segment provenance")


class V8H1CaseRuntime(V6H1CaseRuntime):
    """v8 offline repair-grid diagnostics after Profile-bound qualification."""

    runtime_mode = "v8_training_free_offline_h1_case_runner"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["repair_rounding_policy"] = "ceil"
        super().__init__(*args, **kwargs)
        if self.executor.protocol_version != 8:
            raise ValueError("v8 H1 runtime requires a protocol-v8 executor")
        if self.case.protocol_version != 8 or not self.case.reuse_content_key:
            raise ValueError("v8 H1 requires canonical Segment provenance")
