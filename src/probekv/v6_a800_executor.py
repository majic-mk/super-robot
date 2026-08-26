from __future__ import annotations

import hashlib
import inspect
import math
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .cacheblend_v6_online_engine import (
    CacheBlendV6OnlineEngine,
    CacheBlendV7OnlineEngine,
    TorchLayerwiseSourceLoader,
)
from .contracts import KVLocation
from .model_adapters import ResumableModelSpec
from .v6_a800_jobs import V6A800Job, V6A800JobKind
from .v6_qualification_worker import QualificationJobResult
from .v7_contracts import CanonicalKVArtifact, PhysicalReplica, ReplicaLocator
from .v7_planner import repair_token_count


@dataclass(frozen=True)
class RuntimeFixture:
    prompt_ids: Tuple[int, ...]
    segment_positions: Tuple[Tuple[int, ...], ...]
    # segment -> historical variant -> Transformer layer -> (pre-RoPE K, V)
    canonical_variants: Tuple[
        Tuple[Tuple[Tuple[Any, Any], ...], ...], ...
    ]
    # segment -> Transformer layer -> current request (pre-RoPE K, V)
    current_layers: Tuple[Tuple[Tuple[Any, Any], ...], ...]
    exact_prefix_tokens: int = 0
    exact_prefix_layers: Tuple[Tuple[Any, Any], ...] = ()


@dataclass(frozen=True)
class GenerationTrace:
    token_ids: Tuple[int, ...]
    logits: Tuple[Any, ...]
    gpu_ms: float
    host_ms: float
    source_digests_unchanged: bool = True
    absolute_union_mask_verified: bool = True
    prefix_shadow_digest_before: str = ""
    prefix_shadow_digest_after: str = ""
    prefix_rows_excluded_from_repair: int = 0
    prefix_active_positions_valid: bool = True
    artifact_digests_unchanged: bool = True


def aggregate_relative_l2(observed: Sequence[Any], reference: Sequence[Any]) -> float:
    if not observed or len(observed) != len(reference):
        raise ValueError("logit traces must be non-empty and equally long")
    numerator = 0.0
    denominator = 0.0
    for left, right in zip(observed, reference):
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError("logit trace shapes differ")
        delta = left.float() - right.float()
        numerator += float((delta * delta).sum().item())
        denominator += float((right.float() * right.float()).sum().item())
    return math.sqrt(numerator / max(denominator, 1e-30))


class RealCacheBlendA800Executor:
    """Real A800 executor for the frozen v6 qualification matrix.

    Heavy vLLM/PyTorch imports and model loading happen only at construction,
    so the package remains testable on CPU-only development machines. The
    executor runs the patched layer-resumable model for correctness and union
    repair jobs, and uses CUDA kernels for summary, comparison and transfer
    microbenchmarks. It never emits paper evidence.
    """

    concrete_engine_hook = True

    def __init__(
        self,
        *,
        model_path: str,
        model_spec: ResumableModelSpec,
        max_model_len: int = 512,
        gpu_memory_utilization: float = 0.60,
        sentinel_tokens: int = 32,
        expected_cacheblend_tree: str = "",
        engine_class: Any = CacheBlendV6OnlineEngine,
        protocol_version: int = 6,
    ) -> None:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.sequence import SequenceData, SequenceGroupMetadata

        if not torch.cuda.is_available():
            raise RuntimeError("real A800 qualification requires CUDA")
        self.torch = torch
        self.SamplingParams = SamplingParams
        self.SequenceData = SequenceData
        self.SequenceGroupMetadata = SequenceGroupMetadata
        self.model_spec = model_spec
        if protocol_version not in {6, 7}:
            raise ValueError("qualification executor supports protocol v6 or v7")
        if protocol_version == 7 and engine_class is not CacheBlendV7OnlineEngine:
            raise ValueError("protocol v7 requires CacheBlendV7OnlineEngine")
        self.protocol_version = protocol_version
        self.engine_class = engine_class
        self.repair_rounding_policy = "ceil" if protocol_version == 7 else "floor"
        self.adapter_name = model_spec.adapter_name
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            enable_prefix_caching=True,
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        worker = self.llm.llm_engine.model_executor.driver_worker
        self.runner = worker.model_runner
        self.outer_model = self.runner.model
        self.inner_model = self.outer_model.model
        self.runtime_provenance = self._runtime_provenance()
        if (
            expected_cacheblend_tree
            and self.runtime_provenance["cacheblend_tree"]
            != expected_cacheblend_tree
        ):
            raise RuntimeError(
                "imported vLLM model code differs from audited CacheBlend tree"
            )
        self.kv_caches = worker.cache_engine.gpu_cache
        if len(self.kv_caches) != model_spec.num_layers:
            raise RuntimeError("vLLM KV cache layer count differs from adapter")
        self.source_loader = TorchLayerwiseSourceLoader(torch)
        self.fixtures: Dict[Tuple[int, int, int], RuntimeFixture] = {}
        self.sentinel = self._run_sentinel(int(sentinel_tokens))

    def _runtime_provenance(self) -> Dict[str, Any]:
        from importlib.metadata import version

        model_file = Path(inspect.getfile(type(self.inner_model))).resolve()
        cacheblend = next(
            (
                parent for parent in model_file.parents
                if (parent / ".git").is_dir()
                and (parent / "vllm_blend").is_dir()
            ),
            None,
        )
        if cacheblend is None:
            raise RuntimeError(
                "imported model is not inside an auditable CacheBlend checkout"
            )
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=str(cacheblend), text=True
        ).strip()
        device = self.torch.cuda.current_device()
        return {
            "torch": self.torch.__version__,
            "vllm": version("vllm"),
            "xformers": version("xformers"),
            "torch_cuda": self.torch.version.cuda,
            "gpu_name": self.torch.cuda.get_device_name(device),
            "compute_capability": list(
                self.torch.cuda.get_device_capability(device)
            ),
            "model_module_file": str(model_file),
            "cacheblend_checkout": str(cacheblend),
            "cacheblend_tree": tree,
        }

    def capabilities(self) -> Mapping[str, bool]:
        return self.engine_class.capabilities()

    def _exact_prefix_ids(self, token_count: int) -> List[int]:
        if token_count <= 0:
            return []
        seed = list(self.tokenizer.encode(
            "ProbeKV exact native prefix qualification context. ",
            add_special_tokens=True,
        ))
        continuation = list(self.tokenizer.encode(
            "Stable block-aligned prefix evidence. ",
            add_special_tokens=False,
        ))
        if not seed or not continuation:
            raise RuntimeError("tokenizer could not build a Prefix Cache fixture")
        result = list(seed)
        while len(result) < token_count:
            result.extend(continuation)
        return result[:token_count]

    def _fixture(
        self,
        segment_count: int,
        stored_variants: int = 1,
        exact_prefix_tokens: int = 0,
    ) -> RuntimeFixture:
        cache_key = (
            int(segment_count), int(stored_variants), int(exact_prefix_tokens)
        )
        cached = self.fixtures.get(cache_key)
        if cached is not None:
            return cached
        if segment_count < 1:
            raise ValueError("fixture requires at least one Segment")
        if not 1 <= stored_variants <= 16:
            raise ValueError("qualification fixtures support 1-16 variants")
        encode = self.tokenizer.encode
        prefix_ids = self._exact_prefix_ids(int(exact_prefix_tokens))
        if prefix_ids:
            current = list(prefix_ids)
            current.extend(encode(
                " Dense region after the exact cached prefix.",
                add_special_tokens=False,
            ))
        else:
            current = list(encode(
                "Current request context for ProbeKV qualification.",
                add_special_tokens=True,
            ))
        current_positions: List[Tuple[int, ...]] = []
        segments: List[List[int]] = []
        bridges: List[List[int]] = []
        for index in range(segment_count):
            bridge = list(encode(
                " Dense bridge %d." % index, add_special_tokens=False
            ))
            segment = list(encode(
                " Exact reusable segment %d contains a stable fact." % index,
                add_special_tokens=False,
            ))
            current.extend(bridge)
            current_start = len(current)
            current.extend(segment)
            current_positions.append(tuple(range(
                current_start, current_start + len(segment)
            )))
            bridges.append(bridge)
            segments.append(segment)
        current.extend(encode(
            " Question: summarize the stable facts.",
            add_special_tokens=False,
        ))
        if len(current) + 32 > min(
            self.model_spec.max_context_tokens, self.runner.model_config.max_model_len
        ):
            raise ValueError("qualification fixture exceeds model context")

        metadata = self.inner_model.cache_fuse_metadata
        metadata["check"] = False
        metadata["collect"] = True
        metadata["probekv_resumable"] = False
        per_segment: List[List[Tuple[Tuple[Any, Any], ...]]] = [
            [] for _ in range(segment_count)
        ]
        current_layers: List[List[Tuple[Any, Any]]] = [
            [] for _ in range(segment_count)
        ]
        exact_prefix_layers: List[Tuple[Any, Any]] = []
        try:
            for variant in range(stored_variants):
                historical = list(encode(
                    "Historical variant %d has %d distinct lead tokens. %s"
                    % (variant, variant + 1, "context " * (variant + 1)),
                    add_special_tokens=True,
                ))
                historical_positions: List[Tuple[int, ...]] = []
                for bridge, segment in zip(bridges, segments):
                    historical.extend(bridge)
                    start = len(historical)
                    historical.extend(segment)
                    historical_positions.append(tuple(range(
                        start, start + len(segment)
                    )))
                self.llm.generate(
                    prompt_token_ids=[historical],
                    sampling_params=self.SamplingParams(
                        temperature=0, max_tokens=1
                    ),
                    use_tqdm=False,
                )
                variant_layers: List[List[Tuple[Any, Any]]] = [
                    [] for _ in range(segment_count)
                ]
                for layer in self.inner_model.layers:
                    key, value = layer.self_attn.hack_kv
                    for index, positions in enumerate(historical_positions):
                        start, end = positions[0], positions[-1] + 1
                        variant_layers[index].append((
                            key[start:end].detach().cpu(),
                            value[start:end].detach().cpu(),
                        ))
                for index, layers in enumerate(variant_layers):
                    per_segment[index].append(tuple(layers))
            self.llm.generate(
                prompt_token_ids=[current],
                sampling_params=self.SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
            for layer in self.inner_model.layers:
                key, value = layer.self_attn.hack_kv
                if prefix_ids:
                    exact_prefix_layers.append((
                        key[:len(prefix_ids)].detach().clone(),
                        value[:len(prefix_ids)].detach().clone(),
                    ))
                for index, positions in enumerate(current_positions):
                    start, end = positions[0], positions[-1] + 1
                    current_layers[index].append((
                        key[start:end].detach().cpu(),
                        value[start:end].detach().cpu(),
                    ))
        finally:
            metadata["collect"] = False
        fixture = RuntimeFixture(
            prompt_ids=tuple(int(v) for v in current),
            segment_positions=tuple(current_positions),
            canonical_variants=tuple(tuple(rows) for rows in per_segment),
            current_layers=tuple(tuple(rows) for rows in current_layers),
            exact_prefix_tokens=len(prefix_ids),
            exact_prefix_layers=tuple(exact_prefix_layers),
        )
        self.fixtures[cache_key] = fixture
        return fixture

    def _block_table(self, token_capacity: int, reuse: bool) -> List[int]:
        block_size = int(self.runner.block_size)
        needed = (token_capacity + block_size - 1) // block_size
        total = int(self.kv_caches[0].shape[1])
        reserve = max(needed + 2, 64)
        base = total - reserve if reuse else total - 2 * reserve
        if base < 0 or base + needed > total:
            raise RuntimeError("vLLM KV cache has insufficient qualification blocks")
        return list(range(base, base + needed))

    def _prepare(
        self,
        fixture: RuntimeFixture,
        output_ids: Sequence[int],
        *,
        is_prompt: bool,
        reuse: bool,
        request_id: str,
    ) -> Tuple[Any, ...]:
        blocks = self._block_table(
            len(fixture.prompt_ids) + max(32, len(output_ids)) + 2,
            reuse,
        )
        data = self.SequenceData(
            list(fixture.prompt_ids), list(int(v) for v in output_ids)
        )
        group = self.SequenceGroupMetadata(
            request_id=request_id,
            is_prompt=is_prompt,
            seq_data={0: data},
            sampling_params=self.SamplingParams(
                temperature=0, max_tokens=max(32, len(output_ids) + 1)
            ),
            block_tables={0: blocks},
            token_chunk_size=(len(fixture.prompt_ids) if is_prompt else 1),
        )
        return self.runner.prepare_input_tensors([group], self.kv_caches)

    def _logits(self, hidden: Any, sampling: Any) -> Any:
        selected = sampling.selected_token_indices
        original = selected.clone()
        selected[0] = hidden.shape[0] - 1
        try:
            return self.outer_model.compute_logits(hidden, sampling)
        finally:
            selected.copy_(original)

    def _decode_from_prefill(
        self,
        fixture: RuntimeFixture,
        hidden: Any,
        sampling: Any,
        *,
        reuse: bool,
        token_count: int,
        teacher_tokens: Sequence[int] = (),
        stop_token_ids: Sequence[int] = (),
    ) -> Tuple[Tuple[int, ...], Tuple[Any, ...]]:
        predicted: List[int] = []
        context: List[int] = []
        traces: List[Any] = []
        logits = self._logits(hidden, sampling)
        traces.append(logits.detach().float().cpu())
        predicted.append(int(logits.argmax().item()))
        context.append(
            int(teacher_tokens[0]) if teacher_tokens else predicted[-1]
        )
        stops = {int(value) for value in stop_token_ids}
        if stops and context[-1] in stops:
            return tuple(predicted), tuple(traces)
        for step in range(1, token_count):
            tensors = self._prepare(
                fixture, context, is_prompt=False, reuse=reuse,
                request_id="decode-%s-%d" % ("reuse" if reuse else "dense", step),
            )
            input_ids, positions, metadata, sample = tensors[:4]
            hidden = self.outer_model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=self.kv_caches,
                attn_metadata=metadata,
            )
            logits = self._logits(hidden, sample)
            traces.append(logits.detach().float().cpu())
            predicted.append(int(logits.argmax().item()))
            context.append(
                int(teacher_tokens[step]) if teacher_tokens else predicted[-1]
            )
            if stops and context[-1] in stops:
                break
        return tuple(predicted), tuple(traces)

    def _dense_generate(
        self,
        fixture: RuntimeFixture,
        token_count: int,
        stop_token_ids: Sequence[int] = (),
    ) -> GenerationTrace:
        metadata = self.inner_model.cache_fuse_metadata
        metadata["check"] = False
        metadata["collect"] = False
        metadata["probekv_resumable"] = False
        # The pinned CacheBlend forward indexes one old-KV slot at every layer
        # even for status=full-prefill. Restore its canonical dense sentinels;
        # an empty list fails before the attention path can ignore old KV.
        self.inner_model.old_kvs = [
            [None, None] for _ in range(self.model_spec.num_layers)
        ]
        tensors = self._prepare(
            fixture, (), is_prompt=True, reuse=False, request_id="dense-prefill"
        )
        input_ids, positions, attention, sampling = tensors[:4]
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        self.torch.cuda.synchronize()
        host_start = time.perf_counter()
        start.record()
        with self.torch.inference_mode():
            hidden = self.outer_model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=self.kv_caches,
                attn_metadata=attention,
            )
            token_ids, logits = self._decode_from_prefill(
                fixture, hidden, sampling, reuse=False, token_count=token_count,
                stop_token_ids=stop_token_ids,
            )
        end.record()
        end.synchronize()
        return GenerationTrace(
            token_ids,
            logits,
            float(start.elapsed_time(end)),
            (time.perf_counter() - host_start) * 1000.0,
        )

    def _repair_positions(
        self, positions: Sequence[int], ratio: float
    ) -> Tuple[int, ...]:
        if ratio <= 0:
            return ()
        if ratio >= 1:
            return tuple(positions)
        count = repair_token_count(len(positions), ratio)
        if self.repair_rounding_policy == "floor":
            count = int(len(positions) * ratio)
        return tuple(positions[:count])

    def _reuse_generate(
        self,
        fixture: RuntimeFixture,
        ratio: float,
        token_count: int,
        probe_layer: int,
        winner_variant: int = 0,
        teacher_tokens: Sequence[int] = (),
        boundary_by_segment: Mapping[int, int] | None = None,
        repair_positions_by_segment: Mapping[int, Sequence[int]] | None = None,
        model_signature: str = "a800-qualification",
        stop_token_ids: Sequence[int] = (),
        exact_prefix_tokens: int = 0,
        exact_prefix_layers: Sequence[Tuple[Any, Any]] = (),
    ) -> GenerationTrace:
        prefix_tokens = int(exact_prefix_tokens)
        if prefix_tokens < 0 or prefix_tokens > len(fixture.prompt_ids):
            raise ValueError("invalid exact Prefix Cache token count")
        prefix_layers = tuple(exact_prefix_layers)
        if prefix_tokens and len(prefix_layers) != self.model_spec.num_layers:
            raise ValueError("exact Prefix Cache shadow is incomplete")
        if not prefix_tokens and prefix_layers:
            raise ValueError("prefix shadow requires a native Prefix Cache hit")
        prefix_digest_before = (
            TorchLayerwiseSourceLoader._digest(self.torch, prefix_layers)
            if prefix_layers else ""
        )
        tensors = self._prepare(
            fixture, (), is_prompt=True, reuse=True, request_id="reuse-prefill"
        )
        attention, sampling = tensors[2], tensors[3]
        engine = self.engine_class(
            inner_model=self.inner_model,
            model_spec=self.model_spec,
            source_loader=self.source_loader,
        )
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        self.torch.cuda.synchronize()
        host_start = time.perf_counter()
        start.record()
        with self.torch.inference_mode():
            engine.begin_prefill(
                model_signature=model_signature,
                token_ids=fixture.prompt_ids[prefix_tokens:],
                absolute_positions=tuple(
                    range(prefix_tokens, len(fixture.prompt_ids))
                ),
                exact_prefix_tokens=prefix_tokens,
                exact_prefix_layers=prefix_layers,
                attention_metadata=attention,
                working_kv=self.kv_caches,
            )
            first_probe = min(max(1, int(probe_layer)), self.model_spec.num_layers - 1)
            engine.advance_to_layer(first_probe)
            tickets = []
            for index, positions in enumerate(fixture.segment_positions):
                variants = fixture.canonical_variants[index]
                if not 0 <= winner_variant < len(variants):
                    raise ValueError("locked winner variant is unavailable")
                source_id = "s%d-v%d" % (index, winner_variant)
                canonical_layers = variants[winner_variant]
                if self.protocol_version == 7:
                    logical = TorchLayerwiseSourceLoader._digest(
                        self.torch, canonical_layers
                    )
                    first_key = canonical_layers[0][0]
                    artifact = CanonicalKVArtifact(
                        artifact_id="artifact-%s" % source_id,
                        source_variant_id=source_id,
                        generation=1,
                        parent_source_state_digest=logical,
                        artifact_logical_digest=logical,
                        artifact_bytes_digest=logical,
                        num_layers=len(canonical_layers),
                        num_kv_heads=int(first_key.shape[1]),
                        head_dim=int(first_key.shape[2]),
                    )
                    replica = PhysicalReplica(
                        replica_id="replica-%s-cpu" % source_id,
                        artifact_id=artifact.artifact_id,
                        generation=1,
                        tier=KVLocation.PINNED_CPU,
                        logical_digest=logical,
                        bytes_digest=logical,
                        size_bytes=sum(
                            tensor.numel() * tensor.element_size()
                            for key, value in canonical_layers
                            for tensor in (key, value)
                        ),
                        locator=ReplicaLocator(
                            value="qualification-pinned-cpu",
                            layout_signature="contiguous-bf16",
                        ),
                    )
                    tickets.append(engine.start_artifact_replica_prefetch(
                        segment_id="c%d" % index,
                        source_variant_id=source_id,
                        artifact=artifact,
                        replica=replica,
                        canonical_layers=canonical_layers,
                        segment_positions=positions,
                    ))
                else:
                    tickets.append(engine.start_winner_prefetch(
                        segment_id="c%d" % index,
                        source_id=source_id,
                        canonical_layers=canonical_layers,
                        segment_positions=positions,
                    ))
            for ticket in tickets:
                for event in ticket.layer_events.values():
                    event.synchronize()
            base = first_probe + 1
            boundaries = dict(boundary_by_segment or {
                index: min(base + index % 3, self.model_spec.num_layers)
                for index in range(len(fixture.segment_positions))
            })
            expected_segments = set(range(len(fixture.segment_positions)))
            if set(boundaries) != expected_segments:
                raise ValueError("reuse boundaries must cover every Segment")
            if any(
                not first_probe < int(boundary) <= self.model_spec.num_layers
                for boundary in boundaries.values()
            ):
                raise ValueError("reuse boundary must follow the dense probe")
            supplied_positions = dict(repair_positions_by_segment or {})
            if supplied_positions and set(supplied_positions) != expected_segments:
                raise ValueError("repair positions must cover every Segment")
            for boundary in sorted(set(boundaries.values())):
                if engine.session.current_layer < boundary - 1:
                    engine.advance_to_layer(boundary - 1)
                for index, positions in enumerate(fixture.segment_positions):
                    if boundaries[index] != boundary:
                        continue
                    repair_positions = tuple(
                        int(value) for value in supplied_positions.get(
                            index, self._repair_positions(positions, ratio)
                        )
                    )
                    if not set(repair_positions).issubset(set(positions)):
                        raise ValueError("repair position lies outside its Segment")
                    engine.commit_ready_segment(
                        segment_id="c%d" % index,
                        boundary=boundary,
                        segment_positions=positions,
                        repair_positions=repair_positions,
                        scheduler_boundary=boundary,
                    )
                engine.advance_to_layer(boundary)
            hidden = engine.finish_prefill()
            token_ids, logits = self._decode_from_prefill(
                fixture,
                hidden,
                sampling,
                reuse=True,
                token_count=token_count,
                teacher_tokens=teacher_tokens,
                stop_token_ids=stop_token_ids,
            )
        end.record()
        end.synchronize()
        expected_masks = len(engine.session.layer_audit)
        verified_masks = sum(
            bool(row["union_mask_digest"]) for row in engine.session.layer_audit
        )
        prefix_digest_after = (
            TorchLayerwiseSourceLoader._digest(self.torch, prefix_layers)
            if prefix_layers else ""
        )
        repair_rows = {
            position
            for commit in engine.session.commits.values()
            for position in commit.repair_positions
        }
        prefix_rows_excluded = sum(
            position not in repair_rows for position in range(prefix_tokens)
        )
        prefix_active_positions_valid = all(
            position >= prefix_tokens
            for row in engine.session.layer_audit
            for key in ("active_before", "active_after")
            for position in row[key]
        )
        return GenerationTrace(
            token_ids,
            logits,
            float(start.elapsed_time(end)),
            (time.perf_counter() - host_start) * 1000.0,
            source_digests_unchanged=all(
                ticket.source_digest_before == ticket.source_digest_after
                for ticket in tickets
            ),
            absolute_union_mask_verified=(verified_masks == expected_masks),
            prefix_shadow_digest_before=prefix_digest_before,
            prefix_shadow_digest_after=prefix_digest_after,
            prefix_rows_excluded_from_repair=prefix_rows_excluded,
            prefix_active_positions_valid=prefix_active_positions_valid,
            artifact_digests_unchanged=(
                all(engine.audit.artifact_digest_unchanged_by_segment.values())
                if self.protocol_version == 7
                else True
            ),
        )

    def _run_sentinel(self, token_count: int) -> Dict[str, Any]:
        if token_count != 32:
            raise ValueError("qualification sentinel is frozen at 32 tokens")
        fixture = self._fixture(1)
        dense = self._dense_generate(fixture, token_count)
        reuse = self._reuse_generate(
            fixture,
            ratio=1.0,
            token_count=token_count,
            probe_layer=1,
            teacher_tokens=dense.token_ids,
        )
        relative = aggregate_relative_l2(reuse.logits, dense.logits)
        result = {
            "paper_evidence": False,
            "locked_test_accessed": False,
            "r1_dense_token_ids_equal": reuse.token_ids == dense.token_ids,
            "max_teacher_forced_logit_relative_l2": relative,
            "canonical_source_digests_unchanged": reuse.source_digests_unchanged,
            "absolute_union_mask_verified": reuse.absolute_union_mask_verified,
            "artifact_digests_unchanged": reuse.artifact_digests_unchanged,
            "dense_token_ids": list(dense.token_ids),
            "reuse_token_ids": list(reuse.token_ids),
            "dense_gpu_ms": dense.gpu_ms,
            "reuse_gpu_ms": reuse.gpu_ms,
        }
        if not result["r1_dense_token_ids_equal"] or relative > 1e-4:
            raise RuntimeError("r=1 A800 sentinel differs from dense reference")
        if not reuse.source_digests_unchanged or not reuse.absolute_union_mask_verified:
            raise RuntimeError("r=1 A800 sentinel failed Source/mask integrity")
        return result

    def run_native_prefix_cache_sentinel(
        self,
        *,
        prefix_tokens: int = 192,
        token_count: int = 32,
    ) -> Dict[str, Any]:
        """Exercise vLLM block reuse and the prefix-aware r=1 data plane."""

        if token_count != 32:
            raise ValueError("native prefix sentinel is frozen at 32 tokens")
        block_manager = self.llm.llm_engine.scheduler.block_manager
        block_size = int(block_manager.block_size)
        if not bool(getattr(block_manager, "enable_caching", False)):
            raise RuntimeError("vLLM native Prefix Cache is not enabled")
        if prefix_tokens < 192 or prefix_tokens % block_size:
            raise ValueError("native prefix must be >=192 and block aligned")

        # Building this fixture performs the first full request and leaves its
        # complete prefix blocks in vLLM's cached block allocator.
        fixture = self._fixture(1, 1, exact_prefix_tokens=prefix_tokens)
        prefix = list(fixture.prompt_ids[:prefix_tokens])
        different_tail = list(self.tokenizer.encode(
            " A deliberately different continuation validates native block "
            "reuse without relying on TTFT.",
            add_special_tokens=False,
        ))
        if not different_tail:
            raise RuntimeError("tokenizer produced an empty second continuation")

        captured_block_ids: List[int] = []
        scheduler = self.llm.llm_engine.scheduler
        original_schedule = scheduler.schedule

        def schedule_with_evidence():
            metadata, outputs = original_schedule()
            for row in metadata:
                if row.is_prompt:
                    captured_block_ids.extend(
                        int(value) for value in row.computed_block_nums
                    )
            return metadata, outputs

        scheduler.schedule = schedule_with_evidence
        try:
            self.llm.generate(
                prompt_token_ids=[prefix + different_tail],
                sampling_params=self.SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
        finally:
            scheduler.schedule = original_schedule
        cached_blocks = len(tuple(dict.fromkeys(captured_block_ids)))
        cached_tokens = cached_blocks * block_size
        if cached_tokens > prefix_tokens:
            raise RuntimeError("vLLM reused blocks beyond the exact prefix")

        shadows = fixture.exact_prefix_layers
        expected_row_values = int(self.kv_caches[0].shape[-1]) // block_size
        geometry = bool(shadows) and all(
            key.shape[0] == cached_tokens
            and value.shape[0] == cached_tokens
            and tuple(key.shape[1:]) == tuple(shadows[0][0].shape[1:])
            and tuple(value.shape[1:]) == tuple(shadows[0][1].shape[1:])
            and key.dtype == shadows[0][0].dtype
            and value.dtype == shadows[0][1].dtype
            and key.device == shadows[0][0].device
            and value.device == shadows[0][1].device
            and key[0].numel() == expected_row_values
            and value[0].numel() == expected_row_values
            for key, value in shadows
        )
        dense = self._dense_generate(fixture, token_count)
        reuse = self._reuse_generate(
            fixture,
            ratio=1.0,
            token_count=token_count,
            probe_layer=1,
            teacher_tokens=dense.token_ids,
            exact_prefix_tokens=cached_tokens,
            exact_prefix_layers=tuple(
                (key[:cached_tokens], value[:cached_tokens])
                for key, value in shadows
            ),
        )
        relative_l2 = aggregate_relative_l2(reuse.logits, dense.logits)
        return {
            "paper_evidence": False,
            "locked_test_accessed": False,
            "hit_evidence_source": "vllm_scheduler_computed_block_nums",
            "timing_inference_used": False,
            "requested_prefix_tokens": prefix_tokens,
            "native_prefix_cache_hit": cached_blocks >= 1,
            "cached_prefix_blocks": cached_blocks,
            "cached_prefix_tokens": cached_tokens,
            "cached_prefix_block_ids": captured_block_ids,
            "block_size": block_size,
            "model_num_layers": self.model_spec.num_layers,
            "prefix_shadow_layers": len(shadows),
            "prefix_shadow_rows": cached_tokens,
            "prefix_shadow_dtype": (
                str(shadows[0][0].dtype) if shadows else ""
            ),
            "prefix_shadow_device": (
                shadows[0][0].device.type if shadows else ""
            ),
            "prefix_shadow_geometry_valid": geometry,
            "prefix_shadow_digest_before": reuse.prefix_shadow_digest_before,
            "prefix_shadow_digest_after": reuse.prefix_shadow_digest_after,
            "active_positions_start_after_prefix": (
                reuse.prefix_active_positions_valid
            ),
            "prefix_rows_excluded_from_repair": (
                reuse.prefix_rows_excluded_from_repair
            ),
            "prefix_rows_in_repair_mask": sum(
                position < cached_tokens
                for positions in fixture.segment_positions
                for position in positions
            ),
            "prefix_rows_in_source_comparison": sum(
                position < cached_tokens
                for positions in fixture.segment_positions
                for position in positions
            ),
            "combined_prefix_r1_reuse_exercised": True,
            "dense_token_ids_equal": reuse.token_ids == dense.token_ids,
            "logit_relative_l2": relative_l2,
            "dense_token_ids": list(dense.token_ids),
            "reuse_token_ids": list(reuse.token_ids),
            "dense_gpu_ms": dense.gpu_ms,
            "reuse_gpu_ms": reuse.gpu_ms,
            "cuda_event_timing": dense.gpu_ms > 0 and reuse.gpu_ms > 0,
        }

    def _summary_tensors(
        self,
        fixture: RuntimeFixture,
        layer: int,
        compared_variants: int,
    ) -> Tuple[Any, Any]:
        layer_index = int(layer) - 1
        current_rows = []
        source_rows: List[List[Any]] = [
            [] for _ in range(compared_variants)
        ]
        for segment_index in range(len(fixture.segment_positions)):
            key, value = fixture.current_layers[segment_index][layer_index]
            current_rows.append(
                self.torch.cat((key, value), dim=-1).to("cuda")
            )
            for variant in range(compared_variants):
                source_key, source_value = (
                    fixture.canonical_variants[segment_index][variant][layer_index]
                )
                source_rows[variant].append(
                    self.torch.cat((source_key, source_value), dim=-1).to("cuda")
                )
        current = self.torch.cat(current_rows, dim=0).contiguous()
        sources = self.torch.stack(
            [self.torch.cat(rows, dim=0) for rows in source_rows], dim=0
        ).contiguous() if source_rows else current.new_empty((0,) + current.shape)
        return current, sources

    @staticmethod
    def _cuda_compare(current: Any, sources: Any) -> float:
        dimensions = tuple(range(1, sources.ndim))
        numerator = (
            (sources - current.unsqueeze(0)).float().square().sum(dim=dimensions)
        )
        denominator = current.float().square().sum().clamp_min(1e-12)
        value = (numerator / denominator).sqrt()
        return float(value.min().item())

    def _timed_operation(self, operation: Any) -> Tuple[float, float]:
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        self.torch.cuda.synchronize()
        host = time.perf_counter()
        start.record()
        operation()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)), (time.perf_counter() - host) * 1000.0

    def execute(self, job: V6A800Job) -> QualificationJobResult:
        needs_model = True
        fixture = self._fixture(
            job.segment_count,
            job.stored_variants if job.kind in (
                V6A800JobKind.CORRECTNESS,
                V6A800JobKind.CANDIDATE_COMPARE,
                V6A800JobKind.MULTISEGMENT_COMPARE,
            ) else 1,
        ) if needs_model else None
        winner_variant = (
            job.stored_variants - 1
            if job.kind is V6A800JobKind.CORRECTNESS else 0
        )
        integrity: GenerationTrace | None = None
        r1_equal = bool(self.sentinel["r1_dense_token_ids_equal"])
        relative_l2 = float(
            self.sentinel["max_teacher_forced_logit_relative_l2"]
        )
        if job.kind in (V6A800JobKind.CORRECTNESS, V6A800JobKind.UNION_REPAIR):
            if fixture is None:
                raise RuntimeError("repair qualification lacks a model fixture")
            dense = self._dense_generate(fixture, 1) if job.repair_ratio == 1 else None
            integrity = self._reuse_generate(
                fixture,
                ratio=job.repair_ratio,
                token_count=1,
                probe_layer=job.probe_layer,
                winner_variant=winner_variant,
                teacher_tokens=dense.token_ids if dense is not None else (),
            )
            if dense is not None:
                r1_equal = integrity.token_ids == dense.token_ids
                relative_l2 = aggregate_relative_l2(
                    integrity.logits, dense.logits
                )
                if not r1_equal or relative_l2 > 1e-4:
                    raise RuntimeError(
                        "job-specific r=1 path differs from dense reference"
                    )
            if (
                not integrity.source_digests_unchanged
                or not integrity.absolute_union_mask_verified
            ):
                raise RuntimeError("job-specific Source/mask integrity failed")

        summary_current = summary_sources = None
        if job.kind in (
            V6A800JobKind.SUMMARY_GENERATION,
            V6A800JobKind.CANDIDATE_COMPARE,
            V6A800JobKind.MULTISEGMENT_COMPARE,
        ):
            summary_current, summary_sources = self._summary_tensors(
                fixture,
                job.probe_layer,
                job.compared_variants,
            )

        def operation() -> GenerationTrace | None:
            if job.kind in (V6A800JobKind.CORRECTNESS, V6A800JobKind.UNION_REPAIR):
                return self._reuse_generate(
                    fixture,
                    ratio=job.repair_ratio,
                    token_count=1,
                    probe_layer=job.probe_layer,
                    winner_variant=winner_variant,
                )
            elif job.kind == V6A800JobKind.SUMMARY_GENERATION:
                # exact_bf16 summary: materialize the selected current K/V rows.
                summary_current.clone()
            elif job.kind in (
                V6A800JobKind.CANDIDATE_COMPARE,
                V6A800JobKind.MULTISEGMENT_COMPARE,
            ):
                self._cuda_compare(summary_current, summary_sources)
            elif job.kind == V6A800JobKind.MULTISOURCE_LOAD:
                tickets = [
                    self.source_loader.begin(
                        segment_id="load-c%d" % index,
                        source_id="load-s%d" % index,
                        canonical_layers=fixture.canonical_variants[index][0],
                        segment_positions=fixture.segment_positions[index],
                    )
                    for index in range(job.segment_count)
                ]
                for ticket in tickets:
                    for event in ticket.layer_events.values():
                        event.synchronize()
            else:
                raise ValueError("unsupported qualification job kind")
            return None

        for _ in range(job.warmups):
            operation()
        gpu_rows: List[float] = []
        host_rows: List[float] = []
        for _ in range(job.repeats):
            if job.kind in (
                V6A800JobKind.CORRECTNESS,
                V6A800JobKind.UNION_REPAIR,
            ):
                measured = operation()
                if measured is None:
                    raise RuntimeError("repair operation produced no trace")
                gpu_ms, host_ms = measured.gpu_ms, measured.host_ms
            else:
                gpu_ms, host_ms = self._timed_operation(operation)
            gpu_rows.append(gpu_ms)
            host_rows.append(host_ms)
        return QualificationJobResult(
            job_id=job.job_id,
            passed=True,
            cuda_event_timing=True,
            gpu_ms=statistics.median(gpu_rows),
            host_ms=statistics.median(host_rows),
            r1_dense_token_ids_equal=r1_equal,
            teacher_forced_logit_relative_l2=relative_l2,
            canonical_source_digests_unchanged=(
                integrity.source_digests_unchanged
                if integrity is not None else
                bool(self.sentinel["canonical_source_digests_unchanged"])
            ),
            artifact_digests_unchanged=(
                integrity.artifact_digests_unchanged
                if integrity is not None else
                bool(self.sentinel.get("artifact_digests_unchanged", True))
            ),
            absolute_union_mask_verified=(
                integrity.absolute_union_mask_verified
                if integrity is not None else
                bool(self.sentinel["absolute_union_mask_verified"])
            ),
        )
