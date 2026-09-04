from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple

from .metrics import best_answer_f1, token_id_f1
from .v6_a800_executor import aggregate_relative_l2
from .v6_h1_runtime import V8H1CaseRuntime
from .v8_schema10_profile import SCHEMA10_REPAIR_RATIO_GRID, SCHEMA10_TRIM_GRID


@dataclass(frozen=True)
class SourceResidualObservationV10:
    source_id: str
    completed_depth: int
    source_residual_trim_ratio: float
    residual_score: float


class Schema10DevelopmentCaseRuntime(V8H1CaseRuntime):
    """Real development-only measurements on the schema10 CacheBlend path.

    It reuses one exact-dense fixture for every rho/K/depth analysis.  The
    class accepts only the isolated calibration/development partition and does
    not expose a paper-evidence mode.
    """

    runtime_mode = "v8_schema10_profile_development_case"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["allowed_splits"] = ("calibration", "development")
        super().__init__(*args, **kwargs)
        # The formal CPU backing path is created pinned once.  Per-request
        # pin_memory() would otherwise contaminate TTFT and violate schema10's
        # backing/staging contract.
        torch = self.executor.torch
        pinned_variants = tuple(
            tuple(
                tuple(
                    tuple(
                        tensor if tensor.is_pinned() else tensor.pin_memory()
                        for tensor in (key, value)
                    )
                    for key, value in layers
                )
                for layers in segment
            )
            for segment in self.fixture.runtime.canonical_variants
        )
        runtime = replace(
            self.fixture.runtime,
            canonical_variants=pinned_variants,
        )
        self.fixture = replace(self.fixture, runtime=runtime)

    def residual_observations(
        self,
        completed_depths: Sequence[int],
        trim_ratios: Sequence[float] = SCHEMA10_TRIM_GRID,
    ) -> Tuple[SourceResidualObservationV10, ...]:
        torch = self.executor.torch
        rows = []
        token_count = self.fixture.segment_tokens
        for completed_depth in completed_depths:
            if not 1 <= int(completed_depth) < self.executor.model_spec.num_layers:
                raise ValueError("profile completed depth is outside the model")
            # completed_depth=d observes pre-RoPE K entering Transformer block d+1.
            current = self.fixture.runtime.current_layers[0][int(completed_depth)][0]
            for source_id, source_index in self.source_index.items():
                # Source selection reads the independent SelectionState K;
                # slicing the full-KV Artifact as an implicit fallback is
                # forbidden even during development profiling.
                source = self.fixture.runtime.selection_variants[0][source_index][
                    int(completed_depth)
                ]
                dimensions = tuple(range(1, current.ndim))
                numerator = (current - source).float().square().sum(dim=dimensions).sqrt()
                denominator = current.float().square().sum(dim=dimensions).sqrt().clamp_min(1e-12)
                drift = (numerator / denominator).detach().cpu()
                for ratio in trim_ratios:
                    if float(ratio) not in SCHEMA10_TRIM_GRID:
                        raise ValueError("profile rho is outside the frozen grid")
                    trim = min(token_count - 1, int(math.ceil(float(ratio) * token_count)))
                    # Stable ties use absolute token position (the original row index).
                    ranking = sorted(
                        range(token_count),
                        key=lambda index: (-float(drift[index]), index),
                    )
                    kept = ranking[trim:]
                    score = sum(float(drift[index]) for index in kept) / len(kept)
                    rows.append(
                        SourceResidualObservationV10(
                            source_id,
                            int(completed_depth),
                            float(ratio),
                            score,
                        )
                    )
        return tuple(rows)

    def repair_ratio_observation(
        self,
        *,
        source_id: str,
        first_reuse_layer: int,
        repair_ratio: float,
    ) -> Mapping[str, object]:
        if float(repair_ratio) not in SCHEMA10_REPAIR_RATIO_GRID:
            raise ValueError("repair observation uses an unprofiled ratio")
        source_index = self.source_index[source_id]
        greedy = self._generate(source_index, first_reuse_layer, repair_ratio, teacher=False)
        teacher = self._generate(source_index, first_reuse_layer, repair_ratio, teacher=True)
        output_text = self.executor.tokenizer.decode(
            greedy.token_ids, skip_special_tokens=True
        )
        return {
            "case_id": self.case.case_id,
            "dataset": self.case.dataset,
            "source_id": source_id,
            "first_reuse_layer": int(first_reuse_layer),
            "repair_ratio": float(repair_ratio),
            "answer_f1": best_answer_f1(output_text, self.case.answers),
            "full_answer_f1": self.full_answer_f1,
            "answer_f1_drop": self.full_answer_f1
            - best_answer_f1(output_text, self.case.answers),
            "ordered_token_f1": token_id_f1(greedy.token_ids, self.full.token_ids),
            "token_ids_equal_full": greedy.token_ids == self.full.token_ids,
            "logit_relative_l2": aggregate_relative_l2(teacher.logits, self.full.logits),
            "gpu_ms": greedy.gpu_ms,
            "host_ms": greedy.host_ms,
            "source_digest_unchanged": greedy.source_digests_unchanged,
            "artifact_digest_unchanged": greedy.artifact_digests_unchanged,
            "absolute_union_mask_verified": greedy.absolute_union_mask_verified,
            "cuda_event_timing": True,
            "fake_timing": False,
            "paper_evidence": False,
            "locked_test_accessed": False,
        }

    def _generate(
        self, source_index: int, reuse_layer: int, ratio: float, *, teacher: bool
    ) -> Any:
        """Execute a development-only repair observation at the selected depth.

        The deployable schema10 fast path remains restricted to d1/d2.  The
        development sweep must also execute the frozen legacy checkpoint
        candidate (including its r=1 endpoint) so that fallback selection is
        backed by real A800 evidence rather than a synthetic extrapolation.
        """

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
            force_nonpaper_measurement_admission=True,
        )
