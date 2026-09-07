"""Exact-support measured cost lookup; absence is never a zero-cost estimate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import json
from typing import Mapping

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest
from .v8_schema6_planner import JointTimelineContext, JointTimelineEstimate


class UnsupportedTimelineCost(RuntimeError):
    pass


@dataclass(frozen=True)
class CostLookup:
    status: str
    query_digest: str
    estimate: JointTimelineEstimate | None
    reason: str | None
    measurement_digest: str | None


@dataclass(frozen=True)
class RequestExecutionShape:
    prompt_token_count: int
    cached_prefix_tokens: int
    num_layers: int
    completed_depth: int
    positions_by_segment: Mapping[str, tuple[int, ...]]
    repair_by_segment_by_layer: Mapping[str, Mapping[int, tuple[int, ...]]]
    committed_boundary_by_segment: Mapping[str, int]
    source_state_by_segment: Mapping[str, Mapping]
    dense_reference_identity: Mapping

    def __post_init__(self):
        if not 0 <= self.cached_prefix_tokens < self.prompt_token_count or not 0 <= self.completed_depth < self.num_layers:
            raise ValueError("invalid prefix/depth execution shape")
        positions = [p for group in self.positions_by_segment.values() for p in group]
        if len(positions) != len(set(positions)) or any(not self.cached_prefix_tokens <= p < self.prompt_token_count for p in positions):
            raise ValueError("Segment positions overlap or include native Prefix")
        if not self.dense_reference_identity:
            raise ValueError("matched dense reference identity required")

    def masks(self, context: JointTimelineContext) -> dict[int, tuple[int, ...]]:
        if set(context.inventory_segment_ids) != set(self.positions_by_segment):
            raise ValueError("joint estimate omitted part of the request inventory")
        if set(context.committed_segment_ids) != set(self.committed_boundary_by_segment):
            raise ValueError("committed fixed path is incomplete")
        boundaries = {**self.committed_boundary_by_segment, **context.boundary_by_segment}
        result = {}
        for layer in range(self.completed_depth + 1, self.num_layers + 1):
            active = set(range(self.cached_prefix_tokens, self.prompt_token_count))
            for sid, boundary in boundaries.items():
                if layer >= boundary:
                    support = self.repair_by_segment_by_layer.get(sid, {}).get(layer)
                    if support is None or not set(support) <= set(self.positions_by_segment[sid]):
                        raise UnsupportedTimelineCost("missing or invalid measured repair support")
                    active.difference_update(self.positions_by_segment[sid])
                    active.update(support)
            result[layer] = tuple(sorted(active))
        return result


class ProfiledJointTimelineEstimator:
    """Sentinel tables stay provisional even when populated with real samples.

    A row is an exact joint query (not a sum of per-Segment TTFTs). Component
    event intervals are explanatory only; the critical path is measured wall
    time from the matched boundary to the first token. No extrapolation.
    """
    def __init__(self, *, provenance: Mapping, shape: RequestExecutionShape,
                 measurements, measurement_digest: str, allow_test_measurements=False):
        required = {"model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope"}
        if not required <= provenance.keys() or any(not provenance[k] for k in required):
            raise ValueError("cost provenance is incomplete")
        self.provenance, self.shape = dict(provenance), shape
        self.measurement_digest, self.rows = measurement_digest, {}
        self.formal_profile_frozen = False
        for raw in measurements:
            row = dict(raw)
            claimed = row.pop("row_sha256", None)
            if digest_json(row) != claimed or row.get("provenance") != self.provenance:
                raise ValueError("cost row digest/provenance mismatch")
            if not allow_test_measurements and (row.get("origin") != "real_cuda_execution" or row.get("fake_timing") is not False):
                raise ValueError("non-GPU measurements cannot supply online admission")
            values = row.get("joint_future_wall_ms_samples", [])
            if not values or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("missing real joint wall-clock samples")
            if row.get("warmup_excluded") is not True or row.get("outlier_policy") != "none":
                raise ValueError("cost sample policy differs from the sentinel contract")
            key = digest_json(row["query"])
            if key in self.rows:
                raise ValueError("duplicate exact-support measurement cell")
            self.rows[key] = row
        self.queries = []

    @classmethod
    def from_file(cls, path, *, expected_file_sha256, provenance, shape):
        path = Path(path)
        if file_digest(path) != expected_file_sha256:
            raise ValueError("runtime measurement file digest mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("formal_profile_frozen") is not False:
            raise ValueError("this entry accepts provisional sentinel tables only")
        return cls(provenance=provenance, shape=shape, measurements=payload["rows"],
                   measurement_digest=expected_file_sha256)

    def query(self, context: JointTimelineContext):
        masks = self.shape.masks(context)
        # Rebuilt for EVERY trial subset; caller's former mask digest is never a cache key.
        return {"inventory": list(context.inventory_segment_ids),
                "reuse": sorted(context.reuse_segment_ids), "dense": sorted(context.dense_fallback_segment_ids),
                "committed": sorted(context.committed_segment_ids),
                "boundary": dict(context.boundary_by_segment),
                "committed_boundary": dict(self.shape.committed_boundary_by_segment),
                "layer_active_rows": {str(l): len(p) for l, p in masks.items()},
                "union_mask_digest": digest_json(masks),
                "completed_depth": self.shape.completed_depth,
                "prompt_tokens": self.shape.prompt_token_count,
                "prefix_tokens": self.shape.cached_prefix_tokens,
                "dense_reference": dict(self.shape.dense_reference_identity),
                "source_copy_ready_state": dict(self.shape.source_state_by_segment),
                "scheduler_snapshot": context.scheduler_state_id}

    def lookup(self, context) -> CostLookup:
        try:
            query = self.query(context)
        except UnsupportedTimelineCost as exc:
            result = CostLookup("UNSUPPORTED", "", None, str(exc), self.measurement_digest)
            self.queries.append(asdict(result))
            return result
        key = digest_json(query)
        row = self.rows.get(key)
        if row is None:
            result = CostLookup("UNSUPPORTED", key, None, "no_exact_joint_measurement", self.measurement_digest)
        else:
            # A small sentinel sample maximum is a conservative empirical bound,
            # NOT a calibrated probabilistic guarantee or a formally frozen UCB.
            upper = max(row["joint_future_wall_ms_samples"])
            estimate = JointTimelineEstimate(upper, {"measured_joint_critical_path": upper},
                                             row.get("per_segment_attribution_ms", {}))
            result = CostLookup("SUPPORTED", key, estimate, None, self.measurement_digest)
        self.queries.append(asdict(result))
        return result

    def estimate(self, context):
        result = self.lookup(context)
        if result.estimate is None:
            raise UnsupportedTimelineCost(result.reason)
        return result.estimate


def validate_dense_reference(reference, *, expected_identity):
    if reference.get("identity") != expected_identity or reference.get("origin") != "real_cuda_execution":
        raise UnsupportedTimelineCost("dense reference does not match request/Prefix/endpoints")
    value = reference.get("ttft_ms")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise UnsupportedTimelineCost("dense reference measurement missing")
    return value
