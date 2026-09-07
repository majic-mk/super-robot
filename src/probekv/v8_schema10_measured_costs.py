"""Measured cost service for the live backend; no residual-derived costs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest
from .v8_schema10_cost_provider import ProfiledJointTimelineEstimator, UnsupportedTimelineCost
from .v8_schema10_cost_provider import MeasurementKey, EXECUTION_SHAPE_KEY, LEGACY_IDENTITY_KEY
from .v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound


class MeasuredRequestCostProvider:
    def __init__(self, path, *, expected_sha256, provenance):
        if file_digest(Path(path)) != expected_sha256:
            raise ValueError("sentinel cost table file digest mismatch")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("provenance") != provenance or data.get("formal_profile_frozen") is not False:
            raise ValueError("sentinel costs have different provenance or pretend formal freezing")
        self.provenance, self.sha = provenance, expected_sha256
        self.key_contract = data.get("key_contract", LEGACY_IDENTITY_KEY)
        if self.key_contract not in {EXECUTION_SHAPE_KEY, LEGACY_IDENTITY_KEY}:
            raise ValueError("unknown measurement key contract")
        self.rows, self.joint_rows = {}, data.get("joint_rows", [])
        for row in data.get("rows", []):
            unsigned = {k: v for k, v in row.items() if k != "row_sha256"}
            if digest_json(unsigned) != row.get("row_sha256"):
                raise ValueError("corrupt cost sample row")
            if row.get("provenance") != provenance:
                raise ValueError("cost sample provenance differs from its table")
            if row.get("origin") != "real_cuda_execution" or row.get("fake_timing") is not False:
                raise ValueError("only actual GPU samples can supply admission costs")
            if row.get("warmup_excluded") is not True or row.get("outlier_policy") != "none":
                raise ValueError("measurement sample policy mismatch")
            values = row.get("samples_ms", [])
            if not values or any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("missing/invalid measured samples")
            key = (row["category"], digest_json(row["query"]))
            if key in self.rows:
                raise ValueError("duplicate measurement cell")
            self.rows[key] = row

    @staticmethod
    def identity(context):
        return {"prompt_token_ids_sha256": digest_json(context.request["token_ids"]),
                "cached_prefix_tokens": context.cached_prefix_tokens,
                "prefix_cache_mode": context.prefix_cache_mode,
                "timing_scope": "arrival_to_first_token", "sampling": context.sampling_signature}

    def _lookup(self, category, query):
        return self.rows.get((category, digest_json(query)))

    def dense_reference(self, context):
        row = self._lookup("dense_reference", self.identity(context))
        # Paired reference is an actual fixed observation, not the fastest of
        # opportunistically searched replicas/runs. Its pairing is preregistered.
        if row is None or len(row["samples_ms"]) != 1 or row["samples_ms"][0] <= 0:
            return None
        return row["samples_ms"][0]

    def comparison_ms(self, context, depth, k, n):
        row = self._lookup("comparison_batch", {"depth": depth, "k": k, "tokens": n,
                           "backing_tier": context.selection_backing_tier})
        return max(row["samples_ms"]) if row else None

    def gate1(self, context, sid, source_id, depth):
        query = {"request": self.identity(context), "segment_id": sid, "source_id": source_id,
                 "completed_depth": depth, "first_reuse_layer": depth + 1}
        reuse = self._lookup("source_local_marginal", self._source_query(context, sid, source_id, depth, "source_local_marginal", query))
        dense = self._lookup("source_local_dense", self._source_query(context, sid, source_id, depth, "source_local_dense", query))
        if not reuse or not dense or min(dense["samples_ms"]) <= 0:
            return None
        parts = reuse.get("component_lower_ms", {})
        names = ("support_build", "visible_load", "repair")
        if set(parts) != set(names) or any(not math.isfinite(parts[n]) or parts[n] < 0 for n in names):
            return None
        if sum(parts.values()) > min(reuse["samples_ms"]) + 1e-9:
            raise ValueError("source-local component attribution exceeds measured total")
        return Gate1LocalPlan(source_id, depth, depth, depth + 1, context.actual_repair_check_sunk_ms,
            Gate1MarginalLowerBound(*(parts[n] for n in names)), min(dense["samples_ms"]))

    def candidate_future_ms(self, context, sid, source_id, depth):
        query = {"request": self.identity(context),
            "segment_id": sid, "source_id": source_id, "completed_depth": depth,
            "first_reuse_layer": depth + 1}
        row = self._lookup("source_future", self._source_query(context, sid, source_id, depth, "source_future", query))
        return max(row["samples_ms"]) if row else None

    def _source_query(self, context, sid, source_id, depth, category, legacy):
        if self.key_contract == LEGACY_IDENTITY_KEY:
            return legacy
        shape = context.source_measurement_shape(sid, source_id, depth)
        required = {"prompt_tokens", "prefix_tokens", "positions", "completed_depth",
                    "num_layers", "dtype", "kv_heads", "head_dim", "tier", "bytes", "layout"}
        if not required <= shape.keys():
            raise UnsupportedTimelineCost("incomplete source execution shape")
        return MeasurementKey(category, shape).query()

    def preparation(self, context, sid, source_id):
        # Source-local feasibility is not request admission. Resource/waste
        # admission additionally needs a whole-request dense-fallback timeline.
        estimator = self.joint_estimator(context)
        result = estimator.lookup(context.dense_fallback_joint_context())
        dense = self.dense_reference(context)
        if result.estimate is None or dense is None:
            return None
        query = {"source_id": source_id,
                            "request": self.identity(context), "segment_id": sid,
                            "boundary": context.current_completed_depth + 1}
        copy = self._lookup("winner_visible_preparation", self._source_query(context, sid, source_id,
            context.current_completed_depth, "winner_visible_preparation", query))
        if copy is None:
            return None
        budget = max(0., dense - context.actual_sunk_ms - result.estimate.joint_future_ms)
        return {"resource_admitted": max(copy["samples_ms"]) <= budget,
                "predicted_visible_and_interference_ms": max(copy["samples_ms"]),
                "speculative_waste_budget_ms": budget, "measurement_sha256": self.sha}

    def joint_estimator(self, context):
        return ProfiledJointTimelineEstimator(provenance=self.provenance,
            shape=context.execution_shape(), measurements=self.joint_rows, measurement_digest=self.sha,
            key_contract=self.key_contract)
