"""Live SelectionState bridge for the resumable CacheBlend executor.

Unlike H1 fixtures' offline current_layers, current K here comes directly from
the paused request. Gate1 and final joint profiles are supplied by the runtime
provider, never reconstructed from residual scores.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Any, Callable, Mapping

from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema6_hbm import HBMReservationKind
from .v8_schema10_execution import ProductionSelectionSession, digest_json


class LiveSelectionBridge:
    def __init__(self, session: ProductionSelectionSession, *,
                 gate1_provider: Callable[..., Any], final_planner: Any,
                 snapshot_provider: Callable[[], Any], hbm_manager: Any,
                 metadata_order_by_segment: Mapping[str, tuple[int, ...]],
                 batch_time_predictor: Callable[[int, int, int], float],
                 arrival_ns: int | None = None) -> None:
        self.session, self.gate1_provider = session, gate1_provider
        self.final_planner, self.snapshot_provider = final_planner, snapshot_provider
        self.hbm_manager, self.metadata_order = hbm_manager, metadata_order_by_segment
        self.batch_time_predictor = batch_time_predictor
        self.arrival_ns = time.perf_counter_ns() if arrival_ns is None else arrival_ns
        if self.arrival_ns < 0 or self.arrival_ns > time.perf_counter_ns():
            raise ValueError("invalid request arrival")
        self.final_events: list[dict[str, Any]] = []
        self.comparison_timings: list[dict[str, Any]] = []
        self.first_token_ns: int | None = None

    @staticmethod
    def source_id(fixture: Any, index: int, variant: int) -> str:
        return "s%d-v%d-%s" % (index, variant, fixture.canonical_variant_digests[index][variant][:16])

    def select(self, executor: Any, engine: Any, fixture: Any) -> dict[int, int]:
        torch = executor.torch
        inventory = tuple("c%d" % i for i in range(len(fixture.segment_positions)))
        if inventory != self.session.segment_ids:
            raise ValueError("live request Segment inventory differs from selection session")
        if self.session.selector.checkpoint_depths not in {(1,), (1, 2)}:
            raise RuntimeError("dense-barrier adapter requires FAST dispatch; use the preserved legacy adapter for deep runtime")
        winners = {}
        selection_started = time.perf_counter_ns()
        for depth in self.session.selector.checkpoint_depths:
            probe_start = time.perf_counter_ns()
            engine.advance_to_layer(depth)
            # d observes pre-RoPE K entering block d+1, from THIS request.
            observed = engine.session.observe_pre_rope_k(depth)
            if observed.device.type == "cuda":
                torch.cuda.current_stream(observed.device).synchronize()
            self.session.ledger.observe_shared_interval(
                f"{self.session.request_id}:shared-probe:{depth}", probe_start, time.perf_counter_ns())
            active_positions = tuple(engine.session.active_positions)
            position_index = {int(p): i for i, p in enumerate(active_positions)}
            for index, positions in enumerate(fixture.segment_positions):
                segment_id = inventory[index]
                if segment_id in self.session.decisions:
                    continue
                if any(p not in position_index for p in positions):
                    raise RuntimeError("Prefix or dropped token entered Source comparison")
                variants = fixture.selection_variants[index]
                stored_k = len(fixture.canonical_variants[index])
                if stored_k > 16:
                    raise ValueError("more than sixteen Source Variants")
                ordering = tuple(self.metadata_order[segment_id])
                if len(ordering) != stored_k or set(ordering) != set(range(stored_k)):
                    raise ValueError("metadata ordering must cover the complete correctness-eligible fixture pool")
                available = tuple(v for v in ordering if v < len(variants)
                                  and depth < len(variants[v]) and variants[v][depth] is not None)
                current = observed[[position_index[p] for p in positions]]
                if current.ndim != 3 or current.dtype != torch.bfloat16:
                    raise ValueError("live K must be exact BF16 [tokens, kv_heads, head_dim]")
                # Account for FP32 differences, norms, sorting and masks, not
                # merely the BF16 input bytes. Reserve BEFORE staging sources.
                row_bytes = current.numel() * 32 + current.shape[0] * 32
                capacity = self.hbm_manager.selector_lease_bytes
                batch_k = min(len(available), max(0, (capacity - row_bytes) // row_bytes))
                candidates, plans = [], {}
                offset = 0
                while offset < len(available):
                    if not batch_k:
                        break
                    # A whole HBM-fitting batch may exceed the legacy time
                    # budget. Try the largest time-fitting prefix, not zero.
                    feasible = [(k, self.batch_time_predictor(depth, k, len(positions)))
                                for k in range(1, min(batch_k, len(available) - offset) + 1)]
                    feasible = [(k, ms) for k, ms in feasible if self.session.ledger.may_compare(ms)]
                    if not feasible:
                        break
                    count, predicted = feasible[-1]
                    batch = available[offset:offset + count]
                    event_id = "%s:%s:%d:%d" % (self.session.request_id, segment_id, depth, offset)
                    reservations = self.hbm_manager.reserve_batch(
                        owner_request_id=self.session.request_id,
                        rows=((event_id, row_bytes * (len(batch) + 1), HBMReservationKind.SELECTION_WORKSPACE),))
                    started = time.perf_counter_ns()
                    sources = drift = order = None
                    try:
                        self.session.ledger.reserve(event_id, predicted)
                        tensors = [variants[v][depth] for v in batch]
                        if any(x.shape != current.shape or x.dtype != torch.bfloat16 for x in tensors):
                            raise ValueError("SelectionState geometry/dtype differs from current K")
                        sources = torch.stack([x.to(current.device, non_blocking=True) for x in tensors])
                        drift = (sources.float() - current.float().unsqueeze(0)).square().sum(dim=(2, 3)).sqrt()
                        drift /= current.float().square().sum(dim=(1, 2)).sqrt().clamp_min(1e-12).unsqueeze(0)
                        n = len(positions)
                        trim = min(n - 1, math.ceil(self.session.selector.variant_profile.source_residual_trim_ratio * n))
                        # Segment positions are ascending, giving deterministic absolute-position ties.
                        if tuple(sorted(positions)) != tuple(positions) or not n:
                            raise ValueError("Segment absolute positions must be nonempty and ordered")
                        order = drift.argsort(dim=1, descending=True, stable=True)
                        values = drift.gather(1, order)[:, trim:].mean(dim=1).detach().cpu().tolist()
                        ended = time.perf_counter_ns()
                        self.session.ledger.settle(event_id, started, ended)
                        self.comparison_timings.append({"event_id": event_id, "start_ns": started, "end_ns": ended})
                    except BaseException:
                        if event_id in self.session.ledger.pending:
                            self.session.ledger.cancel(event_id)
                        raise
                    finally:
                        # Retain the lease until the GPU work AND scratch
                        # tensor lifetime have ended (also on failure).
                        if current.device.type == "cuda":
                            torch.cuda.current_stream(current.device).synchronize()
                        sources = drift = order = None
                        self.hbm_manager.release(reservations[0].reservation_id)
                    for variant, value in zip(batch, values):
                        source_id = self.source_id(fixture, index, variant)
                        plan = self.gate1_provider(segment_id, source_id, depth)
                        plans[source_id] = plan
                        candidates.append(ResidualCandidate(source_id, value, plan.predicted_reuse_marginal_lower_ms,
                                                            ordering.index(variant)))
                    offset += len(batch)
                counts = CandidateCounts(stored_k, stored_k, len(available), len(available), len(candidates))
                decision = self.session.step(segment_id, completed_depth=depth, counts=counts,
                                             candidates=candidates, gate1_plan_by_source=plans)
                if decision.selected_source_variant_id:
                    winners[index] = next(v for v in available
                                          if self.source_id(fixture, index, v) == decision.selected_source_variant_id)
                self.session.ledger.observe_shared_interval(
                    f"{self.session.request_id}:selection-window:{segment_id}:{depth}",
                    selection_started, time.perf_counter_ns())
            if self.session.closed:
                break
        if not self.session.closed:
            raise RuntimeError("live selector did not resolve the complete inventory")
        return winners

    def record_dense_fallback(self) -> None:
        if not self.session.closed or any(x.selected_source_variant_id for x in self.session.decisions.values()):
            raise RuntimeError("dense-only disposition requires resolved abstention for every Segment")
        if self.final_events:
            raise RuntimeError("request disposition already recorded")
        event = {"kind": "dense_fallback", "timestamp_ns": time.perf_counter_ns(),
                 "reason": "all_segments_selector_abstained", "accepted_ready_segment_ids": ()}
        self.final_events.append({**event, "event_id": digest_json(event)})

    def final_commit(self, *, boundary_by_segment: Mapping[str, int], union_mask_digest: str) -> Any:
        if not self.session.closed or self.final_events:
            raise RuntimeError("FinalCommit requires a closed selection barrier and one irreversible decision")
        frozen = {s for s, d in self.session.decisions.items() if d.selected_source_variant_id}
        if not set(boundary_by_segment) <= frozen:
            raise RuntimeError("FinalCommit cannot admit a Source that was not selected")
        if any(b <= self.session.last_depth[s] for s, b in boundary_by_segment.items()):
            raise RuntimeError("selective boundary must follow the actual selection depth")
        snapshot = self.snapshot_provider()
        result = self.final_planner.plan_ready_subset(
            inventory_segment_ids=self.session.segment_ids,
            eligible_ready_segment_ids=tuple(boundary_by_segment), committed_segment_ids=(),
            actual_boundary_by_segment=boundary_by_segment,
            actual_sunk_ms=(time.perf_counter_ns() - self.arrival_ns) / 1e6,
            dense_reference_total_ms=self.session.ledger.dense_reference_ttft_ms,
            snapshot=snapshot, current_snapshot=self.snapshot_provider(), union_mask_digest=union_mask_digest)
        snapshot.assert_current(self.snapshot_provider())
        if set(result.accepted_ready_segment_ids) - set(boundary_by_segment):
            raise RuntimeError("FinalCommit accepted an unready Segment")
        self.final_events.append({"kind": "final_commit", "timestamp_ns": time.perf_counter_ns(),
                                  "decision": asdict(result), "event_id": digest_json(asdict(result))})
        return result

    def on_first_token(self) -> None:
        if self.first_token_ns is not None:
            raise RuntimeError("first token recorded twice")
        self.first_token_ns = time.perf_counter_ns()

    def audit(self) -> Mapping[str, Any]:
        return {"execution_kind": "online_policy", "forced_source": False,
                "request_id": self.session.request_id, "arrival_ns": self.arrival_ns,
                "first_token_ns": self.first_token_ns, "completion_ns": time.perf_counter_ns(),
                "request_ttft_ms": ((self.first_token_ns - self.arrival_ns) / 1e6
                                    if self.first_token_ns is not None else None),
                "selection_events": tuple(self.session.events), "runtime_events": tuple(self.final_events),
                "comparison_timings": tuple(self.comparison_timings),
                "final_commit_executed": any(x["kind"] == "final_commit" for x in self.final_events),
                "final_commit_not_applicable_reason": ("no_frozen_sources" if self.final_events
                    and self.final_events[-1]["kind"] == "dense_fallback" else None),
                "selected_source_variant_ids": [x.selected_source_variant_id for x in self.session.decisions.values()
                                                 if x.selected_source_variant_id],
                "selection_active_ms": self.session.ledger.actual_active_ms,
                "selection_budget_policy": self.session.ledger.policy.mode,
                "gpu_runtime_qualified": False, "paper_evidence": False}
