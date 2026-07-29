from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def audit_cacheblend_closed_loop(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Audit server records without silently accepting simulated closure."""

    errors = []
    selected = 0
    accepted = 0
    rejected = 0
    abstained = 0
    for index, row in enumerate(records):
        prefix = "row[%d]" % index
        if row.get("closure_policy") != "two_stage_refined_admission":
            errors.append("%s: wrong closure policy" % prefix)
        selection_state = row.get("selection_state")
        selected_source = row.get("selected_source_id")
        execution_mode = row.get("execution_mode")
        if selection_state == "not_selected":
            abstained += 1
            if selected_source is not None:
                errors.append("%s: abstention retained a Source" % prefix)
            if execution_mode != "full_recompute":
                errors.append("%s: abstention did not execute full" % prefix)
            if row.get("source_ready_ms") is not None:
                errors.append("%s: abstention loaded a Source" % prefix)
            continue
        if selection_state != "selected" or not selected_source:
            errors.append("%s: invalid selection state" % prefix)
            continue
        selected += 1
        if row.get("source_locked_at_probe") != selected_source:
            errors.append("%s: probe Source was not locked" % prefix)
        if row.get("refined_source_id") != selected_source:
            errors.append("%s: refined stage changed Source" % prefix)
        if row.get("runtime_selected_source_id") != selected_source:
            errors.append("%s: CacheBlend executed another Source" % prefix)
        for key in (
            "source_load_start_ms",
            "source_ready_ms",
            "scheduled_step_finish_ms",
            "a_resume_ms",
            "post_ready_blocking_ms",
            "source_load_bytes",
            "evaluated_reuse_boundary",
            "refined_reuse_total_ms",
            "full_total_ms",
            "runtime_realized_ttft_ms",
        ):
            if row.get(key) is None:
                errors.append("%s: missing %s" % (prefix, key))
        ready = row.get("source_ready_ms")
        resume = row.get("a_resume_ms")
        blocking = row.get("post_ready_blocking_ms")
        if ready is not None and resume is not None and blocking is not None:
            if abs(float(blocking) - (float(resume) - float(ready))) > 1e-6:
                errors.append("%s: inconsistent post-ready blocking" % prefix)
        if row.get("refined_cost_value_kind") != (
            "refined_actual_past_profiled_future"
        ):
            errors.append("%s: refined cost provenance is ambiguous" % prefix)
        if not row.get("refined_profile_key"):
            errors.append("%s: missing boundary profile key" % prefix)
        if row.get("reuse_accepted"):
            accepted += 1
            if execution_mode != "reuse":
                errors.append("%s: accepted request did not reuse" % prefix)
            if (
                row.get("actual_reuse_boundary")
                != row.get("evaluated_reuse_boundary")
            ):
                errors.append("%s: executed another boundary" % prefix)
            if row.get("runtime_execution_mode") != "reuse":
                errors.append("%s: CacheBlend did not execute reuse" % prefix)
            if row.get("runtime_wasted_loaded_bytes") != 0:
                errors.append("%s: accepted reuse marked bytes wasted" % prefix)
        else:
            rejected += 1
            if execution_mode != "full_recompute":
                errors.append("%s: rejected request did not run full" % prefix)
            if row.get("actual_reuse_boundary") is not None:
                errors.append("%s: rejected request exposed reuse" % prefix)
            if row.get("runtime_execution_mode") != "full_recompute":
                errors.append("%s: CacheBlend fallback was not full" % prefix)
            if row.get("runtime_wasted_loaded_bytes") != row.get(
                "source_load_bytes"
            ):
                errors.append(
                    "%s: rejected request lost transfer accounting" % prefix
                )
    return {
        "passed": bool(records) and not errors,
        "records": len(records),
        "selected": selected,
        "accepted": accepted,
        "rejected": rejected,
        "abstained": abstained,
        "errors": errors,
        "evidence_class": "server_pilot",
        "paper_evidence": False,
    }
