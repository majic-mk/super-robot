"""Cross-check the frozen YAML contract and enumerate expensive matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from probekv.matrix import main_rag_matrix, profile_matrix
from probekv.config import load_config
from probekv.statistics import minimum_zero_violation_trials
from probekv.cacheblend_patch import load_patch_manifest, patch_files_for_mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", default="configs/experiment_contract.yaml"
    )
    parser.add_argument(
        "--output", default="artifacts/contract_validation.json"
    )
    parser.add_argument(
        "--v8-contract", default="configs/experiment_contract_v8.yaml"
    )
    parser.add_argument(
        "--v8-schema6-contract",
        default="configs/experiment_contract_v8_schema6.yaml",
    )
    parser.add_argument(
        "--v8-schema7-contract",
        default="configs/experiment_contract_v8_schema7.yaml",
    )
    parser.add_argument(
        "--v8-schema8-contract",
        default="configs/experiment_contract_v8_schema8.yaml",
    )
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    v8_contract_path = Path(args.v8_contract)
    v8_contract = yaml.safe_load(v8_contract_path.read_text(encoding="utf-8"))
    v8_schema6_path = Path(args.v8_schema6_contract)
    v8_schema6 = yaml.safe_load(v8_schema6_path.read_text(encoding="utf-8"))
    v8_schema7_path = Path(args.v8_schema7_contract)
    v8_schema7 = yaml.safe_load(v8_schema7_path.read_text(encoding="utf-8"))
    v8_schema8_path = Path(args.v8_schema8_contract)
    v8_schema8 = yaml.safe_load(v8_schema8_path.read_text(encoding="utf-8"))
    errors = []
    if contract.get("schema_version") != 7:
        errors.append("current experiment contract must use schema version 7")
    invariants = contract["invariants"]
    if invariants["canonical_source_origin"] != "full_prefill":
        errors.append("canonical source origin must be full_prefill")
    if invariants["promote_selective_repair"]:
        errors.append("selective repair promotion must be false")
    if invariants["online_kmax"] != 4:
        errors.append("online Kmax must be 4")
    if max(invariants["offline_k"]) != 8:
        errors.append("offline K ablation must include 8")
    v6 = contract.get("protocol_versions", {}).get("v6_main", {})
    required_v6 = {
        "legacy_online_kmax_forbidden": True,
        "max_stored_variants_per_content": 16,
        "candidate_compare_policy": "all_within_request_budget",
        "max_compared_variants_per_segment": 16,
        "probe_metadata_compare_budget_fraction": 0.05,
        "segment_planning_policy": "all_exact_nonprefix",
        "max_detected_segments": None,
        "main_boundary_policy": "causal_staggered",
        "main_selection_execution_policy": "causal_commit_wait",
        "ablation_boundary_policy": "immediate_staggered",
        "ablation_selection_execution_policy": (
            "immediate_staggered_closed_loop"
        ),
        "legacy_boundary_policy": "common",
        "shadow_dense_probe_enabled": False,
        "calibration_policy_match_required": True,
        "partial_reuse_enabled": True,
        "joint_quality_policy": "simultaneous_conformal",
        "interference_accounting_mode": "explicit_penalty",
        "source_pool_policy": "global_hard_model_soft",
    }
    for key, expected in required_v6.items():
        if v6.get(key) != expected:
            errors.append("invalid v6 contract field %s" % key)
    if v6.get("k_ablation") != [1, 2, 4, 8, 16]:
        errors.append("v6 K ablation must cover 1,2,4,8,16")
    v7 = contract.get("protocol_versions", {}).get("v7_main", {})
    required_v7 = {
        "runtime_backend": "cacheblend_v7_closed_loop",
        "canonicalizer_version": "semantic_block_v1",
        "alignment_quantum": 16,
        "alignment_policy": "soft",
        "padding": False,
        "artifact_policy": "single_canonical_lossless",
        "max_artifacts_per_source_variant": 1,
        "canonical_kv_dtype": "bfloat16",
        "canonical_k_semantics": "pre_rope",
        "canonical_v_semantics": "raw",
        "lossy_full_kv_artifacts_enabled": False,
        "replica_policy": "one_backing_plus_transient_hot",
        "max_replicas_per_artifact_per_tier": 1,
        "repair_rounding_policy": "ceil",
        "boundary_policy": "per_segment_staggered",
        "same_source_replica_replan_only": True,
        "qualification_gate_schema": 3,
    }
    for key, expected in required_v7.items():
        if v7.get(key) != expected:
            errors.append("invalid v7 contract field %s" % key)
    if v7.get("replica_tiers") != ["gpu", "pinned_cpu", "ssd"]:
        errors.append("v7 Replica tiers must be gpu/pinned_cpu/ssd")
    if v8_contract.get("schema_version") != 5 or v8_contract.get("protocol_version") != 8:
        errors.append("v8 contract must use protocol 8 schema 5")
    v8_method = v8_contract.get("method", {})
    for key, expected in {
        "selector_type": "training_free",
        "learned_selector_enabled": False,
        "quality_predictor_enabled": False,
        "fixed_repair_ratio": 0.15,
        "max_stored_variants_per_content": 16,
        "min_compared_variants_for_multisource": 2,
        "insufficient_ranking_policy": "abstain_dense",
        "comparison_budget_fraction": 0.05,
        "source_freeze_is_final": True,
        "winner_only_full_kv_prefetch": True,
    }.items():
        if v8_method.get(key) != expected:
            errors.append("invalid v8 method field %s" % key)
    if v8_contract.get("planning", {}).get("segment_count_cap", "missing") is not None:
        errors.append("v8 must not cap detected non-prefix Segments")
    if v8_contract.get("h1_diagnostic", {}).get("total_rows") != 9720:
        errors.append("v8 H1 diagnostic must retain all 9,720 rows")
    if v8_contract.get("h1_diagnostic", {}).get("ratios") != [
        0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0
    ]:
        errors.append("v8 schema-v5 H1 grid must include the online 0.15 point")
    profile_contract = v8_contract.get("profile_and_qualification", {})
    if profile_contract.get("qualification_jobs_per_model_policy") != 140:
        errors.append("v8 requires 140 qualification jobs per Model x Policy")
    if profile_contract.get("qualification_jobs_total") != 560:
        errors.append("v8 requires four independent 140-job qualifications")
    if v8_schema6.get("protocol_version") != 8 or v8_schema6.get("schema_version") != 6:
        errors.append("new v8 runtime contract must use schema-v6")
    if v8_schema6.get("gate2", {}).get("accounting") != "request_joint_timeline":
        errors.append("schema-v6 Gate 2 must use request joint-timeline accounting")
    if v8_schema6.get("gate3", {}).get("decision") != "ready_subset":
        errors.append("schema-v6 Gate 3 must return a ready-subset decision")
    speculative = v8_schema6.get("speculative_preparation", {})
    if not speculative.get("requires_physical_replica_lease") or not speculative.get("requires_hbm_reservation"):
        errors.append("schema-v6 speculative preparation requires lease and HBM reservation")
    if v8_schema6.get("selector", {}).get("checkpoints", {}).get("mistral") != [1, 2, 4, 5, 8]:
        errors.append("schema-v6 Mistral checkpoints must be 1,2,4,5,8")
    if (v8_schema7.get("protocol_version"), v8_schema7.get("schema_version")) != (8, 7):
        errors.append("new v8 gradual-repair contract must use schema-v7")
    if v8_schema7.get("runtime_patch_mode") != "probekv_v8_winner_gradual_streaming":
        errors.append("schema-v7 contract uses another runtime patch mode")
    selection7 = v8_schema7.get("source_selection", {})
    if selection7.get("repair_support_from_score_trim_forbidden") is not True:
        errors.append("schema-v7 must separate Source score trimming from repair")
    repair7 = v8_schema7.get("winner_repair", {})
    if (
        repair7.get("fallback") != "fixed_15"
        or repair7.get("initial_repair_cap") != 0.15
        or repair7.get("support_monotonic") is not True
        or repair7.get("repair_reentry") != "none"
    ):
        errors.append("schema-v7 gradual repair safety contract is invalid")
    integrity7 = v8_schema7.get("integrity", {})
    if (
        integrity7.get("qualification_destination_digest_required") is not True
        or integrity7.get("online_per_request_full_digest_forbidden") is not True
    ):
        errors.append("schema-v7 integrity split is invalid")
    final7 = v8_schema7.get("admission", {}).get("final_commit_admission", {})
    if final7.get("accounting") != "request_joint_timeline" or final7.get("decision") != "ready_subset":
        errors.append("schema-v7 FinalCommitAdmission must preserve joint subset planning")
    if (v8_schema8.get("protocol_version"), v8_schema8.get("schema_version")) != (8, 8):
        errors.append("new v8 barrier contract must use schema-v8")
    if v8_schema8.get("runtime_patch_mode") != "probekv_v8_gradual_barrier_tiered_lru":
        errors.append("schema-v8 contract uses another runtime patch mode")
    selection8 = v8_schema8.get("source_selection", {})
    if (
        selection8.get("checkpoints") != [1, 2]
        or selection8.get("execution_policy") != "dense_selection_barrier"
        or selection8.get("first_reuse_layer_if_all_d1") != 2
        or selection8.get("first_reuse_layer_if_any_d2") != 3
    ):
        errors.append("schema-v8 d1/d2 dense-barrier contract is invalid")
    if (
        selection8.get("fallback_policy")
        != "legacy_multicheckpoint_three_gate"
        or selection8.get("fallback_mistral_checkpoints") != [1, 2, 4, 5, 8]
        or selection8.get("fallback_qwen_checkpoints") != [1, 2, 4, 5, 7]
        or selection8.get("fallback_requires_own_runtime_qualification") is not True
        or selection8.get("per_request_protocol_mixing_forbidden") is not True
    ):
        errors.append("schema-v8 legacy fallback contract is invalid")
    if selection8.get("fast_path_enables_only_after_pass") != [
        "dense_selection_barrier",
        "d1_d2_detached_winner_prefetch",
        "streamlined_final_commit_admission",
        "request_layer_uniform_io_balanced_repair",
    ]:
        errors.append("schema-v8 fast-only feature set is not frozen")
    if v8_schema8.get("gate1", {}).get("gamma") != 1.0:
        errors.append("schema-v8 Gate1 must use positive-saving gamma 1.0")
    if v8_schema8.get("final_commit_admission", {}).get("gamma") != 0.8:
        errors.append("schema-v8 final admission must use request gamma 0.8")
    storage8 = v8_schema8.get("storage", {})
    if (
        storage8.get("backing_policy") != "cpu_preferred_single_backing"
        or storage8.get("lru_timestamp")
        != "one_last_request_use_epoch_per_exclusive_backing"
        or storage8.get("cpu_eviction")
        != "exclusive_backing_lru_demote_to_ssd"
        or storage8.get("ssd_eviction")
        != "exclusive_backing_lru_delete_source"
        or storage8.get("busy_replica_eviction_forbidden") is not True
    ):
        errors.append("schema-v8 tiered backing contract is invalid")
    repair8 = v8_schema8.get("repair_ratios", {})
    if (
        repair8.get("load_recompute_aware_uniform")
        != "request_layer_uniform_io_balanced"
        or repair8.get("quality_reference_ratio") != 0.15
        or repair8.get("uniform_io_same_absolute_layer_all_active_segments")
        is not True
        or repair8.get("io_balance_ratio_candidates")
        != [0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    ):
        errors.append("schema-v8 uniform I/O repair contract is invalid")
    experiments8 = v8_schema8.get("experiments_h1_h5", {})
    h2_metrics8 = experiments8.get("H2_depth_policy_and_dispatch", {}).get(
        "hard_metrics", {}
    )
    if (
        set(experiments8) != {
            "H1_source_opportunity_and_correctness",
            "H2_depth_policy_and_dispatch",
            "H3_repair_quality",
            "H4_systems_and_storage",
            "H5_locked_end_to_end",
        }
        or experiments8.get("H2_depth_policy_and_dispatch", {}).get("decision")
        != "fast_selection_candidate_if_qualified_else_legacy_candidate"
        or experiments8.get("H5_locked_end_to_end", {}).get(
            "locked_test_only_here"
        )
        is not True
        or experiments8.get("H3_repair_quality", {}).get(
            "uniform_io_under_legacy"
        )
        != "diagnostic_only_not_runtime_eligible"
        or experiments8.get("H3_repair_quality", {}).get(
            "final_runtime_dispatch_after_profile_and_runtime_qualification"
        )
        is not True
        or h2_metrics8.get("state_availability_min") != 0.99
        or h2_metrics8.get("selection_coverage_min") != 0.80
        or h2_metrics8.get(
            "early_resolution_rate_at_completed_depth5_min"
        )
        != 0.80
        or h2_metrics8.get("wrong_early_lock_rate_max") != 0.05
        or h2_metrics8.get(
            "mean_stable_normalized_oracle_regret_max"
        )
        != 0.10
        or h2_metrics8.get(
            "selection_critical_path_p95_fraction_max"
        )
        != 0.05
        or h2_metrics8.get(
            "selection_budget_realized_overrun_rate_max"
        )
        != 0.05
    ):
        errors.append("schema-v8 H1-H5 experiment contract is incomplete")
    try:
        schema6_lock = json.loads(
            Path("configs/a800_server_lock_v8_schema6.json").read_text(
                encoding="utf-8"
            )
        )
        manifest_path = Path("patches/cacheblend/manifest.json")
        load_patch_manifest(manifest_path)
        schema6_patches = patch_files_for_mode(
            manifest_path, "probekv_v8_schema6_joint_cfo"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append("schema-v6 server/patch contract is invalid: %s" % error)
    else:
        if (
            schema6_lock.get("schema_version") != 6
            or schema6_lock.get("stack", {}).get("cacheblend_patch_mode")
            != "probekv_v8_schema6_joint_cfo"
        ):
            errors.append("schema-v6 server lock uses another runtime/patch mode")
        if len(schema6_patches) != 8:
            errors.append("schema-v6 CacheBlend patchset must contain eight patches")
        try:
            schema6_profile_lock = json.loads(
                Path("configs/a800_server_lock_v8_schema6_profile.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append("schema-v6 Profile lock is invalid: %s" % error)
        else:
            runtime = schema6_profile_lock.get("runtime", {})
            if (
                schema6_profile_lock.get("schema_version") != 6
                or runtime.get("freeze_runtime_cost_profile") is not True
                or runtime.get("run_140_job_qualification") is not False
                or runtime.get("run_h1") is not False
            ):
                errors.append("schema-v6 Profile lock crosses its frozen stop boundary")
    try:
        local_v6 = load_config("configs/local_system_v6.json")
    except (OSError, ValueError) as error:
        errors.append("local v6 config is invalid: %s" % error)
    else:
        if local_v6.protocol_version != 6:
            errors.append("local v6 config did not load as protocol 6")
    for name in (
        "configs/local_system_v7_causal_wait.json",
        "configs/local_system_v7_immediate_staggered.json",
        "configs/a800_h1_pilot_v7_mistral.json",
        "configs/a800_h1_pilot_v7_qwen.json",
    ):
        try:
            local_v7 = load_config(name)
        except (OSError, ValueError) as error:
            errors.append("local v7 config is invalid: %s" % error)
        else:
            if local_v7.protocol_version != 7:
                errors.append("local v7 config did not load as protocol 7")
    for name in (
        "configs/local_system_v8_causal_wait.json",
        "configs/local_system_v8_immediate_staggered.json",
        "configs/a800_h1_pilot_v8_mistral.json",
        "configs/a800_h1_pilot_v8_qwen.json",
    ):
        try:
            local_v8 = load_config(name)
        except (OSError, ValueError) as error:
            errors.append("v8 config is invalid: %s: %s" % (name, error))
        else:
            if local_v8.protocol_version != 8:
                errors.append("v8 config did not load as protocol 8: %s" % name)
    for name in (
        "configs/local_system_v8_schema6_causal_wait.json",
        "configs/local_system_v8_schema6_immediate_staggered.json",
    ):
        try:
            local_v8_schema6 = load_config(name)
        except (OSError, ValueError) as error:
            errors.append("v8 schema-v6 config is invalid: %s: %s" % (name, error))
        else:
            if local_v8_schema6.v8_schema_version != 6:
                errors.append("v8 schema-v6 config did not select schema-v6: %s" % name)
    for name in (
        "configs/local_system_v8_schema7_legacy_fixed15.json",
        "configs/local_system_v8_schema7_qwen_legacy_fixed15.json",
        "configs/local_system_v8_schema7_d1d2_gradual_causal_wait.json",
        "configs/local_system_v8_schema7_d1d2_gradual_immediate.json",
    ):
        try:
            local_v8_schema7 = load_config(name)
        except (OSError, ValueError) as error:
            errors.append("v8 schema-v7 config is invalid: %s: %s" % (name, error))
        else:
            if local_v8_schema7.v8_schema_version != 7:
                errors.append("v8 schema-v7 config did not select schema-v7: %s" % name)
    try:
        local_v8_schema8 = load_config(
            "configs/local_system_v8_schema8_d1d2_gradual_barrier.json"
        )
    except (OSError, ValueError) as error:
        errors.append("v8 schema-v8 config is invalid: %s" % error)
    else:
        if (
            local_v8_schema8.v8_schema_version != 8
            or local_v8_schema8.gate1_gamma != 1.0
            or local_v8_schema8.gamma != 0.8
        ):
            errors.append("v8 schema-v8 config did not freeze Gate scopes")
    try:
        schema7_patches = patch_files_for_mode(
            Path("patches/cacheblend/manifest.json"),
            "probekv_v8_winner_gradual_streaming",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append("schema-v7 CacheBlend patch contract is invalid: %s" % error)
    else:
        if len(schema7_patches) != 8:
            errors.append("schema-v7 CacheBlend patchset must contain eight patches")
    try:
        schema8_patches = patch_files_for_mode(
            Path("patches/cacheblend/manifest.json"),
            "probekv_v8_gradual_barrier_tiered_lru",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append("schema-v8 CacheBlend patch contract is invalid: %s" % error)
    else:
        if len(schema8_patches) != 8:
            errors.append("schema-v8 CacheBlend patchset must contain eight patches")
    try:
        schema7_lock = json.loads(
            Path("configs/a800_server_lock_v8_schema7.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        errors.append("schema-v7 server lock is invalid: %s" % error)
    else:
        runtime7 = schema7_lock.get("runtime", {})
        if (
            schema7_lock.get("schema_version") != 7
            or schema7_lock.get("stack", {}).get("cacheblend_patch_mode")
            != "probekv_v8_winner_gradual_streaming"
            or runtime7.get("selector_depth_profile_frozen") is not False
            or runtime7.get("repair_policy_profile_frozen") is not False
            or runtime7.get("runtime_cost_profile_frozen") is not False
            or runtime7.get("run_140_job_qualification") is not False
            or runtime7.get("run_h1") is not False
        ):
            errors.append("schema-v7 server lock crosses its no-GPU stop boundary")
    try:
        schema8_lock = json.loads(
            Path("configs/a800_server_lock_v8_schema8.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        errors.append("schema-v8 server lock is invalid: %s" % error)
    else:
        runtime8 = schema8_lock.get("runtime", {})
        capabilities8 = set(runtime8.get("required_capabilities", []))
        if (
            schema8_lock.get("schema_version") != 8
            or schema8_lock.get("stack", {}).get("cacheblend_patch_mode")
            != "probekv_v8_gradual_barrier_tiered_lru"
            or runtime8.get("selection_execution_policy")
            != "dense_selection_barrier"
            or runtime8.get("run_140_job_qualification") is not False
            or runtime8.get("run_h1") is not False
        ):
            errors.append("schema-v8 server lock crosses its no-GPU stop boundary")
        if "legacy_multicheckpoint_three_gate_fallback" not in capabilities8:
            errors.append("schema-v8 server lock omits the legacy fallback capability")
    main_checkpoints = contract["probe_policy"]["main_32_layer_checkpoints"]
    if main_checkpoints != list(range(1, 9)):
        errors.append("32-layer main probe policy must inspect every layer 1-8")
    if max(main_checkpoints) > 32 * invariants["max_probe_fraction"]:
        errors.append("main probe checkpoints exceed the 25% ceiling")
    if max(contract["repair_ratios"]) != 1.0 or min(contract["repair_ratios"]) != 0.0:
        errors.append("repair grid must cover [0, 1]")
    hardware = contract["hardware"]
    primary_hardware = hardware["primary"]
    if primary_hardware["role"] != "formal_matched_primary":
        errors.append("primary hardware must be the formal matched primary")
    if primary_hardware["gpu_count"] != 1:
        errors.append("primary experiment contract requires one GPU")
    if primary_hardware["compute_capability"] != "8.0":
        errors.append("primary A800 compute capability must be 8.0")
    if not hardware["matched_hardware_required_for_all_direct_baselines"]:
        errors.append("all direct baselines must use matched hardware")
    if contract["evidence_policy"]["stage1_cb0_h0_performance_is_paper_evidence"]:
        errors.append("stage1 CB0/H0 performance must remain non-paper evidence")
    tail_minimum = minimum_zero_violation_trials(
        contract["quality"]["tail_violation_upper_bound"], 0.95
    )
    pooled_cases = sum(dataset["locked_test"] for dataset in contract["datasets"]["rag"])
    if contract["quality"]["tail_gate_scope"] == "pooled_three_rag_datasets_per_model":
        if pooled_cases < tail_minimum:
            errors.append("pooled tail gate has insufficient cases")
    matrix_counts = {
        "main_rag_cells_before_replays": sum(1 for _ in main_rag_matrix()),
        "profile_cells_without_ssd": sum(1 for _ in profile_matrix(False)),
        "profile_cells_with_ssd": sum(1 for _ in profile_matrix(True)),
        "minimum_cases_for_zero_violation_exact_95pct_upper_1pct": tail_minimum,
        "pooled_rag_cases_per_primary_model": pooled_cases,
    }
    result = {
        "contract": str(contract_path.resolve()),
        "v8_contract": str(v8_contract_path.resolve()),
        "v8_schema6_contract": str(v8_schema6_path.resolve()),
        "v8_schema7_contract": str(v8_schema7_path.resolve()),
        "v8_schema8_contract": str(v8_schema8_path.resolve()),
        "valid": not errors,
        "errors": errors,
        "matrix_counts": matrix_counts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
