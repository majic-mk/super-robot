# ProbeKV v8 schema10 no-GPU handoff

The no-GPU handoff contains two local policies:

```text
configs/local_system_v8_schema10_gate1_barrier.json
configs/local_system_v8_schema10_gate1_counterfactual.json
```

Both use the same dynamic Variant protocol. The second treats Gate1 as an
advisory only for contract simulation; it does not establish that the advisory
mode is qualified. Real development observations must freeze a distinct
`PreparationPolicyProfile` for every model and runtime policy.

Before server checkout run:

```powershell
$env:PYTHONPATH="$PWD/src"
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_contract.py
python -m probekv.cli --config configs/local_system_v8_schema10_gate1_barrier.json
python -m probekv.cli --config configs/local_system_v8_schema10_gate1_counterfactual.json
git diff --check
git status --short
```

Generate each model's immutable handoff with
`scripts/server/build_v8_schema10_no_gpu_handoff.py`. The first A800 sessions
freeze VariantAdmission, PreparationPolicy and SelectionDepth Profiles only;
they do not run qualification or H1. Mistral and Qwen use independent model,
tokenizer and Variant namespaces.

Expected stop state:

```text
artifact_preparation_ready=true
gpu_rental_ready_for_schema10_profile_freeze=true
variant_admission_profile_frozen=false
preparation_policy_profile_frozen=false
gpu_runtime_qualified=false
h1_h2_execution_allowed=false
full_h1_started=false
paper_evidence=false
locked_test_accessed=false
```
