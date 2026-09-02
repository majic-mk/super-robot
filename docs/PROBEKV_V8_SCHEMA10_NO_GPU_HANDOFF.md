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
freeze SelectionDepth, VariantAdmission, RepairPolicy, RuntimeCost and
PreparationPolicy Profiles only; they do not run qualification or H1. Mistral
and Qwen use independent model, tokenizer and Variant namespaces.

The handoff must bind the exact SHA256 of the schema10 config, experiment
contract, `configs/a800_server_lock_v8_schema10.json`, and the isolated
90-request development partition and its exact tokenized case manifest. A
handoff without any one of these five hashes cannot start the Profile runner.

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

The revised Profile stage freezes five independent artifacts per model:
SelectionDepth, VariantAdmission, RepairPolicy, RuntimeCost and
PreparationPolicy. Run `run_v8_schema10_a800_profile.py`, aggregate with
`aggregate_v8_schema10_profile.py`, and freeze only complete evidence with
`freeze_v8_schema10_profiles.py`. Ninety-case selection and repair sweeps are
stored as immutable six-case shards for bounded resume.

The 90 development cases do not certify a 1% quality-tail violation rate.
After both model Profile bundles freeze, runtime qualification remains 140 jobs
per final model dispatch (280 total), while quality certification is a separate
later set of at least 300 unique requests per model.
