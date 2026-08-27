# ProbeKV v8 A800 runbook

Run from a clean checkout of the final GitHub `main` SHA. Credentials, weights,
datasets and raw results stay outside Git.

## No-GPU preparation

```bash
export PYTHONPATH="$PWD/src"
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_contract.py
python -m probekv.cli --config configs/local_system_v8_causal_wait.json
python -m probekv.cli --config configs/local_system_v8_immediate_staggered.json
python scripts/server/audit_v8_runtime_sources.py \
  --repo . --output "$STAGE/runtime_sources_v8.json"
```

Prepare the fixed CacheBlend tree with patch mode
`probekv_v8_training_free_residual_k`. Generate all four Model x A/C
pre-Profile handoffs with `build_v8_no_gpu_task_manifests.py`, then run
`verify_v8_dual_model_no_gpu_readiness.py`. A passing no-GPU Gate must still
report `selector_profile_frozen=false`, `gpu_runtime_qualified=false` and
`h1_h2_execution_allowed=false`.

Prepare model-specific H1 data with
`prepare_v6_h1_model_data.py --protocol-version 8`. The legacy filename is
retained, but v8 output is schema-v4. Mistral token IDs, content keys and
Sources must never be reused by Qwen.

## A800 order for each Model x A/C Profile

1. Verify A800 80GB, CC 8.0, stack, block size 16, exact SHA and clean tree.
2. Run Prefix Cache, completed-depth K-hook and `r=1` sentinels.
3. Run SelectionState microbenchmarks for depths, `K={1,2,4,8,16}` and tiers.
4. Run pooled three-dataset development/profile-freeze tasks.
5. Freeze `SelectorPolicyProfile` with `freeze_v8_selector_profile.py`.
6. Generate Profile-bound jobs with `build_v8_profile_bound_jobs.py`.
7. Run a five-job canary in a fresh output directory.
8. Run final 140/140 qualification in another immutable directory.
9. Build schema-v4 Gate with `verify_v8_runtime_qualification.py`.
10. Run one case using `run_v8_h1_pilot.py --case-limit 1 --pass primary`.
11. Stop; do not automatically start full H1.

Qualification must follow Profile freeze. An older Profile, code SHA,
tokenizer, patch tree, job manifest or GPU invalidates the Gate.

## Failure rules

- No real Prefix Cache block hit: stop; never infer from TTFT.
- K hook, independent SelectionState backing, RoPE, union mask, digest, `r=1`
  or CUDA timing failure: stop.
- Any qualification job failure: retain evidence and do not run H1.
- Full-KV prefetch before Source freeze or non-winner transfer: stop.
- Selection scratch above 256 MiB: profile smaller microbatches; do not silently raise it.
- Stale Replica: at most two same-Source replans; never change Variant.
- Resume only an immutable successful result prefix; never overwrite failure.

Release the cloud instance from the provider console after archiving logs;
shutdown inside SSH may not stop billing.
