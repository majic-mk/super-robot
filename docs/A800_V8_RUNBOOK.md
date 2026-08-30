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
retained, but v8 output is schema-v5. Mistral token IDs, content keys and
Sources must never be reused by Qwen.

The frozen 90-case development partition must be sampled only from the
pre-isolated `calibration`/development role. The builder rejects train,
H1-pilot and test rows so Profile freeze cannot overlap later H1 evidence.

## Schema-v6 Mistral Runtime Profile session

Schema-v6 does not reuse the schema-v5 comparison-only Profile freezer.  After
the 54/54 sparse sentinel passes at the current code SHA, build a policy-bound
handoff with `build_v8_schema6_runtime_profile_handoff.py`, then run
`run_v8_schema6_mistral_runtime_profile.py` in a new immutable output
directory.  The runner executes 155 cells with 20 warmups and 100 retained
samples, freezes all eight RuntimeCostProfile categories, and binds the
Profile to the actual GPU UUID.  Run A and C independently.  It does not run
the 90-case development selector sweep, 140-job qualification or H1.

If the code changes after the sentinel, rerun the sentinel first.  Failed
Profile output is never resumed; successful prefixes may resume with
`--resume`.  SSD cells must use a staging directory on the data disk.

## A800 order for each Model x A/C Profile

1. Verify A800 80GB, CC 8.0, stack, block size 16, exact SHA and clean tree.
2. Run Prefix Cache, completed-depth K-hook and `r=1` sentinels.
3. Run SelectionState microbenchmarks for depths, `K={1,2,4,8,16}` and CPU/SSD
   source tiers into bounded GPU scratch.
4. Freeze `RuntimeCostProfile` (`R_profile`) and run the pooled 90-case
   development/profile-freeze partition.
5. Freeze `SelectorPolicyProfile` with `freeze_v8_selector_profile.py`; it must
   bind the precommitted Profile-freeze contract and `R_profile`.
6. On the actual qualification GPU, reuse `R_profile` only if still valid;
   otherwise measure and freeze `R_qual`. Generate one 140-job manifest for
   this exact Model x A/C profile with `build_v8_profile_bound_jobs.py`.
7. Run a five-job canary in a fresh output directory.
8. Run final 140/140 qualification in another immutable directory.
9. Build schema-v5 Gate with `verify_v8_runtime_qualification.py`.
10. Run one case using `run_v8_h1_pilot.py --case-limit 1 --pass primary`.
11. Stop; do not automatically start full H1.

Repeat this independently for Mistral-A, Mistral-C, Qwen-A and Qwen-C: 560 jobs
in total. Qualification must follow Profile freeze. An older schema-v4 Gate,
Profile, code SHA, tokenizer, patch tree, job manifest or qualification runtime
profile cannot unlock schema-v5 H1.

## Failure rules

- No real Prefix Cache block hit: stop; never infer from TTFT.
- K hook, independent SelectionState backing, RoPE, union mask, digest, `r=1`
  or CUDA timing failure: stop.
- Any qualification job failure: retain evidence and do not run H1.
- Full-KV prefetch before Source freeze or non-winner transfer: stop.
- Selection workspace is leased from the unified HBM manager with a fixed 4 GiB
  reserve. Use one vectorized comparison when it fits, otherwise use the
  largest deterministic microbatch; do not impose the old 256 MiB schema-v5 cap.
- Stale Replica: at most two same-Source replans; never change Variant.
- Resume only an immutable successful result prefix; never overwrite failure.

Release the cloud instance from the provider console after archiving logs;
shutdown inside SSH may not stop billing.
