# ProbeKV v7 A800 runbook

This is the operational boundary for protocol v7. Every output described here
is non-paper qualification evidence with `paper_evidence=false` and
`locked_test_accessed=false`.

## Frozen runtime contract

- one A800 80GB, compute capability 8.0;
- PyTorch 2.2.1 CUDA 12.1, xformers 0.0.25, vLLM 0.4.1;
- CacheBlend base `b72d7945e6d6306f12be66520196e0f081fa2b0c`;
- patch mode `probekv_v7_single_artifact_runtime`;
- one BF16 pre-RoPE/raw-V Artifact per Source Variant;
- one backing Replica plus optional transient tier Replicas;
- repair count uses conservative ceil rounding;
- alignment quantum and frozen vLLM block size are both 16.

An alignment mismatch rejects this frozen experiment contract. It is not a
claim that KV reuse is mathematically incorrect.

## No-GPU preparation

From a clean checkout of the intended commit:

```bash
export PYTHONPATH="$PWD/src"
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_contract.py
python -m probekv.cli --config configs/local_system_v7_causal_wait.json
python -m probekv.cli --config configs/local_system_v7_immediate_staggered.json
python scripts/server/audit_v7_runtime_sources.py \
  --repo "$PWD" --output "$STAGE/runtime_source_audit_v7.json"
git diff --check
```

Apply and audit the pinned CacheBlend patchset with the existing server setup
scripts. Build the Mistral and Qwen 140-job manifests independently:

```bash
python scripts/server/build_v7_a800_jobs.py \
  --model-key mistral --model-audit "$STAGE/model_audit_mistral.json" \
  --patch-audit "$STAGE/cacheblend_patch_audit.json" \
  --output "$STAGE/qualification-mistral"

python scripts/server/build_v7_a800_jobs.py \
  --model-key qwen --model-audit "$STAGE/model_audit_qwen.json" \
  --patch-audit "$STAGE/cacheblend_patch_audit.json" \
  --output "$STAGE/qualification-qwen"
```

Build each model's v7 data handoff with
`scripts/server/prepare_v6_h1_model_data.py --protocol-version 7`. Despite the
legacy filename, this mode emits a schema-v3 v7 handoff and model-specific
canonical content keys. Mistral token manifests must never be reused by Qwen.
Use `configs/a800_h1_pilot_v7_mistral.json` and
`configs/a800_h1_pilot_v7_qwen.json`; v6 H1 configs are intentionally rejected
as v7 evidence.

Run `verify_v7_dual_model_no_gpu_readiness.py` last. A passing no-GPU gate sets
rental readiness true while leaving GPU qualification and H1/H2 false.

## Per-model A800 session

Run Mistral first and Qwen second. Use a new output directory after a hard
failure. The sentinel invocation is:

```bash
python scripts/server/run_v7_a800_qualification.py \
  --model-key mistral \
  --jobs "$STAGE/qualification-mistral/jobs_mistral.jsonl" \
  --job-manifest "$STAGE/qualification-mistral/manifest_mistral.json" \
  --model-audit "$STAGE/model_audit_mistral.json" \
  --patch-audit "$STAGE/cacheblend_patch_audit.json" \
  --output "$RUN/mistral" --sentinel-only
```

After the real Prefix Cache and `r=1` sentinel pass, run five canary jobs and
resume the immutable result prefix to 140/140. Repeat for Qwen. Generate each
schema-v3 gate with `verify_v7_runtime_qualification.py`.

The gate rejects host/fake timing, any failed or missing job, missing native
Prefix block metadata, Source or Artifact digest mutation, non-ceil repair,
wrong model/commit/patch/tree/GPU, and any v6 qualification artifact.

## One-case H1 sentinel

After the matching model gate passes, invoke `run_v7_h1_pilot.py` with
`--pass primary --case-limit 1 --max-hours 1` and the model's manifest, jobs,
handoff, audit and qualification gate. Mistral uses primary layer 5; Qwen uses
layer 4. Each sentinel must complete four Sources by nine ratios, producing
exactly 36 rows, with `r=1` checked for every Source.

Only `build_v7_joint_gate.py` may combine the two model gates and two successful
H1 sentinels. It emits `ready_for_full_h1_pilot=true` and always leaves
`full_h1_started=false`.
