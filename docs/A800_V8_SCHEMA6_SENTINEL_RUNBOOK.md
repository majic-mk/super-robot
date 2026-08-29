# Mistral A800 schema-v6 sentinel runbook

This session is non-paper runtime qualification preparation.  Do not start it
unless the exact Git SHA is pushed and the platform offers one A800 80GB for at
most CNY 7.5/hour.  Stop after four hours or CNY 30, whichever comes first.

## Before GPU allocation

Run:

```bash
export PYTHONPATH="$PWD/src"
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_contract.py
python -m probekv.cli --config configs/local_system_v8_schema6_causal_wait.json
python -m probekv.cli --config configs/local_system_v8_schema6_immediate_staggered.json
git diff --check
```

Generate the sentinel manifest from `probekv.v8_schema6_jobs` and bind the
exact code, tokenizer, CacheBlend patch/tree, config and contract hashes.
Server code must be a detached checkout of that SHA with a clean worktree.

```bash
scripts/server/prepare_cacheblend.sh \
  "$STAGE/src/cacheblend-schema6" probekv_v8_schema6_joint_cfo
MAX_JOBS=8 PIP_NO_CACHE_DIR=1 \
  "$STAGE/envs/cacheblend-cu121/bin/python" -m pip install \
  --no-deps --no-build-isolation -e "$STAGE/src/cacheblend-schema6/vllm_blend"
python scripts/server/verify_cacheblend_patch.py \
  --cacheblend "$STAGE/src/cacheblend-schema6" \
  --mode probekv_v8_schema6_joint_cfo \
  --output "$STAGE/artifacts/schema6/patch_audit.json"
python scripts/server/build_v8_schema6_sentinel_handoff.py \
  --model-audit "$STAGE/artifacts/model_audit_mistral.json" \
  --patch-audit "$STAGE/artifacts/schema6/patch_audit.json" \
  --output "$STAGE/artifacts/schema6/manifest.json"
```

## GPU order and hard stops

1. Verify A800 80GB, compute capability 8.0, CUDA stack and GPU UUID.
2. Run native Prefix Cache, K-hook/depth and `r=1` dense-equivalence sentinels.
3. Run CFO eager-versus-streaming reference.
4. Run elastic SelectionState comparison and the A/C Gate2/Gate3 closure.
5. Run only the sparse grid in `configs/v8_schema6_a800_sentinel.json`.
6. Archive raw CUDA Event samples, request wall clock, stdout/stderr and hashes.
7. Stop.  Do not freeze profiles, run qualification jobs or launch H1.

The bounded runner requires the platform price as an explicit input and
refuses prices above the frozen limit:

```bash
python scripts/server/run_v8_schema6_mistral_sentinel.py \
  --manifest "$STAGE/artifacts/schema6/manifest.json" \
  --model-audit "$STAGE/artifacts/model_audit_mistral.json" \
  --patch-audit "$STAGE/artifacts/schema6/patch_audit.json" \
  --cacheblend "$STAGE/src/cacheblend-schema6" \
  --output "$STAGE/artifacts/schema6/a800-run-001" \
  --hourly-price-cny 5.98
```

The runner resolves `vllm` before model loading and rejects an editable install
that still points at an older CacheBlend worktree, even when the standalone
patch audit is otherwise valid.

Any token mismatch, logit threshold failure, Source digest mutation, illegal
lease/transfer, stale Planner decision or incorrect CFO accumulator stops the
session.  OOM/reset/crash evidence is retained and never removed as an outlier.
