# ProbeKV schema10 A800 Profile runbook

This stage freezes model-specific Profiles only. It is non-paper evidence and
must not run runtime qualification, H1-H5, or the locked test.

The server checkout must be clean and equal the handoff `code_commit`. Model
revision, tokenizer hash, CacheBlend patch/tree and development partition must
match, including the exact tokenized development case manifest hash. The runner
accepts one A800 80GB, at most four hours, and a declared
price no greater than 7.5 CNY/hour.

```bash
export PYTHONPATH="$PWD/src"
python scripts/server/run_v8_schema10_a800_profile.py \
  --model-key mistral \
  --handoff artifacts/schema10/mistral/handoff.json \
  --development-manifest artifacts/schema10/mistral/development.jsonl \
  --config configs/local_system_v8_schema10_gate1_barrier.json \
  --contract configs/experiment_contract_v8_schema10.yaml \
  --server-lock configs/a800_server_lock_v8_schema10.json \
  --case-manifest artifacts/schema10/mistral/manifest.json \
  --model-audit artifacts/schema10/mistral/model_audit.json \
  --patch-audit artifacts/cacheblend_patch_audit.json \
  --cacheblend /path/to/cacheblend \
  --ssd-staging /data/probekv-stage \
  --output results/schema10/mistral-profile-attempt-001 \
  --hourly-price-cny 7.5 --max-hours 4
```

Use `--resume` only with the same immutable successful prefix. A failed row is
never overwritten; a code fix requires a new SHA and output directory. The
first job is the native Prefix Cache/K-hook/r=1/digest/mask correctness
sentinel. Ninety-case sweeps use six-case shards. A deadline stop never freezes
a partial Profile.

After every measurement job succeeds:

```bash
python scripts/server/aggregate_v8_schema10_profile.py \
  --results results/schema10/mistral-profile-attempt-001/results.jsonl \
  --runtime-audit results/schema10/mistral-profile-attempt-001/runtime_audit.json \
  --development-manifest artifacts/schema10/mistral/development.jsonl \
  --output results/schema10/mistral-profile-attempt-001/profile_measurements.json

python scripts/server/freeze_v8_schema10_profiles.py \
  --measurements results/schema10/mistral-profile-attempt-001/profile_measurements.json \
  --code-commit '<exact-sha>' \
  --cacheblend-patch-sha256 '<patch-sha256>' \
  --model-key mistral --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --model-revision '<revision>' --tokenizer-hash '<tokenizer-sha256>' \
  --development-partition-sha256 '<partition-sha256>' \
  --development-case-manifest-sha256 '<case-manifest-sha256>' \
  --output-dir results/schema10/mistral-profile-attempt-001/frozen
```

The first Profile implementation promotes `static_gradual` or
`load_recompute_aware_uniform` only when a complete per-layer no-reentry and
quality-floor trace is present. If that evidence is incomplete, the aggregator
must explicitly record the rejection reason and freeze the safe `fixed_15`
fallback; it must not infer a gradual schedule from isolated ratio timings.

Repeat independently for Qwen. Never reuse Mistral thresholds, tokenizer data,
repair/runtime measurements, Gate1 mode or Variant namespace for Qwen.

After both per-model bundles freeze, create the joint non-paper Gate:

```bash
python scripts/server/combine_v8_schema10_profile_bundles.py \
  --mistral-bundle results/schema10/mistral-profile-attempt-001/frozen/profile_bundle_manifest.json \
  --qwen-bundle results/schema10/qwen-profile-attempt-001/frozen/profile_bundle_manifest.json \
  --output results/schema10/dual-model-profile-freeze-gate.json
```

Success freezes five Profiles and leaves runtime qualification, quality-tail
certification and H1/H2 disabled.
