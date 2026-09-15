# Single-Segment recovery: implementation and evidence boundary

Baseline: `bc16c24`. Scope: Mistral 512 tokens, then 128/640;
no multi-Segment, Qwen, multi-Source gain or paper qualification.

## Implemented locally

- Deferred-timing capability caching keys the unbound implementation, not a
  bound method which retains its model owner. Missing source fails closed.
- Current-K FP32 conversion and normalization occur once per checkpoint,
  shared across microbatches under a separately retained HBM reservation.
  No values survive into the next checkpoint or request. Stable argsort,
  exact integer trim and reduction order remain unchanged.
- Native dense continuation verifies original paged KV and attention ownership
  before submitting any layer. It remains opt-in and is **not GPU qualified**.
- `run_locked_single_segment.py` independently rebuilds the chosen patch tree,
  checks the actual index/worktree and model asset file hashes, pins imports
  in one process, records extension hashes and creates `environment_lock.json`.
  It accepts no supplied patch audit and cannot promote a tree-only claim.
  Without `--execute`, it performs preparation only, not model inference.

The old `patch-audit-418b0628-schema10.json` was manually annotated without an
independent rebuild. It is invalid evidence; preserve it but do not consume it.
The new launcher produces its own audit using the existing independent verifier.

Local acceptance: 901 tests, 900 passed and one pre-existing skip; compileall,
contract validation and diff check passed. CPU arithmetic equivalence and
reservation cleanup tests do not establish GPU numerical/performance evidence.

## Reproducible first remote run

Use a new exact-SHA ProbeKV clone, not a file copied into an older checkout.
No global editable install changes are required. From that clone on Linux:

```bash
RUN_SHA=$(git rev-parse HEAD)
/root/autodl-tmp/probekv_stage1/envs/cacheblend-cu121/bin/python \
  scripts/server/run_locked_single_segment.py \
  --cacheblend /root/autodl-tmp/probekv_stage2/src/CacheBlend-matched-4881010-0016 \
  --model-audit /root/autodl-tmp/probekv_stage2/artifacts/model-audit-9e296fef-mistral.json \
  --config configs/a800_server_lock_v8_schema10.json \
  --output /root/autodl-tmp/probekv_stage2/artifacts/locked-${RUN_SHA}-512-baseline-v1 \
  --segment-tokens 512 --execute
```

The historical filename of the asset audit is not a code-SHA attestation. Its
actual listed files are fully rehashed and its content digest bound to the new
environment lock. A mismatch stops before inference. All patch/model/setup
hashing costs are explicitly preparation time, not online request TTFT.

If tree verification fails, inspect `independent_patch_rebuild.log`. Rebuild in
a fresh directory from the manifest base; never append patches to a historical
dirty worktree and never edit an audit to match. If only the real index is stale,
the launcher refuses it rather than silently staging an existing directory.

First run preserves baseline flags. Subsequent runs add exactly one control:
`--defer-layer-timing`, then owned-host validation, then native continuation
after numerical validation. Each uses a fresh directory. No mixed-SHA cost table.

## Remaining work (not claimed complete)

1. Restore SSH access to the requested 46068 instance. On 2026-09-15 the port
   refused the connection before authentication; no new remote GPU job started.
2. Execute the independent environment check and real 512-token correctness /
   cost probe. The new launcher itself has not yet been validated on this GPU.
3. Validate native continuation against dense/teacher references (32 positions,
   relative-L2 <=1e-4), token IDs, Prefix/block ownership and cleanup.
4. Recollect exact costs and rerun production FinalCommit. No forced commit.
5. Validate arithmetic optimization and ownership guards on GPU; measure
   normalization savings versus the extra workspace reservation bookkeeping.
6. Audit request-static digest caching separately from dynamic readiness/epoch
   snapshots; these caches are **not yet implemented**. Source tensor stacking
   and cross-checkpoint scratch reuse remain unoptimized.
7. Run 20 interleaved measured pairs plus two excluded warmup pairs per accepted
   factor, three cold processes separately, followed by 128/640 boundary tests.

Do not add nested profiler times to the wall-clock partition. Do not claim
speedup from local tests or confuse a fixed-Source future with full TTFT.
Keep gamma=0.8; if the measured execution lower limit leaves no admission
headroom, report the limitation instead of hiding preparation costs.

All runtime/positive-gain/profile/paper gates remain unpassed for this new SHA.
