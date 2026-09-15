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

Additional no-GPU changes:

- Disabled selection-result caching now skips cache-key/pool digest creation
  and decision publication. The explicit exact-request diagnostic still works.
- Request token identity is memoized with a content comparison; mutations
  invalidate it. Prefix and sampling remain live. Execution-shape queries no
  longer build unused legacy identity dictionaries. Historical sampling string
  signatures remain readable.
- The initial pool snapshot digest is computed once per request. Dynamic
  scheduler/ready/generation/placement checks still execute before commit.
- Unsupported dense outcomes now contain the complete host partition.
  Selection events have names and raw nanosecond bounds. The offline analyzer
  splits nested intervals once and rejects inconsistent endpoints/accounting.
- The launcher also verifies the imported vLLM extension location, preserves
  interrupted/nonzero-exit failure evidence, and forwards paired executor and
  owned-position controls to their existing implementations.
- The recovery manifest generator binds the committed code/config and emits
  36 ordered tasks with actual argv, dependencies and explicit timing scopes.
  Runtime asset/cost hashes remain null until independently measured.

Local acceptance: 914 tests, 913 passed and one pre-existing skip; 24 local CLI
configurations, compileall, contract validation and diff check passed. CPU arithmetic equivalence and
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

## Freeze the GPU task schedule after committing

```powershell
$env:PYTHONPATH='src'
python scripts/server/generate_single_segment_recovery_plan.py --python /root/autodl-tmp/probekv_stage1/envs/cacheblend-cu121/bin/python --cacheblend /root/autodl-tmp/probekv_stage2/src/CacheBlend-matched-4881010-0016 --model-audit /root/autodl-tmp/probekv_stage2/artifacts/model-audit-9e296fef-mistral.json --remote-output-root /root/autodl-tmp/probekv_stage2/artifacts --output artifacts/local-no-gpu-recovery-20260915/gpu-plan.json
```

Run job argv from the exact clean checkout, with the manifest's explicit
PYTHONPATH. Dependencies must pass; this schedule does not execute jobs itself.
The first job is 512-token environment/correctness/cost collection. Online jobs
consume that same SHA's evidence through the existing cost validator. A failed
prerequisite stops dependent jobs. Never resume by overwriting outputs.

Twenty-pair same-process measurements are wired for the existing host-position
and fixed-Source backend comparisons. The online 22-replay sweeps provide two
warmups and 20 measured warm requests; separate sweeps are not paired A/B
evidence. Independent-process correctness runs record setup/cold initialization,
not cold online TTFT. Do not infer a paired confidence interval or cold online
speedup from those sweeps. Additional online paired/cold instrumentation, if
needed for a default promotion, follows the first real correctness results.

The historical raw outcome `online-27c2577-hostpos-r3-v5/outcome-00.json`
was reanalyzed locally: **105244413 ns** accounted, **0 ns** unaccounted.
Its source SHA remains `27c2577`; this is an accounting check, not a new GPU run.

## Remaining GPU work (not claimed complete)

1. Restore SSH access to the requested 46068 instance. On 2026-09-15 the port
   refused the connection before authentication; no new remote GPU job started.
2. Execute the independent environment check and real 512-token correctness /
   cost probe. The new launcher itself has not yet been validated on this GPU.
3. Validate native continuation against dense/teacher references (32 positions,
   relative-L2 <=1e-4), token IDs, Prefix/block ownership and cleanup.
4. Recollect exact costs and rerun production FinalCommit. No forced commit.
5. Validate arithmetic optimization and ownership guards on GPU; measure
   normalization savings versus the extra workspace reservation bookkeeping.
6. Measure whether Source stacking or reservation overhead is material before
   adding further workspace changes. Cross-checkpoint data reuse is forbidden.
7. Run 20 interleaved measured pairs plus two excluded warmup pairs per accepted
   factor, three cold processes separately, followed by 128/640 boundary tests.

Do not add nested profiler times to the wall-clock partition. Do not claim
speedup from local tests or confuse a fixed-Source future with full TTFT.
Keep gamma=0.8; if the measured execution lower limit leaves no admission
headroom, report the limitation instead of hiding preparation costs.

All runtime/positive-gain/profile/paper gates remain unpassed for this new SHA.
Local preparation is ready for the first GPU environment/correctness stage.
Actual server assets, imports, GPU availability and numerical correctness must
still pass onsite; online trace is conditional on valid measured cost support.
