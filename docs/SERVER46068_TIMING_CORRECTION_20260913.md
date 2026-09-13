# Single-Segment timing correction and projection deduplication

## Corrections to the v44 interpretation

The v44 online run used server commit `9e296fef40c6c50c2f8d14c818d092d950a3e2f7`.
Its admission accounting omitted elapsed intervals outside the selection ledger.
That change (local `63f011c`) was incorrect: an arrival-to-first-token comparison
must retain queueing, setup, shared blocks, selection and preparation in actual
elapsed time. A future-only interval does not include already completed blocks.
This revision restores full elapsed wall time; v44 must not qualify an admission
policy. Original raw files remain intact.

The v43/v44 single warm observations (99.27 vs 91.10 ms) are not a controlled
speedup experiment. Changed costs and normal timing variation prevent attributing
the difference to the synchronization change. No 8 ms optimization is established.

## What the raw execution evidence actually shows

Source: `native-9e296fef-server46068-v44-costprobe/cost-probe/` on server 46068.

| Arm | Full first-token host ms | Boundary future ms | Ready future ms |
| --- | ---: | ---: | ---: |
| Native Prefix dense | 58.006775 | n/a | n/a |
| Resumable dense | 89.078415 | 77.044910 | n/a |
| Fixed15 streaming, fixed winner | 53.802542 | 44.596916 | 42.718220 |
| Fixed15 all-ready, fixed winner | 56.542014 | 47.742605 | 42.110571 |

The fixed15 arms **did** commit Source C at layer 2. Their layer audits match the
expected mask: layer 1 has 704 active rows; layers 2--32 have 160 rows (96 repair
rows plus 64 mandatory non-Segment rows). Thus repair-mask wiring works in the
fixed-winner executor. These controls exclude live Source selection and are not
online policy wins or QA-qualified performance results.

Online rejection correctly retains all 704 rows. It is not evidence of a mask bug
after commit. The previous claim that active-row reduction was unimplemented is
withdrawn. For this 640-token case the full range is 256--959, not 256--831.

## Actual fixes

1. Restore FinalCommit `actual_sunk_ms = now - arrival`. Add a regression test
   that keeps 200 ms of pre-selection waiting in the predicted request total.
2. The streaming joint-cost cell was sampled from selection boundary, although
   queried after winner preparation/repair-check. Use its measured
   ready-to-first-token interval instead. This avoids recharging 1.878696 ms in
   the cited sample. Keep boundary-future data for Source-local prediction.
3. Pinned selection already performs a fused QKV projection. Cache that current
   K/V observation inside the request session for the same-depth repair check.
   Invalidate on layer advance, commit, and finish. No Source-side V transfer,
   new quality threshold, or admission bypass is introduced.

The native dense-continuation experiment remains disabled by default because its
numerical qualification previously failed. Do not restart a request from layer 1
and call it a free fallback. New GPU checks must use the new exact code SHA and
fresh output directories; no previous result is overwritten or relabelled.

## Required next evidence

- Repeat Prefix/K-hook/r=1 and source/destination/source integrity on the new SHA.
- Verify the fixed15 704-to-160 row reduction remains valid.
- Collect fresh exact-shape future costs and online replays under full wall-time
  accounting. Report actual TTFT, decision, and cost scopes separately.
- Treat the new cache as an optimization candidate until paired timing evidence
  exists; do not claim a measured speedup from unit tests.

Formal Profile, qualification, H1--H5 and locked test remain disabled.

## Local acceptance and deployment status

Full local unittest discovery: 859 tests, 858 passed and 1 skipped. Contract
validator and `git diff --check` passed. This is CPU/local verification only.

The closure runner now has `--disable-current-kv-cache` for a same-SHA baseline.
The default and this control must use the same new correctness/cost root and
separate fresh output directories. The manifest and summary explicitly record
which mode ran. Interleave control/candidate sessions and retain cold samples;
do not compare different code revisions' single warm observations.

The deployment attempt in this turn did not reach SSH: port 46068 returned
connection-refused twice. Thus no new code or experiment was deployed by that
attempt. The local bundle is ready; server v44 evidence is unchanged.

## Restored connection: v45 results (053cf647)

The later connection succeeded. New correctness/cost execution is archived at
`/root/autodl-tmp/probekv_stage2/artifacts/native-053cf647-server46068-v45`.
Prefix/K-hook/r=1 and matched cost probe passed. Formal qualification remains false.
Native Prefix dense was 58.320671 ms; resumable dense was 79.856565 ms.
The fixed-winner streaming ready future was 43.223463 ms. These are diagnostic
observations, not repeated matched-quality performance estimates.

Online cache-on replay 2 has the following exact, contiguous host partition:

| Interval | ms |
| --- | ---: |
| Arrival to service | 0.018458 |
| Open context | 5.202277 |
| Selection plus preparation | 13.686794 |
| Dispatch to ready check | 0.000125 |
| Ready repair check | 1.118149 |
| Final admission | 5.621966 |
| Cancellation to finish call | 0.031235 |
| Finish entry | 0.001301 |
| Remaining prefill submission | 68.461919 |
| Native prefill bookkeeping | 1.152099 |
| Logits submission | 0.227802 |
| First-token host readiness | 0.117202 |
| First-token callback | 0.001121 |
| **Total** | **95.640448** |

`unaccounted_ns=0`. This request was rejected by FinalCommit, not executed with
selective reuse. The residual was compatible, but compatibility is not economic
admission. Warm cache-on replays: 94.262701, 95.640448, 98.963184 ms; cache-off:
93.385196, 92.526613, 99.620022 ms. These small sequential process-level controls
do not establish an E2E gain from projection caching (physical snapshot IDs also
differ). Retain cold samples separately; do not claim a speedup.

## Next execution-level fix: optional patch 0014

Source inspection found that each resumable layer creates active/target device
index tensors and two identical full-prompt arange tensors, even when masks are
unchanged. Patch 0014 shares a request-owned bounded index workspace, reusing
unchanged rows and one full-prompt range. It leaves QKV, attention kernels,
numerical policy, repair masks and admission thresholds unchanged. Request begin,
finish and exceptional context close clear the cache. Legacy patchsets remain
unchanged; 0014 requires independently audited ordered 0013+0014 patches.

Local acceptance: 863 tests, 862 passed and 1 skipped; compileall, contract
validator and diff check passed. GPU numerical/performance effect remains to be
checked on its own new SHA, patch tree and output directory.

## Why the CacheBlend headline is not our current measured baseline

The author publication reports 2.2--3.3x TTFT reduction versus full KV recompute,
with selective recomputation pipelined with KV retrieval for multi-chunk inputs:
https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/
Our current comparison is one 640-token non-prefix Segment with an already-hit
256-token Prefix. We additionally pay live historical-Source selection and a
custom resumable execution bridge. This does not excuse the overhead: the
native-versus-resumable dense gap must be reduced before expanding the benchmark.
The current evidence does not establish CacheBlend-matched performance or QA.
