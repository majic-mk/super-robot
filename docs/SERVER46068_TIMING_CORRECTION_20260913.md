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
