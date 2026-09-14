# Single-Segment cost and online revalidation

Date: 2026-09-14. Execution SHA: `27c2577f62c7622ce36aa4ff73fb60902d8764da`.
Model: Mistral-7B-Instruct-v0.3; A800 UUID
`GPU-936c826b-dd49-6d32-53f3-6640860c5e47`. Segment: 512 tokens.
One diagnostic request/replay; not a statistical performance comparison.

## Execution and correction

The initial remote SHA was `028b47bfb33bb332884abc41fa9b72fd0305df2d`.
Its fresh cost probe passed, but online outcome assembly failed because the
string `prefix_shadow_transfer_mode` was in the nanosecond-only timing map.
The fix separates transfer metadata and preserves strict timestamp validation.
It does not change kernels, masks, admission thresholds or cost support rules.
Local regression: 885 tests, 884 passed and one existing skip. Contract and
compile checks passed. A separate exact-SHA checkout was deployed; no old
result was overwritten. Correctness and costs were collected again at the fix SHA.

## Evidence locations

Server base: `/root/autodl-tmp/probekv_stage2/artifacts/`.

- `native-028b47b-costprobe-512-v1`: initial cost evidence.
- `online-028b47b-costprobe512-v1.log`: preserved accounting exception.
- `native-27c2577-costprobe-512-v1`: corrected-SHA correctness and cost evidence.
- `online-27c2577-costprobe512-v1`: corrected-SHA online outcome and query audit.

Selected raw JSON/event evidence is downloaded under local
`artifacts/single-segment-27c2577-evidence/`. No formal Profile is frozen.

## Results

- Prefix/K-hook/r=1 prerequisite: passed; reported logit relative-L2 = 0.0.
- Matched-Prefix cost prerequisite: passed.
- Cost builder validated signed raw observations, manifest, matched native
  Prefix baseline, timing landmarks and measured future-layer mask coverage.
- Online reuse and dense joint queries: both SUPPORTED, with MeasurementKey,
  measurement row/table digests and PlannerSnapshot binding.
- Online transfer mode: layerwise; cost probe uses online_immutable with
  `request_full_kv_digest_performed=false`.
- Source prepared: yes; production reuse commit: **no**.
- FinalCommit rejected for economic reasons, not missing cost support.

Matched native Prefix dense TTFT = **53.683378 ms**. Gamma limit = **42.9467024 ms**.
At the planner snapshot, actual sunk = **41.486107 ms** and candidate reuse
joint future = **43.621855 ms**, hence candidate total = **85.107962 ms**.
Planner time adds further cost. The post-prune total is a dense alternative,
not the cost of reuse. Actual online fallback TTFT = **112.975123 ms**.

The fixed15 cost arm's boundary-to-first-token = 47.253975 ms and its
ready-to-first-token = 43.998462 ms. Neither is end-to-end TTFT. The ready
future alone exceeds the 0.8 dense target in this sample; selection/setup-only
optimization cannot establish admission with this measured future unchanged.

## Exact, non-overlapping online wall-clock ledger

| Interval | ms |
|---|---:|
| Queue | 0.018980 |
| Context initialization | 4.080947 |
| Context open to selection closed (includes probe/comparison/preparation) | 32.335131 |
| Gap to readiness check | 0.000147 |
| Readiness check | 4.762739 |
| Final admission region | 6.437493 |
| Cancellation/finish dispatch | 0.006239 |
| Finish entry | 0.001713 |
| Remaining prefill submission | 64.936162 |
| Native bookkeeping | 0.063061 |
| Logits submission | 0.193996 |
| First token host readiness | 0.137070 |
| First token callback | 0.001445 |
| **Total** | **112.975123** |

Raw integer-nanosecond accounting reports zero unaccounted time. These are
host intervals, not additive CUDA service times. The 32.335131 ms region is
not exclusively Source comparison, nor is submission pure GPU kernel time.

## Stop and next diagnostic boundary

The requested reuse-commit prerequisite did not pass. Do not advance to
multi-Source net-gain experiments or label fallback as successful reuse.
No new matched CacheBlend/ProbeKV end-to-end comparison is claimed; earlier
fixed-resident boundary-only comparisons are not interchangeable with this
CPU-backed Prefix request. Next diagnosis should separate the cold comparison
cost, preparation/readiness and resumable dense/repair execution on matched
inputs before another gain claim. Keep gamma=0.8 and all failure evidence.

`formal_profile_bundle_frozen=false`, `gpu_runtime_qualified=false`,
`h1_h2_execution_allowed=false`, `paper_evidence=false`, `locked_test_accessed=false`.
