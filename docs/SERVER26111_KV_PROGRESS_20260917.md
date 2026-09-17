# Server 26111: joint K/V baseline and next evidence gates

## Executed baseline

Code: `1030b784827e05eb965ca3d2c1f89a17a899d79f`.
GPU: `GPU-e88a60ed-6406-5c11-53c3-c01041eb88fd` (A800-SXM4-80GB).
Root: `/root/autodl-tmp/probekv_stage2/artifacts/`.

- `p0-1030b78-server26111-512-kv-v1`: independent environment audit,
  Prefix/K-hook/r=1 sentinel and matched cost probe passed.
- `p0-1030b78-server26111-512-kv-online-v1`: 22 completed online replays.
- `p0-1030b78-server26111-512-kv-profile-v1`: 3 cProfile diagnostic runs;
  not performance samples. Profile covers decode as well as prefill, so its
  aggregate function times are not a TTFT decomposition.
- `p0-1030b78-server26111-128-kv-v1` and
  `p0-1030b78-server26111-640-kv-v1`: both completed; Prefix/K-hook/r=1
  and matched cost probe passed. Online replays at these lengths remain pending.

New runs explicitly use `normalized_kv_deviation`. Old V-only measurements
remain historical evidence, not the new baseline. CFO remains not applicable.

## Measured results (not QA or formal qualification)

Matched cost probe: native Prefix first-token 53.127550 ms;
fixed15 boundary-to-first-token 28.728740 ms; ready-to-first-token
25.242550 ms. These are different timing scopes and are not interchangeable.

Online: exclude the preregistered first 2 warm-up replays, leaving 20.
Only 7/20 committed. All-request mean TTFT is **67.0531742 ms**.
Committed subgroup mean is approximately 41.95 ms; fallback subgroup mean
approximately 80.57 ms. Do not report the committed subgroup as system gain.
All 22 records identify the joint K/V metric and have zero unaccounted ns.
Local raw copies: `artifacts/server26111-kv-1030b78/outcome-*.json`.

Warm mean contiguous host ledger, in ms:

| Interval ending at | Mean |
|---|---:|
| service_start | 0.01390945 |
| context_opened | 10.01928655 |
| selection_closed | 10.71816455 |
| ready_check_begin | 0.00013875 |
| ready_check_end | 1.24430770 |
| final_admission_end | 2.87718740 |
| finish_call | 0.00502250 |
| finish_enter | 0.00155675 |
| remaining_prefill_submitted | 24.98189090 |
| native_prefill_bookkeeping_done | 16.85994795 |
| logits_submitted | 0.18247915 |
| first_token_host_ready | 0.14833565 |
| first_token | 0.00094690 |
| Total | 67.05317420 |

The bookkeeping interval includes `NativeBlockRequest.finish_prefill()`'s
CUDA completion fence. It is not proven CPU bookkeeping waste: asynchronous
GPU work can finish there. Removing its timing or fence does not remove work.
Replay 20 has 103.423807 ms context initialization; root cause remains pending.
Replay 2's candidate total at snapshot is 43.025600 ms versus a 42.502040 ms
admission limit, so fallback is expected under the frozen gamma, not evidence
of an unsupported cost key. Do not relax gamma to force successful commits.

## Ordered remaining work

1. 128/640 joint-metric correctness/cost boundary checks are complete;
   their online performance and QA are not certified by these sentinels.
2. Diagnose initialization variance and supported joint-future prediction error;
   preserve all samples. No new setup optimization without a measured cause.
3. Build audited multi-target cohorts: four prior Sources and at least five
   future targets per shared exact Segment, with original QA context intact.
   Existing MuSiQue recovery proposals contain two groups with at least five
   additional candidates, but have not passed token/partition authorization.
4. Fixed15 joint-K/V quality/cost oracle on every candidate. Deep residual
   minimum is a proxy, never the QA oracle. Then compare d1/d1+d2/legacy.
5. Independent causal K=1/2/4 traces under equal byte budgets; assess complete
   costs and quality before a multi-Source Go/No-Go decision.

The new `source_quality_requests` helper preserves every document and answer,
validates the canonical token reconstruction, requires an explicit fit/validation
role, and rejects overlength requests instead of truncating. It does not by
itself verify partition membership, generate an executable cohort, or authorize
new data. Wiring that audit and cohort runner remains pending.

No Qwen, multi-Segment, SparseX/QCFuse, formal Profile, qualification or locked
test was started. Matched-quality positive gain remains **unproven**.
