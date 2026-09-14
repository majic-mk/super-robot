# Same-Source / same-mask resident executor comparison

## Frozen scope

- Code: `14d4087b06fc8899be0be8e9db2c0c5848dd4df6`.
- Server: port 46068; Mistral; one 512-token Segment, 832 prompt tokens.
- One canonical GPU-resident BF16 Source, fixed 15% repair (77 Segment rows).
- Both arms have zero native Prefix hits. This is **not** the 640-token Prefix-hit online experiment.
- Same absolute mask, boundary=2, greedy sampling and 32 output tokens.
- Patched CacheBlend normal loop vs ProbeKV resumable executor; not a claim of unmodified upstream CacheBlend.
- Two warmups excluded; 20 numbered interleaved pairs per scope. Profiler arms are separate and excluded.
- No Source ranking, economic Planner or production admission. No dynamic repair-ratio changes.
- Canonical capture, GPU residency establishment and full digests are outside the measured warm-hit samples and separately retained.

## Two timing scopes

1. **Setup-inclusive fixed-Source**: dense / pinned CacheBlend normal loop / ProbeKV. Includes each executor's request setup. Necessary lease/reservation safeguards remain active.
2. **Boundary executor only**: both arms use the same common dense execution through completed depth 1 and resident preparation before timing. The interval is layer 2 through first token, not request TTFT. Lease acquisition, HBM reservation, H2D, Source selection and Planner are outside this interval. Their setup duration is separately recorded.

The second CacheBlend arm uses the pinned decoder continuation, explicitly not the original full-forward entry. CacheBlend still performs its native boundary Top-K; ProbeKV receives the frozen mask directly. Both verify actual mask equality. ProbeKV's normal per-layer Source-row installation is not monkeypatched away.

Required checks before accepting timing: r=1 exact token equivalence; fixed15 same-mask backend equivalence; teacher logits relative-L2 <=1e-4; complete ordered layer execution; canonical Source digest unchanged; raw-file hashes; exact repeat count; resource cleanup.

## Evidence handling

- Attempt `matched-4881010-resident512-r20` is preserved as failed: all numerical comparisons completed but aggregation correctly rejected the missing explicit ProbeKV `boundary` field. No manual change of its result files.
- `14d4087` adds the missing boundary audit field and cleanup on aggregation errors; the complete experiment is rerun in `matched-14d4087-resident512-r20`.
- Historical same-mask timing and logits evidence already existed on the server. Earlier statements that no comparable CacheBlend measurements existed were an incomplete search, not absence of all historical evidence. Fresh measurements bind the current patch and implementation.
- These are synthetic backend diagnostics, not QA certification, multi-Source selection evidence, CPU/SSD overlap evidence or paper performance results.

Local validation: 874 tests, 873 passed and 1 skipped; compileall and contract validator passed.

## Completed GPU result (20 measured repeats per arm)

| Setup-inclusive fixed-Source path | Mean ms | Median ms | Reduction vs dense |
|---|---:|---:|---:|
| Native dense, zero Prefix hit | 74.167896 | 74.113707 | — |
| Matched CacheBlend normal loop | 49.464863 | 49.260727 | 33.31% |
| ProbeKV, selector/Planner bypassed | 54.143277 | 54.051436 | 27.00% |

Full-scope paired mean gap: ProbeKV minus CacheBlend = **4.678414 ms**. This is a fixed-Source diagnostic, not the live online selector or the 640-token Prefix-hit fallback path.

| Common boundary 2 → first token, setup excluded | Mean ms | Median ms |
|---|---:|---:|
| Pinned CacheBlend decoder continuation | 43.709245 | 43.688171 |
| ProbeKV executor | 44.964574 | 44.903019 |

Boundary-only paired mean gap: **1.255329 ms (2.87% of CacheBlend boundary time)**. Safeguards are retained outside this interval. The common dense first block, resident setup and mask binding are separately measured; integer-nanosecond setup + executor intervals exactly equal their reported enclosing interval in all 40 boundary samples. These are not request TTFT values.

Both full and boundary-only r=1/fixed15 pairs have identical greedy tokens and teacher-logit relative-L2=0. The frozen mask contains 77 out of 512 Segment rows; the other 320 prompt rows remain mandatory dense. Source digest unchanged; post-run HBM/lease cleanup passed; GPU memory was released. Local reaggregation verified 115 prerequisite/sample file digests.

Instrumented profiles (excluded from the averages) show 738 vs 797 prefill kernels and more runtime copy/synchronization calls in ProbeKV. These guide the next investigation, but their durations must not be subtracted from uninstrumented TTFT to claim exact attribution. Nor should the two independently measured scope gaps be treated as an exact additive setup decomposition.

Interpretation: the same backend library and same repair count do **not** imply the same execution schedule. On current code the remaining executor-tail gap is small; a further setup-inclusive gap remains. The hypothesis that all of the difference comes from Source comparison is rejected, but the older 18–24 ms fixed-Source gap is not reproduced by this current controlled run. Next optimization should target fixed-Source setup/handoff and redundant per-layer work before changing repair ratios.

Local evidence: `artifacts/matched_resident_executor_20260914/raw-json-evidence.tar.gz` (145 JSON files), `result.json`, `patch-audit.json` and `runtime.log`. Full teacher tensors and profiler traces remain in the server's immutable output directory; JSONs bind their hashes.

Patch tree: `dcb41b56ef3ea2831e8fe8b0fcb66f56682bbff4`; patch SHA256: `236f4f85b1751c4c7a84a77f170c23046be738296483921c35956798912ebe61`.
Runtime package: `/root/autodl-tmp/probekv_stage2/src/CacheBlend-matched-4881010-0016/vllm_blend/vllm` (explicit process import, independently audited patches 0001–0016).

No online admission/profile/QA qualification is unlocked by this result. No dynamic repair, multi-Source selection, Qwen or multi-Segment experiments were run.
