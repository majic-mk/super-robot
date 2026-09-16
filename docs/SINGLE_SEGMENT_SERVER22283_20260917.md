# Server 22283 recovery, 2026-09-17

## Evidence identity

- Code: `e446220fedfcacbebdf52e9b59944343e537d80e`.
- GPU: `GPU-dadf3138-3ce7-bd4d-d4f0-4b12ae47df3b` (different from September 15).
- Independently rebuilt CacheBlend tree: `dcb41b56ef3ea2831e8fe8b0fcb66f56682bbff4`.
- CacheBlend directory: `/root/autodl-tmp/probekv_stage2/src/CacheBlend-e446220-server22283`.
- Artifact root: `/root/autodl-tmp/probekv_stage2/artifacts`.

The prior GPU's incomplete online run remains preserved. It is not resumed with
the new GPU's measurements. No historical output is overwritten.

## Completed baseline

Directory: `recovery-e446220-server22283-gpudadf3138-512-baseline-v1`.

Native Prefix/K-hook/r=1 correctness and matched-Prefix cost probe both passed.
CFO is not required and its status remains null. Formal GPU qualification is
false. The cost summary reports real CUDA execution, no per-request full-KV
digest and no frozen profile.

| Measured scope | ms |
| --- | ---: |
| Native Prefix dense first token | 53.266267 |
| Resumable dense first token | 71.014220 |
| Dense boundary to first token | 64.199183 |
| fixed15 boundary to first token | 45.768052 |
| fixed15 ready to first token | 42.671540 |
| Winner preparation | 3.096512 |
| Repair check | 0.812698 |

These scopes are not additive. In particular, boundary timing includes
preparation. The request admission limit is 42.6130136 ms, leaving no useful
headroom in this baseline once selection and planner costs are included.

## Completed online replay

Directory: `recovery-e446220-server22283-gpudadf3138-512-online-v1`.

All 22 outcomes and summary were generated; 0/22 committed a Source. With the
first two replay samples excluded, the remaining 20 have mean TTFT 87.3297525 ms
and range 86.141801–90.504129 ms. This is a sequential replay baseline, not paired
optimization evidence or a confidence interval. It does not establish positive
gain. FinalCommit correctly rejected the uneconomical path.

The inspected final outcome's host ledger accounts for all 90,504,129 ns, with
0 ns unaccounted. Its remaining-prefill interval alone is 65.591917 ms.

## Import-path recovery

The environment's historical `.pth` injection precedes PYTHONPATH. The online
entry point was therefore launched through `runpy` after explicitly inserting
the exact ProbeKV and rebuilt CacheBlend directories at `sys.path[:0]`. The
factory still validates the patch/source evidence. Global editable installation
was not changed. PYTHONPATH alone is not a reliable replay command here.

## Next controlled run

`recovery-e446220-server22283-gpudadf3138-512-deferred-v1` enables only
`--defer-layer-timing` relative to the baseline. Correctness and cost collection
must finish before its online replay. Owned-host validation and native dense
continuation remain subsequent, separately qualified controls.

## Deferred timing result

The deferred correctness and cost probe passed. Resumable dense first token was
64.719433 ms; fixed15 ready-to-token was 39.519546 ms; matched native dense was
53.226810 ms. These are individual cost observations, not paired effect sizes.

The separate `recovery-e446220-server22283-gpudadf3138-512-deferred-online-v1`
completed 22 outcomes, with 0 commits. The last 20 averaged 81.457100 ms
(78.814471–99.084283 ms); all 22 host ledgers balanced exactly. The decrease from
87.3297525 ms is descriptive across sequential sweeps, not a paired confidence
claim. The full path still does not outperform dense.

Next completed control: `recovery-e446220-server22283-gpudadf3138-512-hostpos-v1`,
keeping deferred timing and adding owned-host position validation with the
existing 20-pair position-validation A/B. Correctness and cost checks passed.
Native dense continuation is not enabled in this run.

After excluding the two prespecified warmup pairs, device validation averaged
65.5627211 ms versus owned-host validation 58.6122094 ms on the resumable dense
control (not online TTFT). Mean paired saving: 6.9505117 ms. A paired percentile
bootstrap (NumPy default_rng seed 20260726, 10,000 resamples of 20 pairs, linear
quantiles) gives 95% interval [6.26599233375, 7.82759047375] ms. All paired token
IDs match. Raw pairs remain in `native/position-validation-ab`.

The host-position online replay completed 22 outcomes, 0 commits. The last 20
averaged 73.92162275 ms (range 72.781032–76.012438 ms). No missing cost cells or
selection failures were reported in the inspected final outcome. Separate warm
sweeps are descriptive, not paired online optimization certification.

## Native dense continuation rejected

`recovery-e446220-server22283-gpudadf3138-512-continuation-v1` stopped during
correctness, before cost collection. Layers 2–32 were actually executed by the
continuation; free greedy IDs matched, but teacher logits did not meet 1e-4.

- Continuation versus uncached dense: aggregate relative L2 0.0104780877;
  maximum per-position relative L2 0.0449286923 over 32 positions.
- Native Prefix control versus uncached dense: aggregate 0.0075702299.
- Continuation versus native Prefix control: aggregate 0.0092460876,
  maximum per-position 0.0372725949.
- Source r=1 versus uncached dense still has aggregate relative L2 0.
- Prior baseline/deferred/hostpos resumable teacher controls each had relative
  L2 0 versus uncached dense.

Thus equal greedy tokens are insufficient. Switching the remaining layers to
native Prefix attention is not numerically interchangeable under this frozen
contract. Different attention paths are implicated, but a layer-level causal
diagnosis is still pending; these logits alone do not establish the precise
kernel/accumulation cause. Do not promote this flag or relax the threshold.

Next launched diagnostic: `recovery-e446220-server22283-gpudadf3138-512-matched-executor-v1`
with deferred timing and owned-host validation, native continuation disabled.
It runs the existing matched repair backend / CacheBlend loop controls (20
repeats), isolating executor cost from online selection. It failed argument
validation before model execution: the outer launcher omitted `--gpu-hot-cache`,
which the CacheBlend loop control requires. The launcher is being corrected
with a prerequisite regression test; any retry must use a new SHA/directory.

## Matched executor retry completed

Fix SHA `59a4f904c76d941c335aa12969f9d41486316a62` was committed, pushed and
deployed to its own checkout. Local acceptance: 916 tests, 915 passed and one
pre-existing skip; compileall and contract validator passed. No algorithm or
threshold changed.

Output: `recovery-59a4f90-server22283-gpudadf3138-512-matched-executor-v1`.
The summary verified raw digests, numerical/boundary equivalence, fixed15 mask,
Source integrity and resource cleanup. Each arm has 20 measured repeats after
two excluded warmups.

| Scope | Dense | Adapted CacheBlend | ProbeKV | Paired ProbeKV minus CacheBlend |
| --- | ---: | ---: | ---: | ---: |
| Setup-inclusive fixed-source first token (ms) | 74.12117355 | 48.9038818 | 64.24326755 | 15.33938575 |
| Boundary executor (ms) | N/A | 43.5223698 | 44.1892811 | 0.6669113 |

This fixture has **zero Prefix**, a resident Source and shared mask. Selection
and planner are not executed. It is an adapted CacheBlend control, not untouched
upstream. It does not certify fixed15 QA against dense, online FinalCommit,
or end-to-end production gain. Do not compare its 74.12 ms dense directly with
the ~53 ms native-Prefix baseline.

The residual gap is concentrated outside the narrowly matched boundary executor.
Separate profiler traces exist, but their instrumented durations cannot be
subtracted as an exact partition of these uninstrumented timing samples. Setup,
Source preparation and pre-boundary work need attribution before another
optimization. The rejected native-continuation flag remains disabled.

## Isolated contiguous-row copy control

SHA `275125cc3a3e1f6b62da1b24a6de06fc1f70103d` adds an opt-in
`--contiguous-source-rows`. It selects a slice only for exactly contiguous
ascending positions; noncontiguous rows keep the old indexing. Allocation stays
legacy, repair stays fixed15, and native continuation remains disabled. Local
acceptance: 917 tests, one existing skip, compileall and contract check passed.

Output: `recovery-275125c-server22283-gpudadf3138-512-slice-executor-v1`.
The same zero-Prefix resident fixture passed numerical, mask, integrity and
cleanup prerequisites. Twenty measured samples per arm, after two warmups:

- Setup-inclusive: dense 74.1885549 ms; adapted CacheBlend 48.88095905 ms;
  ProbeKV 52.75619345 ms. Paired ProbeKV/CacheBlend gap 3.8752344 ms.
- Boundary executor: CacheBlend 43.6152312 ms; ProbeKV 44.3809523 ms;
  paired gap 0.7657211 ms.

This reduces the descriptive setup-inclusive gap from 15.33938575 ms to
3.8752344 ms across runs. The two optimization configurations were not themselves
interleaved, so this is not a paired configuration-effect confidence interval.
Both runs contain their own interleaved CacheBlend control.

Separate profiler evidence corroborates the mechanism: prefill
`cudaStreamSynchronize` count drops from 73 to 11. The prior longest waits nest
under `probekv.source_rows_install` / `aten::to` / `aten::copy_`. Instrumented
inclusive wait durations are not additive to uninstrumented TTFT. This evidence
does not qualify the ordinary online path by itself.

Next pending: `recovery-275125c-server22283-gpudadf3138-512-slice-prefix-v1`,
revalidating Prefix, CPU backing, r=1 and matched cost support on the new SHA.

The Prefix/CPU run subsequently passed correctness and cost support. Native
dense measured 53.181305 ms; fixed15 boundary-to-token 28.501414 ms and
ready-to-token 25.380233 ms. These remain single-cell observations.

`recovery-275125c-server22283-gpudadf3138-512-slice-online-v1` completed all 22
replays, 0 commits; the last 20 averaged 73.17949235 ms. A warm sample (05)
shows sunk at snapshot 16.99208 ms, supported joint future 26.666203 ms and
planner elapsed 5.550599 ms, giving 49.208882 ms versus admission limit
42.545044 ms. The correct rejection explains why executor improvement has not
yet become an online speedup: the request still falls back to dense.

Next diagnostic is `recovery-275125c-server22283-gpudadf3138-512-host-profile-v1`,
three host-profiled replays. Its latency must not enter uninstrumented performance
summaries. It targets planner and initialization overhead; no threshold changes.

Host profiling completed all three replays. Warm replay 02 attributes about
7.18 ms inclusive to `plan_ready_subset`, including two cost `lookup` calls
totaling about 7.05 ms. Across the profiled request, 27 `digest_json` calls take
8.42 ms inclusive and deepcopy takes 5.08 ms inclusive. These overlapping,
instrumented totals include work outside TTFT and must not be summed or used
as latency savings. They identify repeated query/mask construction, serialization
and copying as the next investigation target; no planner check has been removed.

## Exact-query construction optimization

SHA `1a59009bf70ca4a277c32e8ce137dd034148b0a3` reuses equal layer masks and their
serialization within a query. Canonical SHA bytes remain unchanged; no cross-
decision identity-based cache was added. Mutation and canonical-encoding tests
passed. Local suite: 920 tests, one existing skip, contract/compile checks pass.

`recovery-1a59009-server22283-gpudadf3138-512-query-prefix-v1` passed GPU
correctness and cost support. Native dense 53.144363 ms; fixed15 ready-to-token
26.237103 ms. New SHA uses its own measurements.

`recovery-1a59009-server22283-gpudadf3138-512-query-online-v1` completed 22
replays, zero commits. Excluding two warmups, TTFT mean was 69.2475648 ms and
mean planner elapsed 1.4892739 ms across all 20 decisions (both serialized event
forms included). Thirteen initially acceptable estimates failed after actual
planner time was added; seven were pruned on initial cost. Candidate totals
after planner ranged 42.545345–45.954069 ms versus limit 42.5154904 ms. All warm
host ledgers balanced exactly. No cost check or planner time was bypassed.

Sequential configuration sweeps remain descriptive, not paired proof of final
online speedup. No production reuse commit or positive complete-path gain yet.

No multi-Segment, Qwen, multi-Source gain study, frozen Profile, qualification,
H1–H5 or locked test has been authorized by these results. GPU hourly price and
total monetary cost remain unknown, not zero.
