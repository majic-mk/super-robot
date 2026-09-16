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

No multi-Segment, Qwen, multi-Source gain study, frozen Profile, qualification,
H1–H5 or locked test has been authorized by these results. GPU hourly price and
total monetary cost remain unknown, not zero.
