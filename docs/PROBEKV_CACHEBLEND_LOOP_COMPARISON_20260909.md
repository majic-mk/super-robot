# Matched CacheBlend loop control and remaining setup audit

## Preregistration before execution

First Mistral 512-token Segment; 256-token textual Prefix, 32-token bridge,
32-token suffix, boundary/check block 2, resident immutable canonical Source.
Fixed15 and r=1; no Source selection, no natural-QA/performance qualification.
Two warmups plus three alternately ordered measurements per arm. No new patch.

The pinned upstream base is CacheBlend b72d7945e6d6306f12be66520196e0f081fa2b0c.
Source inspection establishes that original `LlamaModel.forward` sets org_seq_len
from only the input rows, while its Prefix attention branch calls contiguous
`PagedAttention.forward_prefix` even after sparse query filtering. Native Prefix
plus arbitrary sparse queries cannot be presented as an already supported,
matched unmodified-upstream baseline.

Two explicitly separate strata:

1. Native Prefix+dense versus ProbeKV resident packed, existing paired test.
2. Zero native Prefix in ALL arms: pure native dense, pinned CacheBlend normal
   forward loop with Segment adaptation, and ProbeKV resumable packed.

The second control is named `cacheblend_pinned_segment_adapter`, not vanilla
CacheBlend. It uses the pinned normal forward/layer loop, not ProbeKV resumable
hooks. Existing corrections retained: arbitrary absolute causal mask, mandatory
current Prefix/bridge/suffix dense ownership, ceil ratio count, flat-head RoPE
and the same numerical kernel/norm policy. Legacy CacheBlend raw V-difference
ranking and ProbeKV normalized V ranking are not silently treated as identical.
Per-arm masks and generated tokens must be saved and compared.

Request-private working KV allocation and canonical row installation are inside
TTFT. Canonical creation and GPU warm residency are outside all warm-hit arms.
Native block allocations, common sampling and decode endpoint are shared. No
fixed block tables or new Source are introduced. Corpus tokens remain unchanged.

Before any paired timings: run actual r=1 free generation and common-teacher
32-position logits for dense, CacheBlend-loop and ProbeKV. Require identical
free tokens and L2 <=1e-4. Save failed artifacts and stop the comparison if any
arm fails, never substitute dense for failing reuse. Independently hash resident
Source before/after the comparison, outside timed arms.

Setup audit: separate instrumented Prefix-matched ProbeKV run with real CPU/CUDA
profiler trace. Mark native input/sampling setup, per-request patch source check,
Prefix shadow H2D, composite creation, Prefix installation and Source row copy.
CUDA envelopes and profiler host/device intervals are diagnostic, not additive
TTFT. Inspect per-layer validation-induced synchronization without removing
checks or modifying the patch in this run.

No equality to published CacheBlend speedups is claimed. A future fully matched
Prefix-enabled CacheBlend arm needs an explicitly audited compatibility adapter.
Runtime/paper/locked-test gates remain false.

## Executed results and limitations

Execution code: `c44811bbe74a8f41e09d9a403f0ac922f897a55e`, Mistral on
A800-SXM4-80GB, server port 10411. Existing pinned patch unchanged. Output:
`native-mistral-c44811b-cbloop-profile-512`.

Warm-hit fixed-Source diagnostic only: 832 input tokens, C=[288,800), two
warmups excluded and three alternating measurements. No live multi-Source
selector, canonical creation, natural QA or production economic admission is
included. Private working-KV setup IS inside first-token wall-clock.

| Zero native Prefix in every arm | Mean first-token wall-clock (ms) |
| --- | ---: |
| Pure native dense | 74.182 |
| CacheBlend pinned normal-loop Segment adapter | 49.483 |
| ProbeKV packed resumable | 69.102 |

CacheBlend adapter reduces time by 33.29% versus this dense reference;
ProbeKV reduces it by 6.85%. ProbeKV is still 19.619 ms slower than this
CacheBlend control. This is a negative comparison result, not parity.

The separate Prefix-matched stratum in the SAME run measured native
Prefix+dense at 53.248 ms, ProbeKV resident legacy layout at 51.117 ms and
packed layout at 46.604 ms (12.48% reduction versus native Prefix+dense).
Do not compare the 46.604 ms Prefix-enabled number with the 49.483 ms
Prefix-disabled CacheBlend number as evidence that ProbeKV wins.

Both reuse arms pass r=1: greedy token identity and 32-position common-teacher
logit relative-L2=0. All timed fixed15 arms generated identical token IDs and
raw row digests verify. Canonical Source before/after digests match. These are
constructed correctness results, not natural-RAG quality qualification.
Raw versus normalized V ranking remains a confound: both repair 77 C tokens,
but the successful first retry had 20 positions in the masks' symmetric
difference. A future mask-replay control is required to isolate execution-only
overhead precisely. The prior native-Prefix versus no-cache-dense logit
discrepancy is not resolved by the zero-Prefix r=1 test.

## Profiler evidence: CPU submission and synchronization are priorities

Additional instrumented runs were executed AFTER ordinary paired timings.
CPU-to-GPU External-id correlation selects prefill operations only, includes
their asynchronous device tail, and excludes warmup/decode. Missing hardware
activity remains null. The phase analyzer is explicitly Mistral/32-layer scoped.

| Instrumented prefill diagnostic | CacheBlend loop | ProbeKV |
| --- | ---: | ---: |
| GPU kernel count | 734 | 915 |
| GPU kernel busy union, ms | 44.043 | 55.339 |
| GPU activity span, ms | 46.470 | 82.342 |
| GPU activity union, ms | 44.061 | 55.721 |
| cudaStreamSynchronize calls | 2 | 102 |
| cudaMemcpyAsync calls | 13 | 175 |

The activity span-minus-union is 2.409 versus 26.621 ms. This is a profiler
activity gap, not proven idle hardware or directly recoverable TTFT. Kernel
durations are also larger in the instrumented ProbeKV run; the gap cannot all
be assigned to setup. Profiler overhead, submission timing, mask differences
and GPU scheduling remain possible contributors.

Observed source-level causes to test separately: repeated per-layer active/
target-position tensor construction, small H2D copies, GPU boolean validation
that synchronizes the host, and resumable metadata reconstruction. The normal
loop reuses its position state. Setup annotations additionally identify roughly
1.50 ms host patch-capability inspection per request and 1.37 ms native input/
sampling preparation in the instrumented zero-Prefix run. These are nested/
overlapping diagnostic envelopes; never sum them into a claimed speedup.

Next bounded optimization experiments, each with new SHA/output and r=1 gate:

1. Mask-replay execution control, preserving original ranking as a separate arm.
2. Cache immutable patch-capability verification by loaded implementation and
   audit generation; invalidate on code/patch change.
3. Build validated request-local GPU position/index state once per mask
   generation, reuse across layers; retain bounds/subset/Prefix ownership
   checks and stale-state rejection.
4. Replace avoidable host synchronization with valid CUDA event dependencies;
   do not remove completion checks, leases or destination lifetime protection.
5. Compare a frozen-winner steady layer loop with resumable scheduling; retain
   the latter where selection or readiness actually requires pausing.

Do not change repair ratio, shorten the request, drop mandatory rows, change
timing endpoints or claim published CacheBlend speedups to close the gap.

## Failure preservation, tests and evidence

The first `7ff71dc` run failed in the CacheBlend adapter before its paired
timings: inherited `local_imp_indices=None` made a decoder index residual with
`None`, adding an invalid dimension. `88d0768` removes that resumable-only key
when installing normal-loop metadata. No numerical threshold was relaxed and
no patch was edited. The failed output and log remain intact. The retry and
final profiler run use new output directories.

Local regression: 748 tests, 747 passed and 1 skipped (ijson unavailable);
compileall and diff-check pass. Real GPU results do not imply general runtime
qualification. No formal Profile, paper experiment or locked test was run.
GPU was idle with 0 MiB allocated after the run; the cloud instance remains on.

Evidence archive (failure, retry, final run, raw tensors, traces and logs):
`artifacts/server10411-cbloop-c44811b-20260909.tar.gz`.
SHA256: `442a3fcac88434859f4b0630da86c67a8fd34105604815b60cb1db535922cc80`.
