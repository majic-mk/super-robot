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
