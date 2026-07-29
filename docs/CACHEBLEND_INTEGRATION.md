# CacheBlend integration boundary

`CacheBlendBackend` owns invariant checking and exposes the stable ProbeKV
backend API. The A800 implementation is split between the tracked CacheBlend
patchset and the case-scoped worker in `cacheblend_server_runtime.py`:

1. `stage_canonical_source`: stage exact full-prefill KV and return measured
   latency.
2. `selective_repair`: repair only tokens in repeated segment C, while treating
   S as mandatory dense work, and return quality, token F1, token-index audit,
   CUDA/host timing and canonical-source digests.
3. `dense_remaining_ms`: measure the matching full-recompute remainder.
4. `provenance`: report CacheBlend, vLLM, PyTorch and CUDA versions.

The adapter and server result schema reject a mutated canonical source, invalid
ratio counts, negative timings, non-1-based layers and incomplete provenance.
Intermediate repair ratios use causal rows from each selected token's absolute
position. The former bottom-right triangular mask was invalid for arbitrary
non-contiguous C queries and is not an accepted CB3 runtime.
The H1 pilot uses repaired-token-layer cost as its primary source-sensitivity
metric. Its `full_remaining_*` fields are explicitly marked
`full_prefill_total_proxy`; exact remaining-layer timing is still required
before H4 admission or any performance claim.

## Server milestones

- CB0-original: pinned unmodified CacheBlend failure remains archived.
- CB0-patched: the one-line suffix compatibility patch runs all ten examples.
- CB1: shim stages and hashes one full-prefill source without repair.
- CB2: ratios 0 and 1 match expected endpoints on one case.
- CB3: all repair-grid ratios emit quality, absolute-position causal-mask
  provenance and timing medians from two warmups plus five measurements.
- CB4: current-state features and four-source selection run end to end.

Do not begin H1 jobs until CB0-patched through CB3 and the H0 requirements in
`A100_RUNBOOK.md` pass.

## Pinned CB0 issue observed on A800

On 2026-07-28, commit `b72d7945e6d6306f12be66520196e0f081fa2b0c`
loaded Mistral and entered the xFormers backend, but its unmodified
`example/blend.py` failed with `KeyError: suffix_len`. The same commit's
`xformers.py` requires `cache_fuse_metadata["suffix_len"]` when `status == 1`,
while `blend.py` never initializes that field.

A diagnostic copy that added
`cache_fuse_metadata["suffix_len"] = len(q_ids + s_end)` immediately before
cached generation completed all 10 cached and full paths without an OOM, CUDA,
or KV-shape error. This diagnostic is not accepted as an unmodified CB0 pass.
Keep the original failure log and the one-line diagnostic log separate. Before
CB1, either obtain an upstream-compatible invocation or pre-register a minimal
fork patch and rerun every directly compared method on that same fork.

The selected policy is now frozen in `patches/cacheblend/manifest.json`.
`0001-cb0-fix-suffix-length.patch` is the compatibility correction;
`0002-probekv-segment-repair-mask.patch` adds P/C/S boundaries and logit
instrumentation without changing CacheBlend's V-drift ranking algorithm.
All same-stack methods must use the same patched runtime.

`repair_gpu_ms` and `repair_host_ms` currently cover the complete generation
call; TTFT covers prefill through the first token. They are explicitly tagged
with `repair_timing_scope` and are pilot diagnostics, not repair-kernel-only
timings. Exact remaining-layer timers are still required for v4 final
admission and formal H4 performance evidence.
