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

## Runtime modes after protocol v5

The same repair patch is used by two deliberately different runtime modes:

- `cacheblend_case_runner` is the existing CB1-CB3/H1 worker. It builds all
  case Sources and executes complete `generate()` calls for fixed
  `(Source, layer, ratio)` jobs. It is valid for Source-sensitivity labels but
  is not an online scheduler.
- `cacheblend_closed_loop` uses `CacheBlendClosedLoopRuntime` and the
  `CacheBlendOnlineEngine` contract. It requires asynchronous winner-only
  Source loading, layer-resumable prefill, real scheduler feedback,
  boundary-conditioned profiles, immutable canonical Sources and CUDA-event
  timing.

The case runner advertises these missing capabilities as `false`; the
closed-loop adapter rejects it before any Source transfer. A complete
`generate()` call may therefore never silently stand in for layer-resumable
execution.

The v5 request order is fixed:

1. Probe checkpoints may select a Source before `L_probe_max`.
2. The selected Source ID is locked.
3. The preliminary component-cost upper bound is checked before loading.
4. Only the winner begins asynchronous transfer.
5. The real waiting-window scheduler returns Source-load start, Source-ready,
   scheduled atomic-step finish, A resume, post-ready blocking, transferred
   bytes and the evaluated 1-based reuse boundary.
6. The boundary profiler supplies conservative future repair-selection,
   repair and remaining-layer costs for that exact boundary.
7. Final admission either calls CacheBlend selective reuse with the locked
   Source or executes full recomputation. A rejected request records all
   transferred Source bytes as wasted.

Refined cost is not described as fully measured before execution. Its past
components are real runtime events; its not-yet-executed repair and remaining
components are conservative A800 boundary-profile values. Records use
`refined_actual_past_profiled_future` and retain the exact profile key.
Realized TTFT is recorded after reuse or full execution and must later be used
to audit profile coverage.

Configuration `a800_closed_loop_v5.json` refuses to run unless async loading,
resumable prefill and CUDA-event timing are all required. The existing pinned
CacheBlend model loop does not yet expose resumable prefill; implementing and
validating that stack-specific engine hook on A800 is the remaining CB4
hardware gate. Until CB4 passes, closed-loop records remain
`server_pilot`/`paper_evidence:false`.
