# CacheBlend integration boundary

`CacheBlendBackend` owns invariant checking and exposes the stable ProbeKV
backend API. A small runtime shim inside the pinned CacheBlend fork must provide:

1. `stage_canonical_source`: stage exact full-prefill KV and return measured
   latency.
2. `selective_repair`: repair at a layer and ratio, returning quality, token F1,
   CUDA-event latency and canonical-source digests before and after.
3. `dense_remaining_ms`: measure the matching full-recompute remainder.
4. `provenance`: report CacheBlend, vLLM, PyTorch and CUDA versions.

The adapter rejects a mutated canonical source, invalid metrics, negative
timings and incomplete provenance. Local tests use a fake runtime, so the only
server-specific code is the shim that maps CacheBlend tensor handles and CUDA
events to this protocol.

## Server milestones

- CB0: pinned unmodified CacheBlend commit runs its own example.
- CB1: shim stages and hashes one full-prefill source without repair.
- CB2: ratios 0 and 1 match expected endpoints on one case.
- CB3: all repair-grid ratios emit quality and separated timing fields.
- CB4: current-state features and four-source selection run end to end.

Do not begin E1 until CB0-CB3 and the H0 requirements in `A100_RUNBOOK.md` pass.

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
