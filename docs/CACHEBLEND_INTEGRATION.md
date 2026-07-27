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
