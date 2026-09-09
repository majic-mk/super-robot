# Single-Segment working-KV layout diagnostic

## Frozen scope before GPU execution

Mistral only, same 256-token native Prefix hit, 32-token dense bridge,
512-token non-prefix Segment and 32-token suffix. Reuse boundary 2,
fixed15, prefetch window 2, GPU-resident fixed winner. No live Source
selection, real QA qualification, multi-Segment or formal profile claims.

Two independent opt-in changes are combined as `kv_layout_mode=packed_slice`:

- request-private `(layers, 2, rows, KV geometry)` zeroed storage, instead of
  64 separate zeroed K/V allocations; layer views retain the existing ABI;
- contiguous Source spans use a slice copy; noncontiguous spans retain exact
  indexed-write semantics. Canonical tensors never alias working storage.

The legacy layout remains selectable/default. No CacheBlend patch is changed.
Prefix and suffix geometry, numerical kernels and repair ratio are unchanged.

## Protocol

1. CPU regression suite and contract validator.
2. New exact SHA and fresh output directory, retain previous evidence.
3. Complete `r=1` free-token/logit/source-integrity checks on packed layout,
   including resident Source. No performance inference if checks fail.
4. Within one loaded model, alternate order of native Prefix+dense,
   resident legacy fixed15 and resident packed fixed15. Rebuild native Prefix
   using the same warm request for every arm. Two full warmup rounds followed
   by three measurement rounds. Record all rows; do not drop slow samples.
5. One additional instrumented run per reuse layout, reporting allocation+
   zeroing, Prefix install and per-layer Source install envelopes. Those
   instrumented TTFT values never enter the performance table. Envelopes may
   overlap/nest and must not be summed into request TTFT.
6. Hash resident Source before/after the complete paired block, outside timing.

Native Prefix teacher-vs-no-cache logit discrepancy from earlier runs remains
an open issue. r=1 matching is relative to the saved no-cache reference, not a
claim that all baseline kernel paths are numerically identical. Fixed15 free
token agreement is reported, not treated as natural-RAG quality certification.

The previous ~76 ms prepared-dense control includes unnecessary Source
preparation; it is not a pure dense baseline. Do not report a 33% full-dense
speedup from that value. CacheBlend parity requires a separately matched
CacheBlend arm and is not established by these measurements.

All outputs remain non-paper, unqualified diagnostic evidence. The old user
Git bundle is preserved. No automatic promotion to production layout or GPU
qualification is authorized by this sentinel alone.
