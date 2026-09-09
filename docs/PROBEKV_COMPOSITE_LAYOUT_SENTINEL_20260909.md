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

## Results (code 8cd74a6136ad904d2adcc15a0c8102b675a55a89)

After the 512-token diagnostic passed, a separate 640-token validation was
announced before execution. Prefix, bridge, suffix, boundary, window, ratio,
paired warmup/repeat counts and numerical policy remained unchanged. Both
length strata are retained; longer requests are not substituted for short ones.

| Segment tokens | Native Prefix+dense ms | Resident legacy ms | Resident packed+slice ms | Packed reduction vs native |
|---|---:|---:|---:|---:|
| 512 | 53.374 ± 0.069 | 50.947 ± 0.144 | 46.472 ± 0.150 | 12.93% |
| 640 | 58.113 ± 0.162 | 52.149 ± 0.321 | 47.138 ± 0.238 | 18.89% |

Mean ± sample standard deviation of three non-instrumented first-token host
measurements; two preceding warmup rounds excluded. Old and new layouts run
inside the same model process with alternating arm order. Versus resident
legacy, packed+slice saves 4.474 ms (512) and 5.011 ms (640). Fixed winner,
Source materialization and hot-cache construction are outside the warm-hit
endpoint; these values are not complete online multi-Source selection TTFT.

All raw paired observation SHA256 values recomputed successfully. All paired
arms have the same token identity, cached Prefix count and sampling signature;
their full generated token lists match. Both CPU-loaded and GPU-resident r=1
arms pass at each length: greedy tokens identical and 32-position logit L2=0.
Resident Source digests before/after the entire paired block are identical.
Constructed text token agreement is not real QA certification.

The extra 512-token instrumented controls report:

| Component | Legacy CUDA envelope sum ms | Packed CUDA envelope sum ms |
|---|---:|---:|
| Allocation + zeroing | 0.240 | 0.087 |
| Prefix shadow install | 0.475 | 0.228 |
| Per-layer Source install | 4.551 | 0.507 |

These are diagnostic CUDA-event envelopes, including possible scheduling gaps,
not kernel busy time. Their host enqueue durations can overlap GPU work; do
not add either set to TTFT. The intervention combines packing and contiguous
copy, so a factorial ablation would be required to assign exact independent
speedups. The measurements support advanced indexing as a substantial overhead,
not a claim that all remaining latency is now explained.

Current limitations / next boundaries:

- Neither mean reaches the formal native-Prefix-relative gamma=0.8 threshold
  (ratios approximately 0.871 and 0.811), even before live selector costs.
- No matched unmodified CacheBlend arm has been run; parity is unproven.
- No Source-selection cost, real QA, cold-start amortization, full Profile,
  multi-Segment or production GPU LRU qualification is claimed.
- Original native Prefix teacher numerical discrepancy remains unresolved.
- Layout optimization remains opt-in. Source and Prefix are never aliased to
  mutable composite tensors, and suffix/dense holes remain initialized.

Server outputs under `/root/autodl-tmp/probekv_stage2/artifacts/`:

- `native-mistral-8cd74a6-layout-ab-512`
- `native-mistral-8cd74a6-layout-ab-640`

Local validation: 741 tests, 740 pass / 1 ijson-related skip; compileall and
contract validator pass. Original local user bundle is preserved.
