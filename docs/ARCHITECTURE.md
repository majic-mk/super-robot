# ProbeKV system architecture

## Data path

1. The tokenizer identifies an exact repeated non-prefix segment `C`.
2. Prefix Cache gets first refusal. ProbeKV runs only when prefix reuse does
   not solve the request.
3. For every historical context (`A`, `B`, `E`, ...), the system independently
   runs `full_prefill(context | C)` and registers a read-only canonical source.
   `KV(C | B)` must never be derived from `KV(C | A)`.
4. The current request computes fresh early layers. At calibrated checkpoints,
   current K/V/hidden/query features are compared with compact source summaries.
5. A source is selected only when its conservative cost upper bound is below
   every competitor's lower bound. At `L_probe_max`, uncertainty causes abstain.
6. The selected source is prefetched. The reuse planner checks layer-by-layer
   buffer readiness and the full cost:

   `probe + compare + visible_load + repair <= gamma * full_recompute`.

7. While the source is loading, the scheduler may compute more dense layers of
   request A and/or run short ready microbatches from requests B/C.
8. A source-ready event promotes A. If the economic boundary has passed, reuse
   is cancelled and A finishes with full recomputation.
9. Repaired output is consumed by the request but is never registered as a new
   source.

## Components

| Component | Implementation | Contract |
|---|---|---|
| Canonical store | `source_store.py` | exact full-prefill only, Kmax=4 |
| Probe selector | `selector.py` | conservative interval early exit |
| Budget calibration | `calibration.py` | isotonic baseline + split conformal upper |
| Case manifest | `manifest.py` | token hash plus content/document split isolation |
| RAG normalization | `rag_data.py` | three schemas; controlled and corpus-repeat kept separate |
| HF reference state | `reference_hf.py` | full-prefill pre-RoPE K/V/hidden/query correctness |
| Local E1/E2 loop | `local_e1e2.py` | labels, fit/calibration, locked evaluation, resume |
| Repair label | `labeling.py` | suffix-monotone safe ratio |
| Reuse planner | `cost.py` | total-cost admission and dynamic layer |
| Prefetch | `prefetch.py` | P0-P4 and HBM-aware Dynamic |
| Scheduler | `scheduler.py` | No-overlap/A-only/B-only/Hybrid |
| Repair integration | `backend.py`, `cacheblend_backend.py` | stable runtime shim; canonical input remains immutable |
| Statistics/gates | `statistics.py`, `gates.py` | paired grouped inference |
| Audit trail | `io.py` | JSONL, optional Parquet, environment manifest |

## Meaning of the bandwidth inequality

`BW_available >= KV_bytes_per_layer / compute_time_per_layer` means that the
copy engine can deliver at least one layer of KV during the time the GPU spends
computing one layer. It is necessary for steady-state overlap, not sufficient
for perfect overlap: the first needed layer must already be present, HBM and
copy traffic may interfere, and scheduling gaps can expose transfer latency.
