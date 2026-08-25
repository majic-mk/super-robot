# ProbeKV system architecture

## Data path

1. The tokenizer identifies every exact repeated non-prefix segment
   `C_1 ... C_n`; v3-v5 retain their single-segment representation.
2. Prefix Cache gets first refusal. ProbeKV runs only when prefix reuse does
   not solve the request.
3. For every historical context (`A`, `B`, `E`, ...), the system independently
   runs `full_prefill(context | C)` and registers a read-only canonical source.
   The current prefix `P` and all historical prefixes are mutually distinct;
   `KV(C | B)` must never be derived from `KV(C | A)`.
4. The current request computes fresh early layers. At calibrated checkpoints,
   current K/V/hidden/query features are compared with compact source summaries.
5. Before `L_probe_max`, a source is selected only when its conservative total
   cost upper bound is below every quality-safe competitor's lower bound. At
   `L_probe_max`, the configured selector policy either preserves strict
   abstention or applies an explicit quality-safe, predicted-economic fallback.
6. Source selection and reuse admission are separate states. A selected source
   is prefetched, then the scheduler returns actual source-ready, A-resume and
   boundary events. Only after those events exist does the reuse planner check
   the refined full cost:

   `probe + metadata + compare + visible_load + post_ready_blocking + interference_charge + repair_selection + repair + remaining_layers <= gamma * full_total`.

7. While the source is loading, the scheduler may compute more dense layers of
   request A and/or run complete, non-preemptible dense steps or microbatches
   from requests B/C. Strict scheduling never starts work that crosses
   source-ready. The bounded-overrun policy may start one complete step whose
   crossing is within its explicit budget.
8. A source-ready event promotes A after any already-running atomic step.
   Post-ready blocking is charged to A, while B/C service remains useful system
   work. If refined admission fails, the selected Source ID remains audited but
   execution switches to full recomputation.
9. Repaired output is consumed by the request but is never registered as a new
   source.

The enforced v4/v5 call order is:

`selector -> load_and_schedule -> measured/refined cost -> final admission -> reuse/full`.

If the selector abstains, the controller calls full recomputation before any
Source load. It is illegal to substitute a latest or default Source. If final
admission rejects, the selected Source and evaluated boundary remain in the
audit, but `actual_reuse_boundary` is null because reuse did not execute.
Protocol v5 additionally requires preliminary and refined values to use the
same origin, endpoint and interference accounting mode. A Source selected at
any early checkpoint is locked before loading; refinement cannot reselect.

Protocol v6 lifts the legacy per-request Kmax and single-segment restrictions
without altering old artifacts. It keeps up to 16 variants per retained
content in a global byte pool, compares summaries within one 5% request budget,
and applies the same call order independently to all detected candidate
segments. The multi-source scheduler feeds actual layer readiness into a
request planner that uses the explicit A or C per-Segment boundary policy and
may accept only an economical subset (`PARTIAL_REUSE`). The old common path is
an explicit reproduction configuration. See `V6_MULTI_SEGMENT.md`.

## Probe depth objective

Increasing raw drift at deeper layers does not prove that Source ranking becomes
better. All Sources may simply move farther from the current state, while noise
and probe cost also increase. ProbeKV therefore minimizes decision time rather
than maximizing feature separation:

`choose the earliest layer whose calibrated intervals separate and whose total reuse path is economical`.

Every layer through the 25% ceiling is checked in the main policy. A full-layer
scan is allowed only on a small pilot subset to diagnose where the signal peaks;
it is not an online policy. If reliable ranking appears only near or beyond the
25% ceiling, or its net TTFT gain is below the gate, the Probe hypothesis fails
and the request falls back to full recomputation.

## Selector and admission policies

`strict_interval` is the legacy reproduction policy: overlapping calibrated
intervals at the maximum selection layer abstain. Two explicit final-layer
policies are available for new experiments:

- `final_economic_min_cost` selects the lowest predicted total-cost upper bound
  after quality and predicted `gamma` filtering. This matches ProbeKV's core
  safe-total-cost objective.
- `final_economic_max_reuse` first keeps Sources within
  `reuse_ratio_tolerance` of the best conservative repair-ratio upper bound,
  then minimizes predicted total cost inside that set. It is an ablation, not
  a silent replacement of the core objective.

The selector never claims that selecting a Source accepts reuse. Refined
timing produces a separate execution decision. The following invariants are
enforced by the data model:

- no selected Source implies reuse is rejected;
- accepted reuse implies a selected Source;
- full-recompute mode implies reuse is rejected;
- refined time rejection retains the selected Source for audit.
- a Source may be locked before `L_probe_max` when its quality-safe,
  predicted-economic interval separates;
- scheduler and refined measurement cannot replace the locked Source.

## Atomic waiting-window scheduling

`hybrid_strict` starts only complete steps that finish by `source_ready_ms`.
`hybrid_bounded_overrun` may start at most one complete step satisfying

`max(0, step_finish_ms - source_ready_ms) <= max_post_ready_overrun_ms`.

The simulator records `scheduled_step_finish_ms`, `a_resume_ms`,
`post_ready_blocking_ms`, hidden pre-ready work, useful B/C work, GPU busy time
and `load_interference_ms`. An interference value of zero means "not added by
this profile"; it is not a claim that copy and compute cannot interfere.

The corresponding configuration keys are `max_selection_layer`,
`selector_policy`, `reuse_ratio_tolerance`, `gamma`, `scheduler_policy`,
`max_post_ready_overrun_ms`, `load_interference_ms` and
`closed_loop_policy`. Positive overrun is
valid only with `hybrid_bounded_overrun`; all other policies reject such a
configuration instead of silently ignoring it.

## Repair ratio definition

For Source `s`, reuse boundary `l` and repeated segment `C`, the nominal
layer-wise repair ratio is

`r_l = number of C tokens recomputed at layer l / number of tokens in C`.

The pinned CacheBlend backend ranks C tokens by its V-drift rule. For a
requested ratio `r`, it recomputes zero tokens at `r=0`, all C tokens at
`r=1`, and otherwise `floor(r * |C|)` tokens. Ratio grids must use nested
sets: a token repaired at
10% is also repaired at 20%. Thus `r=0` means reuse all token KVs, while `r=1`
means recompute all `C` tokens at every active repair layer.

ProbeKV labels a ratio as safe only when both quality conditions pass. On the
tested grid `R`, the conservative label is

`r_safe(s,l) = min {r in R: every tested r' >= r also passes}`.

If no such ratio exists, the Source has no safe label at that boundary. Because
`r` does not include probe layers, early termination or loading time, the system
also reports the effective token-layer work ratio and uses measured total time
for admission; nominal repair ratio is never treated as a speedup estimate.

## Components

| Component | Implementation | Contract |
|---|---|---|
| Legacy canonical store | `source_store.py` | v3-v5 exact full-prefill, Kmax and reproduction lifecycle |
| v6 global Source pool | `global_source_pool.py` | 1-16 variants, global hard tiers, model soft quotas, value/FR eviction and model lifecycle |
| v6 request contracts | `v6_contracts.py`, `v6_manifest.py` | ordered regions, all-segment assignment and split isolation |
| v6 candidate budget | `candidate_budget.py`, `multisegment_selector.py` | linear all-within-budget comparison and independent early Source lock |
| Probe selector | `selector.py` | strict early exit plus explicit final policies |
| Budget calibration | `calibration.py` | isotonic baseline + case-grouped simultaneous conformal bounds |
| Case manifest | `manifest.py` | token hash plus content/document split isolation |
| RAG normalization | `rag_data.py` | three schemas; controlled and corpus-repeat kept separate |
| E1 orchestration | `experiment_jobs.py`, `e1_analysis.py` | deterministic shards, failure audit, safe labels |
| HF reference state | `reference_hf.py` | full-prefill pre-RoPE K/V/hidden/query correctness |
| Local E1/E2 loop | `local_e1e2.py` | labels, fit/calibration, locked evaluation, resume |
| Repair label | `labeling.py` | suffix-monotone safe ratio |
| Reuse planner | `cost.py` | refined total-cost admission; selection retained on rejection |
| Closed-loop controller | `orchestration.py` | selector abstention guard and scheduler-before-admission state machine |
| v6 request controller | `multisegment_orchestration.py` | multi-source feedback, marginal pruning, A/C staggered boundaries, legacy common reproduction and partial reuse |
| Prefetch | `prefetch.py` | P0-P4 and HBM-aware Dynamic |
| Scheduler | `scheduler.py` | strict atomic and bounded-overrun policies |
| Repair integration | `backend.py`, `cacheblend_backend.py` | stable runtime shim; canonical input remains immutable |
| Statistics/gates | `statistics.py`, `gates.py` | paired grouped inference |
| Audit trail | `io.py` | JSONL, optional Parquet, environment manifest |

## Meaning of the bandwidth inequality

`BW_available >= KV_bytes_per_layer / compute_time_per_layer` means that the
copy engine can deliver at least one layer of KV during the time the GPU spends
computing one layer. It is necessary for steady-state overlap, not sufficient
for perfect overlap: the first needed layer must already be present, HBM and
copy traffic may interfere, and scheduling gaps can expose transfer latency.
