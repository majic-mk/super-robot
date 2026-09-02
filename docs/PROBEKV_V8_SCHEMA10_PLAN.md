# ProbeKV v8 schema10: dynamic Variant growth and Gate1 counterfactual

Schema10 preserves the schema8/9 execution data plane and separates three
questions that must not share one state or Profile:

1. whether an exact-dense historical context may be materialized;
2. whether an existing Source should be prepared;
3. whether the real request may irreversibly commit reuse.

## Variant identity and maturity

Every materialized Source has `DENSE_EXACT_CANONICAL` provenance from creation.
Value maturity is independent (`PROBATION`, `VERIFIED`, `EXPIRED`) and lifecycle
is independent (`ACTIVE`, `RETIRED`, `EVICTED`). Selective repair and the `r=1`
correctness endpoint never create canonical Sources.

A new Variant is comparison-eligible during probation. Two real current-state
comparisons verify it. At most two probation Variants per content are protected;
two subsequent same-content lookup opportunities without enough observations
move the Variant to `EXPIRED` maturity. Extra probation Variants remain usable
but are not eviction-protected. Migration between CPU and SSD is not a request use and does
not refresh recency.

Materialization reasons are content miss, complete-scope absolute mismatch and
budget-truncated exploration. Exploration requires exact dense execution, a
free Variant slot, an explicit quota and the write budget. It never claims that
the stored pool has been fully disproved and cannot replace an existing Variant
at `K=16`.

## Selection and Profiles

The Source score uses the exact integer trim count

```text
m_score = min(N - 1, ceil(rho_trim * N))
J_s = sum(non-top-m residuals) / (N - m_score)
```

The development grid is `rho_trim={0.10,0.15,0.20,0.25,0.30}`. Schema10 writes
`source_residual_trim_ratio`; schema9's `source_score_trim_ratio` remains a
read-only historical field.

`VariantAdmissionProfile` freezes residual thresholds, trim ratio, exploration,
probation and materialization budgets. `PreparationPolicyProfile` separately
freezes `gate1_mode=explicit_barrier|fused_advisory`. Atomic preparation
reservation and `FinalCommitAdmission` are mandatory in both modes.

At `K=16`, replacement is never implicit. Only complete-scope absolute
mismatch evidence may invoke the frozen `per_content_variant_lru_full_scope_only`
replacement policy, and replacement work must fit its own frozen budget.
Content miss and budget-truncated exploration cannot silently evict a Variant.

Within one canonical Segment/content bucket, the replacement victim is the
Source Variant with the oldest real request-use epoch. Merely comparing a
candidate does not refresh this epoch; Source selection/binding or actual reuse
does. Probation protection and in-flight leases override LRU. Cross-content
global capacity may still use value density, while CPU and SSD backing tiers
retain their independent LRU policies.

Gate1 removal is never inferred from pass rate. A side-effect-free development
counterfactual measures extra bytes, visible copy, staging, interference, HBM
byte-ms, wasted wall-clock and TTFT without Gate1. Fused advisory is eligible
only with zero correctness/final-gamma violations, complete FinalCommit
coverage, mean overhead at most 0.5% of dense, P95 at most 1%, and additional
winner bytes at most 1%.

## Materialization and growth metrics

Stored-pool Mismatch Precision includes only claims backed by a full-candidate
oracle.
Budget-truncated exploration is reported separately using Exploration Yield@32.
Useful Materialization Precision@32 requires a later positive-saving
`FinalCommitAdmission`. H1 replays the same trace from `K=1` under no growth,
complete-mismatch growth, controlled exploration and deep-oracle growth.

H2 freezes selection and admission Profiles. H3 evaluates fixed/gradual/I/O
repair. H4 evaluates storage, concurrency, probation, churn and Gate1 waste.
Only H5 may access the locked test.

All schema10 Profile, Gate, job and result artifacts are version-isolated from
schema3-9. This no-GPU implementation is not paper evidence and cannot unlock
H1/H2 before model-specific real-A800 qualification.

## Revised A800 Profile contract

Probation is a bounded grace policy, not a permanent value class. The newest
two probation Variants per content are protected for two same-content lookup
opportunities. Afterwards an unverified Variant is labelled `EXPIRED` for
audit and enters ordinary per-content LRU. `VERIFIED` receives no additional
LRU privilege. CPU/SSD migration and comparison do not refresh request-use
recency; selection, binding or actual reuse do.

New artifacts use `no_compatible_stored_variant_proven`. This proves only that
the current stored pool has no compatible Variant. Budget-truncated exploration
must set it false; the protocol no longer claims global context novelty.

The absolute threshold key is `(model, completed_depth, rho_trim)`. Mistral
freezes depths `{1,2,4,5,8}` and Qwen freezes `{1,2,4,5,7}` for every trim
ratio in `{0.10,0.15,0.20,0.25,0.30}`. Selection and admission are frozen on
the reference path `fixed_15 + explicit_barrier` before Repair, RuntimeCost and
Preparation Profiles. Final consistency may accept or invalidate the bundle;
it cannot retune selection.

RuntimeCost is factorized into selection, transfer, repair and scheduler
measurements plus a small joint-anchor set. The repair grid is
`{0.10,0.12,0.15,0.20,0.30,0.50,0.75,1.00}`. A full Cartesian product is
forbidden. Deterministic coverage/growth replay is marked as derived evidence,
not CUDA timing.

Operational `Coverage(K)` for `K={1,2,4,8,16}` is replayed causally and cannot
see future Variants. Oracle Coverage is a separately labelled future-pool upper
bound. The Profile run reports historical-inventory coverage; dynamic
materialization growth remains an H1 claim.

The 90 development requests per model select Profiles only. Even with zero
violations, the exact one-sided 95% Clopper-Pearson upper bound is about 3.27%,
so this stage cannot certify a 1% quality-tail rate. A later independent stage
uses at least 300 unique request-level units per model. Timing repetitions,
ratio rows and the 140 runtime qualification jobs are not independent quality
certification units.

Gate1 fused advisory additionally requires 18 real paired A/B anchors to
calibrate the side-effect-free shadow estimator. Atomic preparation reservation
and FinalCommitAdmission remain mandatory in either Gate1 mode.
