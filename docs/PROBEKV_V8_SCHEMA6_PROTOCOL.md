# ProbeKV v8 schema-v6 runtime contract

Schema-v6 is an explicit successor to the read-only schema-v5 result format.
It does not authorize old results, and schema-v5 gates cannot authorize it.
The research method remains training-free current-state Residual-K historical
Source selection.  This schema changes runtime accounting and concurrency
semantics, not the repair algorithm.

## Closed loop

The exact Prefix Cache runs first.  Remaining exact non-prefix canonical
Segments are CFO-ordered and compared using BF16 pre-RoPE K SelectionStates.
Gate 1 uses the Source-local theoretical boundary `completed_depth + 1`.  A
passing winner is frozen and protected by a LogicalSourceLease.

Gate 2 evaluates one request critical path:

```
T_G2 = T_sunk_actual + T_joint_future(inventory, boundaries, masks,
                                      states, scheduler/resources)
```

Every unresolved Segment is represented by dense fallback.  Segment-level
attributions are diagnostic and are never added to request TTFT when they
overlap on the same Transformer execution timeline.

A provisional winner may acquire an execution Replica lease and HBM
reservation.  A deferred winner may be prepared only after the 1.0x-dense
speculative-waste rule passes, using a `SPECULATIVE_PREPARATION` physical
lease.  Full KV transfer before Source freeze and non-winner full-KV transfer
are forbidden.  A ready speculative lease is atomically promoted to execution
when a later Gate 2 passes; there is no eviction window.

Gate 3 receives only the physically ready, Gate-2-provisional and policy-ready
subset.  It returns disjoint accepted/rejected sets.  Deterministic marginal
pruning finds a passing partial-reuse plan without enumerating Source
combinations.  Reuse commit is irreversible.

## Orthogonal state and snapshots

Selection, Gate 2 admission, physical preparation and commit are separate
axes.  In particular, `SOURCE_FROZEN + DEFERRED + READY + UNCOMMITTED` is a
legal state.  READY is not reuse admission.

Every planner result binds request generation, Segment-inventory generation,
scheduler snapshot, HBM reservation epoch and RuntimeCostProfile SHA.  A stale
result is retried or left deferred; it is never applied.

## HBM and measurement

Selection workspace, winner prefetch and committed execution share one HBM
reservation manager.  Free-reservable bytes are already net of active leases;
they are not subtracted twice.  Four GiB is held as the frozen safety reserve.
All candidates are compared in one vectorized batch when that batch fits;
otherwise the largest deterministic microbatch is used.

RuntimeCostProfile schema-v6 has category-specific axes for comparison,
SelectionState transfer, full-KV tier loading, joint dense remaining, repair,
union-mask remaining, copy interference and scheduler blocking.  SSD timing is
the real `SSD -> pinned CPU -> GPU` path.

The first Mistral A800 session is a four-hour sparse sentinel only.  It cannot
freeze a RuntimeCostProfile, run 140-job qualification or start H1.

After that sentinel passes at the same immutable code SHA, the next session
may run `run_v8_schema6_mistral_runtime_profile.py`.  Its 155-cell matrix uses
20 warmups and 100 retained measurements per cell, covers all eight Profile
categories, and freezes one RuntimeCostProfile for exactly one A/C policy.
The SSD cells measure the complete `SSD -> pinned CPU -> GPU` path.  This
session still cannot run the development selector sweep, 140-job qualification
or H1.  A new code SHA invalidates the earlier sentinel and requires it to be
rerun before Profile measurement.

## CFO provenance

`CFO_raw = alpha * CCI * (1 - beta_prime)` reproduces Cache-Craft Equation 12.
`CFO_operational = clip(CFO_raw, 0, 1)` is ProbeKV's explicitly named shortlist
handling.  Metadata is generated only during canonical full prefill.  The
attention accumulator uses post-RoPE Q/K, causal masking, real GQA head mapping
and a global streaming log-sum-exp denominator; per-block softmax is invalid.
