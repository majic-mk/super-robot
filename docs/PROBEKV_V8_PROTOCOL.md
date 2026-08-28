# ProbeKV v8 frozen protocol

## Claim and non-claims

The v8 method is **Training-free Current-state Residual-K Historical Source
Selection**. The claim is the choice among canonical historical KV variants of
the same exact non-prefix Segment. Native Prefix Cache, CacheBlend token repair,
Cache-Craft CFO ordering, storage tiers, leases and scheduling are supporting or
baseline mechanisms, not separate novelty claims.

v8 has no learned selector, probability calibration, conformal predictor or
online `r_safe` model. Online repair is fixed at 0.15. The full ratio grid is an
offline H1 quality/correctness diagnostic only.

Experiment splitting and KV identity are deliberately separate. The frozen
train/development/test group is derived from the normalized document identity
(title + text) and is therefore identical for Mistral and Qwen. The per-model
`content_hash` is still derived from that model's exact token IDs and remains
the only key for KV content-bucket matching. Changing tokenizers may change KV
identity, but must never move the underlying document between experiment
partitions.

## Request path

```text
native exact Prefix Cache
  -> dense unaligned prefix tail
  -> canonical non-prefix Segments
  -> exact content bucket lookup
  -> correctness and K-state availability filtering
  -> CFO/metadata ordering
  -> budgeted current-state Residual-K comparison
  -> SELECTOR_DECISION_READY
  -> Gate 1 Source-local economic check
  -> Source freeze + LogicalSourceLease
  -> incremental Predicted Joint Planner (Gate 2)
  -> atomic PhysicalReplica leases
  -> winner-only, layer-wise full-KV preparation
  -> real scheduler feedback
  -> Refined Joint Planner (Gate 3)
  -> fixed 15% CacheBlend repair and reuse, or dense
```

Every exact non-prefix Segment receives an assignment. There is no Segment-count
cap and no Cartesian product of Source candidates. A request may reuse all,
some, or none of its Segments.

## Completed-depth K semantics

`completed_depth=d` means exactly `d` causal self-attention Transformer blocks
have completed. The observed pre-RoPE K enters block `d+1`:

\[
K_{obs}^{(d)}=W_K^{(d+1)}\operatorname{Norm}_{d+1}(h^{(d)}).
\]

`d=0` is a negative control and cannot lock a Source. Online checkpoints are
Mistral `{1,2,4,5,8}` and Qwen `{1,2,4,5,7}`. The last value is a maximum,
not a mandatory fixed layer; early exit remains enabled.

For Source `s` and token `j`:

\[
d_{s,j}=\frac{\lVert K_{current,j}-K_{source,s,j}\rVert_2}
{\max(\lVert K_{current,j}\rVert_2,\epsilon)}.
\]

The Source score ignores the rows that fixed repair would recompute:

\[
m_{score}=\min(N-1,\lceil0.15N\rceil),\qquad
J_s=\frac{1}{N-m_{score}}\sum_{j\notin TopK_{m_{score}}(d_s)}d_{s,j}.
\]

Actual CacheBlend repair uses a separate count:

\[
m_{repair}=\min(N,\lceil rN\rceil).
\]

Thus the score always retains a denominator, while the `r=1` correctness
endpoint can repair every token. Equal drifts use increasing absolute token
position as the deterministic tie breaker.

## Candidate counts and `compared_k=1`

```text
stored_k -> correctness_eligible_k -> selection_state_available_k
         -> metadata_ranked_k -> compared_k
```

If only one correctness-eligible Source exists, it may use the explicit
single-candidate path after the first legal K observation; no ranking margin is
claimed. If several Sources are eligible but budget permits one comparison, the
main method abstains because there is no runner-up. CFO Top-1 is an explicit
baseline only. Undefined margins are serialized as `null`.

For at least two compared Sources:

\[
M_d=\frac{J_{second}-J_{best}}{\max(J_{second},\epsilon)}.
\]

A strong margin locks immediately. A normal margin requires the same winner at
two consecutive checkpoints. One `eta` and one `eta_strong` are shared across
depths in each Model x A/C Profile. At maximum depth:

\[
J_s\le(1+\tau_{rel})J_{min}+10^{-6},
\]

then the lowest predicted total upper cost in the band wins. `10^-6` is only
numeric slack.

The schema-v5 state transition is explicit:

```text
PROBING -> SELECTOR_DECISION_READY -> Gate 1 -> SOURCE_FROZEN
```

Before `Lmax`, a Gate-1 rejection returns the Segment to probing. At `Lmax`,
the selector considers the residual band in increasing predicted-cost order;
if no member passes Gate 1, the Segment abstains and stays dense. A frozen
Source can never be replaced by the runner-up.

## Selection budget and movement

\[
B_{compare}=0.05T_{dense}-T_{shared-probe}-T_{metadata}
-T_{other-selection-sunk}.
\]

If non-positive, execution is dense. Otherwise a measured vectorized batch
curve chooses the largest feasible `compared_k`; CFO only orders entry. CUDA
Events measure components and request wall time drives admission, with no double
charging of overlap.

All Segments share one request-level `BudgetLedger`. A comparison batch must
reserve its predicted upper bound before launch, then settle actual critical
path time and release unused reservation. Starting an inadmissible batch is an
`admission_violation` (implementation error). A legally admitted batch that
runs longer than predicted is a `realized_overrun` (profile error, not a failed
job); further comparisons stop after the request budget is exhausted.

`SourceSelectionState` is exact BF16 pre-RoPE K at one completed depth. Its
identity excludes tier and locator. CPU/SSD hold backing data and GPU uses a
256 MiB bounded scratch. Selection cannot actively transfer any full-KV
Artifact, and non-winner full-KV transfer is zero. A candidate already resident
in HBM because of another request is not falsely counted as selection traffic.

## Artifact, Replica, lease and eviction

Each Variant has one canonical lossless BF16 full-KV Artifact. It normally has
one healthy backing Replica and optionally temporary CPU staging or GPU hot
Replicas. These are not three permanent copies. Promotion can temporarily make
extra copies; unused hot copies are evicted first, and a low-value Source may
lose every Replica and remain only as a tombstone.

`LogicalSourceLease` freezes the Variant and preserves its namespace, Artifact
and at least one healthy backing, without pinning a particular physical copy.
Predicted planning atomically leases concrete Replicas. Stale placement may
replan at most twice within the same Source and Artifact; it cannot switch to a
runner-up Source. TTL first marks `SUSPECT`; only confirmed orphan recovery
releases resources.

## Two-stage planning and admission

Source is selected once. Preliminary filtering, Predicted planning and Refined
admission share one cost identity:

\[
T_{reuse}=T_{probe}+T_{metadata}+T_{selection}+T_{visible-load}
+T_{post-ready-blocking}+T_{interference}+T_{repair-selection}
+T_{repair}+T_{remaining}.
\]

Gate 1 is Source-local. Gate 2 is an incremental request-level predicted plan;
only a Gate-2 `PROVISIONAL_REUSE` Segment may lease a physical Replica and
start winner-only preparation. Later frozen Segments are added without
removing already prepared Segments. Refined Gate 3 consumes real layer-ready,
A-resume, blocking, interference, transfer and boundary measurements.
Reuse requires

\[
T_{reuse,refined}\le0.8T_{dense-reference}.
\]

```text
UNDECIDED -> PROVISIONAL_REUSE -> FINAL_REUSE or REFINED_DENSE
UNDECIDED -> PREDICTED_DENSE (terminal for reuse)
```

Refinement can downgrade but cannot promote or reselect. Rejected selected
Sources remain in the audit.

Every request-level Gate-2/Gate-3 evaluation prices unresolved or uncommitted
Segments as dense fallback; it cannot assume future savings. Once a Segment
enters `REUSE_COMMIT`, an economic rollback is forbidden.

## Multi-Segment A/C policies

Policy A (`causal_commit_wait`) waits before an earlier reuse commit would alter
the clean state used to select unresolved later Segments. Policy C
(`immediate_staggered_closed_loop`) commits at the earliest feasible
per-Segment boundary and selects later Segments from the real policy-conditioned
state. Both use staggered boundaries, absolute-position union repair masks and
request-level resource planning. They receive separate frozen Profiles.

Policy A may select, lease and prefetch an early winner before downstream
selection closes; only commit waits. Policy C may commit an early Segment, and
a later Gate-3 rejection leaves that earlier commit intact while the later
Segment becomes dense.

## Profile and qualification evidence

Schema-v5 separates `SelectorPolicyProfile` from `RuntimeCostProfile`. The
pre-result `ProfileFreezeContract` fixes data partitions, metric definitions,
thresholds and regret normalization. The Selector Profile binds the runtime
profile used during profile freeze (`R_profile`). Final qualification binds a
runtime profile measured on the actual qualification GPU (`R_qual`); the two
may differ, but both hashes remain in the Gate.

Hard profile metrics use fixed denominators: StateAvailability and
SelectionCoverage divide by requests with at least one correctness-eligible
Source; EarlyResolutionRate@completed-depth-5 divides by legal locks.
MultiSourceEarlyExitRate@5 excludes single-candidate requests and is reported,
not gated. Residual regret reports both absolute regret and stable normalized
regret with denominator floor `2^-7`; that floor is a frozen numerical scale,
not the resolution of `J`.

## Evidence boundary

No-GPU output is non-paper evidence. The A800 order is correctness sentinels,
microbenchmarks, pooled development Profile freeze, Profile-bound canary,
four independent Model x A/C 140/140 qualifications, Gate, one H1 sentinel,
then stop. Profile parameters
cannot change after qualification or H1 inspection.
