# ProbeKV phase-1 architecture decision

## Evidence boundary

This decision is based on real Mistral A800 single-active-request traces at
128, 512 and 640 non-prefix tokens, using the deferred layer-timing patch and
exact measured joint-cost support. It does not claim a Profile, qualification,
multi-Segment result, or paper performance evidence.

## Findings

1. Prefix handling, K-hook depth semantics, `r=1` dense equivalence,
   absolute-position masks, digest invariants, lease cleanup and layer-wise
   H2D/compute overlap are correct on the tested path.
2. The fixed-winner reuse kernel is faster than the matched dense remainder
   after the reuse boundary. At the 640-token point the measured post-boundary
   times were 45.669 ms (reuse) versus 70.634 ms (dense remainder).
3. End-to-end online selection did not pass the request-level `gamma=0.8`
   admission at 128, 512 or 640 tokens. Both legacy multi-checkpoint and
   d1-only selection were tested. The dominant cost is the per-request
   selection/preparation/planner path, not failed overlap or KV corruption.
4. Missing or mismatched joint cost geometry is fail-closed: the estimator
   returns `UNSUPPORTED` and the request executes dense. No unsupported cell is
   interpolated or assigned a zero cost.

## Frozen architecture decisions

The following components remain part of ProbeKV:

- native Prefix Cache first, with dense remainder ownership;
- exact content bucket and canonical Source Variant identity;
- Residual-K Source selection and d1/d2 plus legacy fallback paths;
- winner-only layer-wise preparation with PhysicalReplicaLease and HBM
  reservation;
- request-level FinalCommit admission with `gamma=0.8`;
- fixed-winner reuse/repair and load-compute overlap instrumentation;
- strict measured-cost support and dense fallback.

The following are **not** promoted to the main performance claim yet:

- a single short-request online selection followed by reuse;
- any result that only measures the fixed-winner kernel;
- any unsupported-cost or forced-admission run;
- multi-Segment execution before the single-Segment policy has a positive
  end-to-end case.

## Required next architecture experiment

The next experiment must test amortization rather than add another selector:

```text
one exact dense request publishes Source
→ Source/selection state remains resident
→ repeated identical current-prefix requests
→ selection result is reused only when token identity, model signature,
  Source/artifact digest and pool generation all match
→ FinalCommit still runs for every request
```

The cache must be an exact-identity optimization, not a learned prediction.
Any pool generation, artifact digest, prefix condition or model/runtime
signature change invalidates it. A cache hit must still verify Source lease,
Replica readiness and measured joint-cost support.

If repeated-request amortization cannot produce a positive end-to-end case,
ProbeKV should be restricted to workloads with sufficiently long remaining
prefill or high Source reuse frequency; adding more checkpoint comparisons or
more Source candidates is not justified by the current evidence.

## Stop conditions before multi-Segment

Do not start multi-Segment or formal Profile work until at least one of the
following is demonstrated with real CUDA timing and matched dense baselines:

- a repeated-request exact-selection-cache path passes `gamma=0.8`; or
- a long-context single-Segment path passes `gamma=0.8` without forced
  admission; or
- a measured workload-level amortization analysis shows positive aggregate
  TTFT including all selection, preparation and fallback costs.

All failures, unsupported queries and dense fallbacks remain valid evidence
and must be preserved.
