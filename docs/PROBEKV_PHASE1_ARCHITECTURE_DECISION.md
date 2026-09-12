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

At the 640-token diagnostic, the fixed-winner post-boundary saving was
approximately `70.634 - 45.669 = 24.965 ms`. The online policy exceeded the
matched dense reference by approximately `54.15 ms`, giving a simple
break-even estimate of `ceil(54.15 / 24.965) = 3` successful same-shape hits,
before cache-maintenance costs. This is a planning estimate, not a performance
claim; the repeated-request experiment must measure it with real wall-clock
traces.

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

## Hot-cache ownership correction (2026-09-12)

The 64 MiB terminal reservation failure was an adapter-ownership mismatch:
`execute_fixed_source_arm` retained the winner in the legacy adapter, while
the d1-only sentinel cleared the FAST adapter's separate hot-cache dictionary.
Process-exit GPU memory reclamation did not prove in-process cleanup correct.
The earlier blanket reservation release and removal of unfinished-event checks
were not valid fixes and have been removed.

Commit `b6b8d52011dd60cfc5e065159fdb301ad1553126` fences all diagnostic
adapters, releases only their explicitly owned hot reservations, preserves
unknown reservations for leak detection, and rejects active/pending requests.
Local regression: 842 tests run, 841 passed, 1 skipped.

Server 15695 run `/root/autodl-tmp/probekv_stage2/artifacts/native-b6b8d52-hotcache`
exited 0 with native Prefix/K-hook/r1 and matched cost probe passing.
`transfer.json` reports zero active HBM reservations; the GPU-hot r1 report
has 32 logit positions and relative-L2 0.0. These are diagnostic correctness
results, not production commit or QA/performance qualification.
The independent online process cannot inherit this process's GPU-resident
tensors; its use of these cost files alone must not be called a hot-cache hit.
Formal Profile, GPU qualification, online trace permission and paper evidence
remain false in this sentinel result.
