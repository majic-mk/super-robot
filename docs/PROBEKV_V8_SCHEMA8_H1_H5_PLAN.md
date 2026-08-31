# ProbeKV v8 schema-v8 H1-H5 plan

This plan supersedes the older hypothesis ordering for schema-v8. Older
manifests remain readable but cannot be reported as schema-v8 H1-H5 evidence.
Every phase is model-specific and repeats for Mistral and Qwen.

## Runtime dispatch before H1-H5

The fast path is not assumed. Development/Profile evidence first evaluates
`d1_only` and `d1_d2_rescue` against the frozen full-candidate shadow. A fast
path is enabled only when the SelectionDepthProfile, RepairPolicyProfile and
RuntimeCostProfile are frozen and the schema-v8 runtime qualification passes.

```text
d1_only or d1_d2_rescue passes all frozen gates
  -> schema8 dense barrier
  -> request/layer-uniform I/O-balanced repair
  -> streamlined FinalCommitAdmission

otherwise, if legacy runtime qualification passes
  -> legacy_multicheckpoint
  -> Predicted Gate2
  -> scheduler feedback
  -> Refined Gate3

otherwise
  -> full dense
```

The selected protocol is frozen for an entire run. A request cannot start on
the fast path and silently switch to the legacy state machine. Both fast and
legacy qualification artifacts are retained, and the dispatch decision is
written into every result row.

The dense d1/d2 selection barrier, detached d1/d2 winner prefetch,
streamlined FinalCommitAdmission and request/layer-uniform I/O-balanced repair
are a single qualified fast feature set. They are not enabled piecemeal when
the fast selection Gate fails. The legacy fallback keeps its own multi-depth
Source selection, Predicted Gate2, scheduler-feedback accounting and Refined
Gate3 semantics.

## H1: Source opportunity and correctness

Question: for one exact non-prefix Segment, do historical Source variants
produce materially different safe repair costs?

- Use the three frozen development/pilot datasets and 1-16 exact-content
  Source variants.
- Compare the deep full-candidate oracle, legacy multi-checkpoint selection,
  fixed Source, latest Source, random Source and Cache-Craft CFO baseline.
- Retain the full diagnostic ratio grid including `r=0`, `r=0.15` and `r=1`.
- `r=1` must be dense-equivalent; Source/Artifact digests must remain unchanged.
- Primary opportunity gates remain: at least 25% of cases have Source spread
  at least 10 percentage points, and the oracle reduces safe repair cost by at
  least 10% relative to fixed/latest baselines.

H1 does not require the d1/d2 fast path. Failure stops the multi-Source main
claim rather than being hidden by scheduler improvements.

## H2: Selection depth and runtime dispatch

Question: can `d1_only` or `d1_d2_rescue` replace the preserved legacy
multi-checkpoint selector?

- Evaluate `d1_only`, `d1_d2_rescue`, `legacy_multicheckpoint` and the offline
  deep full-candidate oracle on the pre-isolated Profile-freeze partition.
- Use the same candidate budget, Source pool and CFO shortlist for all online
  policies.
- Freeze the first fast candidate satisfying all hard metrics:
  StateAvailability >= 0.99, SelectionCoverage >= 0.80,
  EarlyResolutionRate at completed depth 5 >= 0.80,
  WrongEarlyLockRate <= 0.05, mean stable normalized oracle regret <= 0.10,
  selection critical-path P95 <= 5% of dense TTFT, and zero illegal locks or
  budget-admission violations; realized selection-budget overrun rate must be
  <= 5%.
- If neither fast candidate passes, freeze the legacy multi-checkpoint +
  three-gate runtime. This is a valid negative result, not a job failure.

H2 outputs an immutable `h2_selection_candidate`. It deliberately does not
depend on the RepairPolicyProfile that H3 has not frozen yet. A legacy H2
candidate can never be upgraded to fast later.

## H3: Repair quality and policy freeze

Question: under the selected dispatch path, which winner-specific repair
policy preserves quality?

- Compare `fixed_15`, `static_gradual` and
  `load_recompute_aware_uniform` using K, V and normalized K/V diagnostics.
- If H2 selected the fast path, all three policies are runtime candidates. If
  H2 selected the legacy fallback, only `fixed_15` and `static_gradual` may be
  frozen for runtime; uniform I/O balancing remains a reported development
  diagnostic and cannot be smuggled into the legacy execution contract.
- The uniform controller uses one requested ratio for all active Segments at
  an absolute layer; each Segment still selects its own top-drift tokens.
- The I/O grid is measured, not interpolated. 0.15 is a quality reference,
  not a cap; any lower quality floor must be certified from development data.
- Require task-score non-inferiority lower CI >= -0.01, one-sided 95% tail
  violation upper bound <= 0.01, `r=1` dense equivalence and the preregistered
  no-reentry oracle-recall threshold.

The repair Profile is frozen before qualification and cannot be retuned from
H4/H5 results.

After H3, the system freezes the final `selection_runtime_path`. A fast H2
candidate becomes executable only if the repair and runtime-cost Profiles and
the schema-v8 runtime qualification also pass. If any later prerequisite
fails, it may only downgrade to the independently qualified legacy path (or
full dense); it may not switch to another fast depth policy. H4 and H5 consume
this final immutable dispatch artifact.

## H4: Systems, multi-Segment and storage

Question: does the selected complete system save request critical-path time
under realistic resource contention?

- Cover Prefix Cache + remaining non-prefix Segments, 1/2/5/10/37 Segments,
  K=1/2/4/8/16, partial/all/dense outcomes and mixed GPU/CPU/SSD Sources.
- Measure winner-only transfer, pinned staging/GDS fallback, layer readiness,
  union repair, blocking, interference, LRU demotion/deletion and concurrent
  requests.
- Run the selected fast or legacy contract end to end; also retain a matched
  legacy-three-gate ablation when the fast path is selected.
- Every admitted request must satisfy measured
  `T_reuse <= 0.8*T_dense-reference`; overlap intervals are charged once.
- Report TTFT, P95 TTFT, throughput, transferred/wasted bytes, admission rate
  and storage hit/eviction statistics.

## H5: Locked end-to-end evaluation

H5 is the only phase allowed to open the locked test partition. It starts only
after H1 passes, H2 freezes a selection candidate, H3 freezes repair, the
final runtime dispatch is frozen, and H4 passes systems correctness/admission.

Baselines are Full, native Prefix Cache, CacheBlend, Cache-Craft CFO, preserved
legacy ProbeKV and Oracle. Qwen is primary and Mistral is the second-model
replication. All methods use the same GPU, model stack, prompt, prefix hit,
generation settings and first-token endpoint.

Publication bands remain preregistered: Q1 candidate requires at least 10%
mean TTFT and throughput improvement plus at least 5% P95 TTFT improvement;
5-10% is a Q2 candidate range; below 3% stops or restructures the systems
claim. Missing, dense fallback, OOM and unfavorable rows remain in the audit.
