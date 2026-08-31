# ProbeKV v8 schema-v8 no-GPU handoff

Schema-v8 is the new online candidate.  Schema-v7 remains immutable for legacy
A/C and multi-checkpoint reproduction.

The selection phase is a dense d1/d2 barrier:

```text
all non-prefix Segments dense through d=1
  -> resolve winners that pass Source-local Gate1
  -> frozen d1 winners may start detached winner-only I/O
  -> one d=2 rescue for every unresolved Segment
  -> close selection for the whole request
  -> promote prepared winners / prepare remaining winners
  -> request-level FinalCommitAdmission before layer 2 or layer 3
```

Gate1 and final admission deliberately have different responsibilities. Gate1
is only an optimistic Segment-local feasibility screen with `gamma1=1.0`. Its
reuse-side lower bound contains unavoidable marginal support construction,
visible winner load and marginal repair work. It does not assign base
Transformer work, union repair, aggregate copy-stream time, or the already
sunk dense repair-check to every Segment independently. Final admission owns
the full request critical path, actual sunk time, aggregate load, union repair,
dense fallbacks for rejected Segments, and `gamma=0.8`.
PreparationAdmission remains a resource check, not an economic reuse gate.
Detached d1 preparation may overlap the dense d2 rescue, but it is never
execution-visible before barrier closure. After closure, independently ready
Segments may enter incremental request-level FinalCommitAdmission; this is not
the legacy A/C policy because no Source selection remains unresolved.

Storage has one backing copy per Artifact, not three permanent tier copies.
Pinned CPU is preferred after a Source is used.  CPU LRU demotes an idle source
to SSD.  SSD LRU removes the complete idle Source when SSD is full.  A leased,
copying, or executing Source cannot be selected as a victim.  GPU replicas are
transient winner-only copies.

The A800 sentinel must validate both barrier endpoints, Gate1 accounting,
joint final admission, CPU/SSD LRU transitions, single-backing identity, and
fixed/static/adaptive ratio-scope rules before any Profile is frozen.

The no-GPU handoff is now conditional on
`audit_v8_schema8_runtime_sources()`.  A contract-only implementation cannot
set `gpu_rental_ready_for_schema8_sentinel=true`: the audit requires the real
schema-v8 engine, request controller, selector, executor binding and
`run_v8_schema8_a800_sentinel.py` entry point.

Schema-v8 keeps three independent Profile identities:

```text
SelectionDepthProfileV8
RepairPolicyProfileV8
RuntimeCostProfileV8
```

The legacy `selector_profile_*` fields cannot unlock schema-v8.  Adaptive
per-Segment repair is invalid in the online path until the RepairPolicyProfile
is frozen from real GPU/development measurements. Adaptive ratios may differ
between Segments, but a request-level controller must choose one bounded ratio
vector using aggregate load and union-repair critical-path cost; independent
per-Segment decisions are forbidden. Static gradual uses reuse age rather than
absolute layer. The final 140-job matrix is generated only after all three
Profile SHAs exist.

The Profile-bound 140-job qualification matrix always includes `fixed_15` as
the correctness fallback plus the RepairPolicyProfile's selected online policy.
If adaptive gradual wins development, qualification must exercise adaptive
joint vectors; a matrix hard-coded to fixed/static cannot qualify it.

Physical preparation has two enforcement layers.  The request barrier admits
only frozen winners; the transfer call then requires a binding-specific
authorization that proves a logical lease, physical Replica lease and HBM
reservation all exist.  Final non-endpoint reuse additionally requires a
request-level joint admission decision.  Forced admission is labeled and
allowed only inside non-paper measurement cells; it cannot unlock H1/H2.
