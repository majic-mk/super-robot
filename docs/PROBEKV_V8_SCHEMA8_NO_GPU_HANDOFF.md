# ProbeKV v8 schema-v8 no-GPU handoff

Schema-v8 is the new online candidate.  Schema-v7 remains immutable for legacy
A/C and multi-checkpoint reproduction.

The selection phase is a dense d1/d2 barrier:

```text
all non-prefix Segments dense through d=1
  -> resolve winners that pass Source-local Gate1
  -> one d=2 rescue for every unresolved Segment
  -> close selection for the whole request
  -> winner-only preparation
  -> request-level FinalCommitAdmission before layer 2 or layer 3
```

Gate1 and final admission deliberately have different responsibilities.  Gate1
uses the Source-local same-origin horizon and `gamma1=1.0`; final admission uses
the full request critical path, actual sunk time, dense fallbacks for rejected
Segments, and `gamma=0.8`.  PreparationAdmission remains a resource check, not
an economic reuse gate.

Storage has one backing copy per Artifact, not three permanent tier copies.
Pinned CPU is preferred after a Source is used.  CPU LRU demotes an idle source
to SSD.  SSD LRU removes the complete idle Source when SSD is full.  A leased,
copying, or executing Source cannot be selected as a victim.  GPU replicas are
transient winner-only copies.

The A800 sentinel must validate both barrier endpoints, Gate1 accounting,
joint final admission, CPU/SSD LRU transitions, single-backing identity, and
fixed/static/adaptive ratio-scope rules before any Profile is frozen.
