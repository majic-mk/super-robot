# Protocol v4: Source selection and final reuse admission

Protocol v4 treats two decisions as independent state:

- Source selection: `not_selected` or `selected`.
- Reuse admission: `not_evaluated`, `accepted` or `rejected`.

The online controller in `orchestration.py` enforces this order:

1. The probe selector uses conservative quality and predicted-cost bounds.
2. Abstention immediately executes full recomputation. No Source is loaded and
   no latest/default Source may be substituted.
3. A selected Source enters real loading and waiting-window scheduling.
4. The runtime returns the evaluated 1-based boundary, source-ready time,
   scheduled atomic-step finish, A resume time, post-ready blocking, overlap,
   interference and useful A/B/C work.
5. The runtime measures or profiles repair and matched full cost for that
   actual boundary.
6. `DynamicReusePlanner` recomputes total cost from these facts.
7. Only an accepted refined plan calls `execute_reuse`; otherwise
   `execute_full` runs while the selected Source remains in the audit.

The main configuration is:

```json
{
  "closed_loop_policy": "two_stage_refined_admission",
  "preliminary_economic_filter": true,
  "source_eviction_policy": "fifo",
  "replica_eviction_policy": "lru",
  "fixed_resident_sources": false
}
```

`legacy_pre_schedule_admission` remains available to reproduce v3 artifacts.
It records later scheduling feedback but deliberately does not use it to
change its already-final decision. New performance experiments must not use
that policy.

## Source lifecycle

Canonical Source identity is `(model_signature, content_hash, source_id)`.
Version retention and physical replica eviction are deliberately separate:

- `reject_when_full` preserves the old Kmax behavior.
- `fifo` evicts the oldest unleased canonical version without consulting
  predicted Source quality or cost, avoiding selector-dependent survivorship.
- `lru` evicts unleased GPU/CPU replicas by byte capacity and access time. It
  does not delete the canonical Source metadata or promote repaired KV.
- `fixed_resident_sources: true` forbids eviction and is used for the H1
  four-Source sensitivity pilot so placement cannot confound Source quality.

The store emits an `EvictionEvent` for every version or replica eviction.
Concrete tensor copy/free callbacks remain the responsibility of the serving
runtime adapter; the store owns policy, leases and audit semantics.
