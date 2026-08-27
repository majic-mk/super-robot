# ProbeKV v8 implementation status

## Locally implemented

- The complete CPU/no-GPU suite contains 353 passing tests with no v3-v7
  regression at the v8 freeze candidate.

- Independent protocol-v8 schema/configs; v3-v7 remain readable.
- Training-free residual-K selector, independently backed K-only SelectionState
  and bounded GPU comparison scratch; compare-time full-KV fallback is forbidden.
- Distinct single-candidate and budget-truncated `compared_k=1` behavior.
- Stable/strong early exit and maximum-depth residual-plus-cost selection.
- Single Artifact, tiered Replicas, logical/physical leases and orphan recovery.
- Predicted/Refined request planners with monotonic state transitions.
- Arbitrary multi-Segment simulation, A/C policies and partial reuse.
- Completed-depth K hook for Mistral and Qwen adapters.
- CacheBlend v8 mode, fixed 15% ceil repair and winner-only prefetch invariant.
- Profile generator, four Model x A/C handoffs, 140-job generator, schema-v4
  qualification Gate and v8 H1 offline diagnostic runner.

## Deliberately pending GPU evidence

- A800 Selector Profiles are not frozen.
- Neither model has passed Profile-bound 140/140 qualification.
- No v8 H1 sentinel or full H1 has run.
- No v8 output is paper performance evidence.
- v8 has not opened locked test data.

GPU rental is allowed only after the no-GPU readiness Gate is clean and bound
to the final pushed Git SHA.
