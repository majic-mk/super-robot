# Dense-state handoff to pinned repair decoder loop

## Scope and implementation

Runtime SHA: `3b365bc` on `codex/probekv-schema10-evidence-integration`.
Implementation began at `2e32d8b`; the later commit fixes the profiler collector
to include the early dense layers and handoff as well as the continuation.
No production backend or admission policy is switched.

The one-shot continuation moves hidden states, residual, completed depth,
absolute position tensor, native attention and working paged KV references from
the request's real resumable session. The old session becomes terminal; stale
request/generation/position/KV references and second consumption are rejected.
The pinned decoder layer loop resumes at depth+1. It does not call embeddings
again and does not invoke completed Transformer layers again. The next-layer
K observation still has a projection cost; no claim of eliminating every
repeated projection is made.

The arm performs a real current K observation inside TTFT, but fixes the winner
for diagnostic isolation. There is no live multi-Source ranking or production
FinalCommit in this arm. Existing resident Source leases and native block
lifetimes remain owned by the request. Exceptions consume the continuation and
the outer request cleanup fences GPU work before freeing its reservations.

Native Prefix, streaming CPU/SSD and previously selective state are rejected.
All tests use Mistral, a 512-token Segment, zero cached Prefix, resident canonical
Source and the independently patched matched first-layer KV semantics. This
is a ProbeKV adapter using pinned CacheBlend decoder/attention code, not
unmodified upstream CacheBlend.

## Evidence

Patch tree: `b4cc27e2756be9266ce28a1726521c46d581bf26`.
Cumulative patch SHA256:
`24dc6fe958fcec7dd8fedb3fcb60f66e34e636aaa8f21dbe30de83fbc324c5007`.

Output root:
`/root/autodl-tmp/probekv_stage2/artifacts/`.

### Boundary 2: completed depth 1

Run: `native-mistral-continuation-3b365bc-b2`.
All three reuse backends pass r=1 against dense: exact greedy tokens and
teacher-logit relative-L2=0. Fixed15 continuation versus matched loop also has
exact tokens and relative-L2=0 (>=32 teacher positions). Shared repair mask and
mandatory dense ownership match. Layer invocation audit is exactly 1..32.

Source digest before/after:
`17afff09aeeeb206719a90559a9398512e12e1fe3b804af0a2e424adb25e77e9`.

Uninstrumented first-token host wall-clock, two warmups excluded:

| Arm | Three samples, ms | Mean, ms |
|---|---|---:|
| Dense | 73.918156, 73.881637, 73.969777 | 73.923190 |
| Matched pinned loop | 48.633562, 48.540246, 48.437808 | 48.537205 |
| Existing ProbeKV executor | 60.931364, 59.929628, 59.919057 | 60.260016 |
| Dense-state continuation | 50.513697, 50.492780, 50.414407 | 50.473628 |

Continuation saves about 16.24% versus the existing executor and 31.72% versus
zero-Prefix dense in this one shape. It remains about 1.94 ms slower than the
matched loop without current-state K observation/resumable handoff setup.
Three samples are diagnostic, not publication-quality performance evidence.

### Boundary 3: completed depth 2

Run: `native-mistral-continuation-3b365bc-b3`.
The same r=1 and fixed15 checks pass with exact greedy tokens and teacher-logit
relative-L2=0. All layers execute exactly once; Source before/after digest is
the same as the boundary-2 experiment.

| Arm | Three samples, ms | Mean, ms |
|---|---|---:|
| Dense | 74.040408, 73.813702, 73.899307 | 73.917806 |
| Matched pinned loop | 49.551313, 49.392407, 49.448377 | 49.464032 |
| Existing ProbeKV executor | 62.309249, 60.879627, 60.908549 | 61.365808 |
| Dense-state continuation | 51.512591, 51.471698, 51.414603 | 51.466297 |

Continuation saves about 16.13% versus the existing executor and 30.37% versus
zero-Prefix dense. Raw observation SHA256 values were recomputed for every
timing row in both runs; cached Prefix=0 and continuation layer order=1..32
were rechecked. No measurements were discarded from the three declared samples.

Both GPU processes finished. A800 memory usage returned to 0 MiB and GPU
utilization to 0%. The rented instance remains running; billing may continue.

Raw outputs, profiler traces, logits, both successful runs, the failed attempt
and cumulative patch audit are archived as
`artifacts/continuation-3b365bc-evidence.tar.gz` locally (same archive on server).
SHA256: `93dce90e47c3e2b12f4872eff5cf5e769fc5c6852f16f517c202f61248d42d8f`.

## Retained failed attempt

`native-mistral-continuation-2e32d8b-b2` passed numerical checks but failed during
final profiler aggregation with `ValueError: unknown control arm`. Its raw data
and failure are retained; the run is not declared successful and its timings
are not substituted for the fresh `3b365bc` results.

## Validation and next gate

764 local tests: 763 pass, one existing optional `ijson` skip. Compileall,
contract validator and diff whitespace check pass. Local tests validate state
contracts; they are not substituted for GPU results.

Next after both boundary checks: wire the actual independent SelectionState
lookup/comparison and winner freeze into this handoff, retaining all selector,
lease and admission costs in TTFT. First use the same single Segment and
matched Prefix condition; do not jump to multi-Segment. Production Prefix and
CPU/SSD streaming require independent integration and correctness gates.

Formal profile, runtime qualification, paper evidence and locked-test access
remain false. The fixed15 backend-equivalence check does not certify QA versus
dense or prove a live Source-selection benefit.
