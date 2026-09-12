# ProbeKV single-Segment phase-1 report (2026-09-12)

## Scope

This checkpoint covers the single-active-request, single-RAG-Segment path only:
native Prefix handling, `r=1` correctness, cost support, winner preparation,
and layer-wise copy/compute instrumentation. It is not a Profile freeze,
qualification run, H1/H2 result, or paper evidence.

## Code and environment binding

- Local development commit used by the latest trace: `1bba8ed5e55fe6e550bee552de8c6c0d1c488d0e`.
- Server checkout used for the latest run: `/root/autodl-tmp/probekv_stage2/checkout-f15ddf5`.
- GPU: NVIDIA A800-SXM4-80GB, UUID `GPU-65efeff4-8f05-07b6-ecfa-3a19adf29493`.
- Model: Mistral audit `model_audit_mistral_f15ddf5.json`.
- CacheBlend patch audit: `patch_audit_1dee25a.json`.

## Verified results

The latest fresh run is stored under
`native-6499c82-window4-final2` on the server. It reports:

- native Prefix/K-hook/`r=1`: passed;
- matched cost probe: passed with real CUDA execution;
- online immutable full-KV digest: not executed on the request path;
- winner preparation and repair-check timings: present;
- expected H2D activity count: 64.

Representative cost values from this run (matched Prefix boundary to first
token) are:

| path | wall-clock ms |
|---|---:|
| dense remainder | 49.472 |
| fixed15 reuse path | 47.194 |
| fixed15 ready-to-first-token | 45.810 |
| winner preparation | 1.384 |
| repair check | 0.639 |

The result remains diagnostic only (`formal_profile_frozen=false` and
`paper_evidence=false`).

The exact-cost online closure run (`online-6499c82-cost128-closure`) selected
and prepared a Source, but correctly rejected final reuse because the refined
request cost was about 90.94 ms versus a matched dense reference of about
30.95 ms. This is an economic dense fallback, not a correctness failure.

## Overlap evidence

The current-SHA deferred window-1 trace is stored under
`native-1bba8ed-deferred-window1` on the server and reports:

```text
hardware_copy_kernel_overlap_observed = true
copy_kernel_overlap_union_ms = 0.768
h2d_union_ms = 0.853
kernel_union_ms = 19.691
layer_attribution_complete = true
expected_h2d_activity_count = 62
```

The 62 expected transfers are correct for window=1: layer 1 is initially
resident and layers 2--32 are the marked pending-copy set. The trace is
complete and provides current-SHA, correlated CUPTI evidence of real
load/compute overlap.

A fresh 512-token rerun using the same current SHA and audited 0013 tree is
stored under `native-1bba8ed-deferred-window1-current`. It independently
reports:

```text
hardware_copy_kernel_overlap_observed = true
copy_kernel_overlap_union_ms = 2.591
h2d_union_ms = 2.855
kernel_union_ms = 26.853
layer_attribution_complete = true
expected_h2d_activity_count = 62
```

This 512-token run is the primary overlap evidence for the checkpoint; the
128-token run above is retained as a shorter-shape cross-check.

The earlier `6499c82` window-4 trace reported:

```text
hardware_copy_kernel_overlap_observed = false
copy_kernel_overlap_union_ms = 0.0
h2d_union_ms = 0.752
kernel_union_ms = 20.394
layer_attribution_complete = false
expected_h2d_activity_count = 64
```

Because attribution is incomplete, this run cannot be used to claim either
successful or absent overlap for every layer; it is retained as an incomplete
measurement and not used for performance conclusions.

For comparison, two earlier A800 deferred-window runs, bound to commit
`56a100ed071f0e9db0c1ba40d457bc8c561f9313`, contain complete CUPTI evidence of
real overlap:

| run | overlap union ms | H2D union ms |
|---|---:|---:|
| `native-mistral-56a100e-deferred-window1-512-attempt2` | 2.558 | 2.769 |
| `native-mistral-56a100e-deferred-window2-512` | 2.445 | 2.671 |

These historical runs are retained as an independent cross-check of the
asynchronous data plane; the current-SHA window-1 result above is the evidence
used for this checkpoint. Instrumented traces remain diagnostic and are not
paper performance evidence.

An earlier attempt with the non-deferred CacheBlend tree was rejected before
model execution because that tree did not contain the independently audited
0013 patch; the failure remains in `/tmp/probekv-window1-defer.log`. It was
superseded by the successful current-SHA run above using the audited deferred
tree, and is retained only as environment/patch provenance.

## Local no-regression gate

The current tracked worktree passes:

```text
compileall: passed
unittest: 833 passed, 1 skipped
validate_contract.py: valid=true, errors=[]
git diff --check: passed
```

The skipped test is the pre-existing optional-`ijson` dependency skip. No
locked test, formal Profile, or multi-Segment experiment was accessed in this
checkpoint.

## Phase-1 disposition

`r=1` correctness and matched cost support are ready for the next controlled
diagnostic. The single-Segment online path is safe (it falls back to dense when
reuse is not economical), but positive end-to-end gain and current-SHA overlap
remain **unproven**. The next experiment should therefore target only the
current-SHA overlap/attribution path before any multi-Segment or Profile work.
