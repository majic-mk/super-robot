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
- CacheBlend patch audit: deferred-56a100e tree, audit
  `/root/autodl-tmp/probekv_stage2/artifacts/deferred-56a100e-patch_audit.json`.

## Verified results

The earlier matched-prefix cost run is stored under
`native-6499c82-window4-final2` on the server. It reports:

- native Prefix/K-hook/`r=1`: passed;
- matched cost probe: passed with real CUDA execution;
- online immutable full-KV digest: not executed on the request path;
- winner preparation and repair-check timings: present;
- expected H2D activity count: 64 for this older window-4 run (the current
  deferred window-1 evidence below uses 62).

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

The later current-SHA closure also exercised the strict execution-shape
lookup. Its provisional table contained a single measured boundary/readiness
shape, while the live legacy dispatch reached a different completed depth and
copy-readiness vector. The estimator therefore returned
`no_exact_joint_measurement` and the request stayed dense. This is intentional:
the cost provider does not interpolate across depth, repair masks, or
ready-layer state, and it never substitutes a residual score or zero for an
unsupported cell. A reuse claim requires a new real-CUDA cost probe with the
same dispatch and execution shape.

A fresh current-SHA 512-token online closure replay was run under
`online-1bba8ed-deferred-closure-repeat-20260912-164515`. Prefix/K-hook/r=1
prerequisites passed and the winner was prepared, but FinalCommit returned
`UNSUPPORTED` with reason `no_exact_joint_measurement`; the request therefore
executed dense. This is the intended fail-closed behavior: the provisional
table contains a different exact joint mask/ready shape than the live repair
plan, so the estimator must not extrapolate or fill a zero-cost value. The
run is retained as evidence that cost-shape support, rather than correctness,
is the remaining online-closure blocker.

After correcting dense-counterfactual query construction, the 128-token
closure bound to `bc802de` (`online-bc802de-costshape-closure`) reached two
SUPPORTED exact joint queries and completed FinalCommit accounting. It
rejected reuse because the measured request total was 83.239 ms versus a
30.923 ms dense reference (`refined_marginal_pruned`); this is an economic
dense fallback, not a correctness failure. A GPU-hot/all-ready variant was
also exercised under `online-7aa000f-gpuhot-costshape-closure`; its live query
had `copy_in_flight=true` while the measured hot row had `copy_in_flight=false`,
so it correctly returned `UNSUPPORTED` rather than treating an all-ready row
as equivalent. These two runs establish both supported-cost rejection and
strict fail-closed behavior for an unsupported readiness shape.

To isolate selection depth, a d1-only single-Segment run was executed under
`native-78362a-d1-costshape-128` and
`online-78362a-d1-costshape-closure`. Both exact joint queries were
SUPPORTED, and Prefix/K-hook/r=1 correctness passed. The final predicted
request total was 112.947 ms against a matched dense reference of 47.682 ms,
so FinalCommit rejected reuse (`refined_marginal_pruned`). Reducing the
selection path from legacy checkpoints to d1 therefore did not produce a
positive single-request commit in this shape; the dominant issue is the
request-level selection/preparation overhead relative to a short dense
remainder, not only the number of checkpoint comparisons.

A longer 640-token d1-only run (`native-78362a-d1-costshape-640` /
`online-78362a-d1-costshape-640-closure`) also passed correctness and exact
cost support but rejected reuse: predicted total 112.502 ms versus dense
58.352 ms. The fixed-winner source arm itself was faster after the matched
boundary (45.669 ms versus 70.634 ms dense remainder), demonstrating that the
layer-wise reuse kernel can save compute in isolation; the end-to-end policy
loses that saving to selection, preparation, and request orchestration
overhead. This separates the remaining architecture problem from KV
correctness or load/compute overlap.

After adding an exact partial-ready observation to the cost-table builder, a
new 128-token revalidation was run under
`online-5015d3f-costshape-closure`. The query audit contains one supported
primitive/joint shape and one `UNSUPPORTED` live joint shape; FinalCommit again
failed closed with `no_exact_joint_measurement` and executed dense. This
confirms that the added row is consumed only when its complete mask and
ready-layer geometry match; it does not authorize unsafe shape reuse.

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

`r=1` correctness, matched cost support, and current-SHA layer-wise overlap
are ready for the next controlled diagnostic. The single-Segment online path
is safe (it falls back to dense when reuse is not economical or when an exact
cost cell is unsupported), but positive end-to-end gain and a reuse commit
remain **unproven**. Before any multi-Segment or Profile work, the next
experiment must collect a real cost-probe row for the live selected
completed-depth, repair-mask, ready-layer shape (or deliberately keep the
unsupported dense fallback as the documented boundary).

## Repeated live-pool replay diagnostic (2026-09-12)

Using commit `d502f567f084a30476c75ce293fa2ce284d9880f`, a 640-token Mistral
request was replayed with `--no-restore` so the same live Source Pool/runtime
was retained. The recorded replay remained `dense`: actual TTFT was 121.605 ms
versus a matched dense reference of 57.996 ms; predicted total was 118.864 ms
and no Source Variant was committed. This is valid real-CUDA evidence, not a
runtime failure. It shows that keeping the pool resident alone does not
amortize the current per-request selection/planning path; an exact, snapshot-
validated selection-result cache (with FinalCommit still evaluated per request)
is the next safe optimization candidate.
