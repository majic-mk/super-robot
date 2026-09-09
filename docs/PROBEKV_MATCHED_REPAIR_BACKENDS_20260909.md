# Resident single-Segment backend handoff: first equivalence gate

## Scope and architecture boundary

Keep Source Pool, exact lookup, Residual-K, leases, canonical provenance,
storage and FinalCommit in ProbeKV. Test replacing only repair execution.
The existing resumable backend remains the stable implementation. No default
backend switch, new schema, multi-Segment deployment or quality qualification.

This first experiment is **fixed winner, zero native Prefix, GPU-resident**.
It does not include online Source selection, Source creation, cold H2D or QA.
An 832-token constructed request contains C=[288,800), i.e. 512 tokens, and
32 mandatory suffix tokens. Boundary is layer 2; fixed15 repairs ceil(76.8)=77
C rows. Both backends retain the 320 mandatory current rows, for 397 active
rows total. Textual prefix tokens are present but native Prefix hits are zero
in every compared arm. Prefix+reuse evidence from other experiments is NOT
substituted for this stratum.

`ResidentRepairPlan` is immutable and binds Source/digest, request token hash,
Segment span, boundary, ratio and exact repair positions. It is diagnostic
only, not a production FinalCommit/admission token. Common repair positions
are captured from an untimed CacheBlend mask-bootstrap run. Each measured
CacheBlend run independently computes and must reproduce that mask; ProbeKV
still pays its ordinary repair-check computation before executing the same
external mask. Thus this is not a pure kernel benchmark with repair-check
overhead removed. All 32 layer masks are checked on the resumable path.

## Important semantic mismatch found, not a timing failure

Run `native-mistral-sharedmask-5ae398c-512` (code `5ae398c`) stopped before
paired performance measurements. r=1 was exact in both backends, and fixed15
greedy token IDs matched, but the 32-position teacher-logit relative-L2 was
0.02111928164958954 (>1e-4), despite identical masks. Failure evidence is kept.

Source inspection revealed a check-layer mismatch:

- The existing CacheBlend normal-loop adapter filters queries at status=1,
  but that layer's attention still sees current **full dense K/V**.
- ProbeKV status=1 already installs current active rows into historical K/V,
  so inactive C rows use **Source K/V at the same boundary**.

These are different execution semantics. Identical r=1 output alone cannot
establish equivalence for partial repair. The prior 49ms-vs-60ms comparisons
therefore cannot be attributed exclusively to loop overhead or GQA copies.

## Explicit matched-boundary adapter

Independent patch `0016-probekv-matched-boundary-source-kv.patch` adds an
opt-in metadata flag, default false. For the normal loop's check layer it
updates active current rows and uses historical inactive rows, matching
ProbeKV's first-reuse-layer semantics. Ranking, Source and active mask are
unchanged. This is a **CacheBlend-derived matched execution adapter**, not
unmodified upstream CacheBlend. Stable patches are not overwritten.

```text
patch tree: b4cc27e2756be9266ce28a1726521c46d581bf26
cumulative diff SHA256: 24dc6fe958fcec7dd8fedb3fcb60f66e34e636aaa8f21dbe30de83fbc324c5007
fixed15 mask SHA256: a24163be98b3d4d27627b3e516a3405ab62074e4dd1c2958c676cb0ee8f0e60d
```

Run `native-mistral-matched-boundary-80e32f3-512` passed both r=1 and fixed15:
greedy token equality and 32-position logit relative-L2=0. Canonical resident
Source digest before/after is unchanged. Three alternating measurements after
two warmups gave 74.305303ms dense, 54.274786ms matched adapter, 60.088517ms
resumable ProbeKV. These numbers included a newly introduced repeated source
inspection in the adapter's TTFT and are retained as an intermediate result.

A separate CPU measurement identified `inspect.getsource(forward)` at about
5.536822ms per invocation. This was a diagnostic-entry overhead, not GQA work.
Commit `bec83c6` caches this static capability check by loaded function identity
and records process-capability time outside the warm-hit endpoint. The frozen
on-disk patch tree/source checks remain mandatory before execution. Replaced
functions are checked again. No time is subtracted from previous measurements;
the corrected path is rerun in a fresh output directory.

Intermediate full evidence archive (including failed run, raw logits, traces,
logs and cumulative patch): `artifacts/matched-backends-80e32f3-evidence.tar.gz`.
SHA256: `011a14b5f8673fd766d403bcdcbcd629b35a50fd0552ed3108f4d220a68cc9b7`.

## Rerun after static capability caching

Execution code `bec83c6`, same independently frozen patch, fresh output
`native-mistral-matched-boundary-bec83c6-512`. r=1 and fixed15 again both have
identical free-generation token IDs and 32-position teacher-logit relative-L2
of **0.0**. Fixed15 mask hashes match in every repetition. Resident canonical
Source digest is unchanged. All raw observation hashes were recomputed and
verified. All compared arms have zero native cached Prefix tokens.

Two warmups excluded, three alternating uninstrumented measurements per arm:

| Arm | Samples, first-token wall-clock ms | Mean ms |
| --- | --- | ---: |
| Native dense | 73.999107, 74.036612, 73.922947 | 73.986222 |
| Matched-boundary CacheBlend adapter | 48.694014, 48.585903, 48.745967 | 48.675295 |
| ProbeKV resumable, same external mask | 61.044271, 60.137376, 60.107022 | 60.429556 |

Adapter warm-hit latency is 19.4512% lower than the resumable arm and 34.2103%
lower than this zero-Prefix dense reference. No outlier was removed, including
the 61.044ms ProbeKV sample. Three repetitions are a diagnostic result, not a
robust workload-wide performance estimate or published CacheBlend reproduction.
Capability cache lookup takes approximately 0.006ms per arm, separately
recorded; the first source-inspection cost is paid in bootstrap/setup. No prior
sample was adjusted by arithmetic subtraction.

The normal-loop adapter still computes its native V ranking, while ProbeKV
retains its normalized repair-check computation before mask replay. Therefore
this result supports the integration experiment, but does not establish that
all 11.754ms comes from the execution loop alone. In particular, this is not
evidence of live Residual-K selection benefit or natural-QA accuracy.

Local tests: 758 total, 757 successful and one existing `ijson` skip. GPU
processes finished and released their allocations (A800 reported 0 MiB).
The cloud instance was not stopped or released.

Final raw evidence is archived locally and remotely as
`matched-backends-bec83c6-evidence.tar.gz`. SHA256 verified at both ends:
`78e7f2cadb0fdce38b1990352ac24425ecf58a6ea2f5fd0734a4ed7b1a79dac3`.

## Next integration gate: preserve selector state, do not recompute it

The normal-loop adapter currently starts at token embeddings. Calling it after
a live selector would duplicate early-layer compute, so it is NOT yet a valid
production handoff. The next implementation must pass an owned continuation:

```text
completed_depth + hidden_states + residual + absolute active positions
+ native attention/block references + private working KV + sampling state
+ frozen Source/leases + layer-ready events + admitted repair plan
```

Continue at completed_depth+1, never rerun completed Transformer layers.
Retain the request arrival time so selector and handoff overhead remain in
TTFT. Validate r=1, fixed15 equivalence, per-layer execution counts, immutable
Source and resource cleanup before adding live multi-Source selection. The
zero-Prefix resident adapter must reject unsupported Prefix/streaming paths;
those remain on the established resumable backend until separately tested.

## Preregistered continuation diagnostic

The next opt-in `--repair-backend-continuation` arm transfers the real dense
resumable hidden state and residual after depth 1 or 2 into a pinned decoder
loop continuation. It uses the existing patched decoder/attention layers, but
does not claim to call unmodified normal model forward. No embeddings or
completed layers may execute again. It includes a current-state K observation
and handoff in request TTFT, but no live multi-Source ranking or FinalCommit;
the Source remains explicitly fixed for this diagnostic.

Cases: Mistral, 512-token C, zero native Prefix, GPU-resident Source, boundary
2 and 3 separately. Each must pass shared-mask fixed15 backend equivalence and
r=1 dense equivalence (greedy IDs and >=32 teacher logits, relative-L2 <=1e-4)
before 2 warmups and 3 measured alternating repetitions. Every Transformer
layer must be invoked exactly once; source digest must remain unchanged.
Failures retain their own SHA-bound output directory; no performance samples
after a correctness failure. This gate does not enable online deployment.

This experiment alone does not qualify natural QA, Source-selection benefit,
Prefix-enabled normal-loop execution, CPU/SSD streaming or full H1-H5.
All formal Profile, runtime qualification, paper and locked-test flags remain
false. No GQA5D experiment is promoted by these results.
