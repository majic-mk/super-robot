# ProbeKV v7 next-phase plan: canonical segments, layered identity, and joint planning

## 0. Status and research boundary

This document consolidates the design decisions reached after the frozen v6
prefix-hardening commit `fea69c9`. It is the implementation contract for the
next development phase; it is not evidence that v7 has been implemented or
GPU-qualified.

The v7 paper claim remains narrow:

> Fresh target-model early state selects, among canonical historical states for
> the same exact non-prefix Segment, the Source Variant with the smallest
> conservative safe total-cost upper bound.

The following remain supporting mechanisms or baselines rather than separate
novelty claims:

- native exact Prefix Cache;
- canonical RAG segmentation;
- CacheBlend repair;
- Cache-Craft CFO metadata and eviction baselines;
- Artifact/Replica management;
- prefetch, scheduling, and request-level joint admission.

Protocol v3-v6 configurations and artifacts remain readable and unchanged.
v7 must use an explicit new protocol version and new result schema. No v7
artifact may unlock a v6 gate, and no v6 qualification gate may unlock v7 H1.

## 1. Problems identified and frozen resolutions

| Problem | Frozen v7 resolution |
| --- | --- |
| `192` was confused with the online Prefix Cache minimum | `192` remains only a qualification-sentinel fixture. Production native prefix reuse uses the longest exact prefix exposed as complete engine blocks. |
| CacheBlend's 512-token experimental chunk was confused with a fixed KV allocation slot | `512` is a target RAG segmentation length and experiment point, not padding or a mandatory Segment length. |
| Mechanical `512/512/1` splitting creates an uneconomic tiny Segment | Use deterministic semantic boundary search and a short-tail penalty/rebalance; never pad the token content. |
| Hard block alignment may split an important sentence | Alignment is a soft optimization objective. Paragraph/sentence integrity may win over exact alignment. |
| Runtime block size could silently change canonical boundaries | Freeze `alignment_quantum=16` in the canonicalizer signature. Audit the runtime block size separately. Never derive canonical boundaries dynamically from the current device. |
| A chunker-version mismatch was conflated with KV mathematical incompatibility | Canonicalizer namespace controls provenance and calibration eligibility, not exact token-content identity. |
| Model mathematics and runtime/storage format were mixed in one identity | Separate `ModelMathSignature`, `TokenizerSignature`, Source Variant provenance, KV Artifact format, and Physical Replica placement. |
| Different historical positions could collapse into one Source | Source Variant identity includes the historical prefix and position-ID provenance. |
| GPU/CPU/SSD copies could appear to be different Sources | A Source Variant owns one canonical Artifact; that Artifact owns physical Replicas. Tier movement never changes the Source Variant. |
| Artifact fields were checked before an Artifact had been chosen | Split Source Variant correctness, calibration coverage, Artifact compatibility, Replica feasibility, and runtime admission. |
| Moving all storage checks after Source selection would hide access cost from the selector | Validate the unique canonical Artifact and preview its Replica access plans before selection; bind and revalidate one Replica after Source lock. |
| Probe cost was multiplied by the number of candidates | Shared early forward, request metadata, and current-state extraction are request-level sunk costs and are charged once. |
| Only the selected candidate's comparison cost was charged | Charge the actual comparison batch covering every compared candidate. |
| A preview could become stale under concurrency | Version every preview and use compare-and-bind with entity generations, placement epochs, leases, and same-Source replanning. |
| Independent multi-Segment admissions ignore shared resources | Per-Segment Source locks feed a request-level joint Replica, load, boundary, union-repair, and admission planner. |

## 2. Prefix Cache and non-prefix Segment granularity

### 2.1 Native Prefix Cache

Let `L_exact` be the longest exact token prefix shared by the current request
and an engine cache entry, and let `B_runtime` be the actual vLLM block size.
For the pinned vLLM 0.4.1 implementation, only complete immutable blocks are
shared, so the native reusable prefix is:

\[
L_{\text{native-prefix}}
=
B_{\text{runtime}}
\left\lfloor\frac{L_{\text{exact}}}{B_{\text{runtime}}}\right\rfloor.
\]

For `L_exact=1025` and `B_runtime=16`, the engine reuses 1024 tokens and
recomputes the one-token exact tail. The tail is not a ProbeKV Source candidate.
ProbeKV does not add a custom partial-block Prefix Cache merely to save at most
`B_runtime-1` tokens.

The 192-token prefix remains a non-paper sentinel length chosen to expose
multiple real cached blocks. It is not an online minimum.

### 2.2 Canonical non-prefix RAG Segments

RAG retrieval works with logical, content-addressable Segments. They are not
required to equal the physical vLLM block size and are never padded to 512.
Their real token IDs and real token count define their content.

The canonicalizer freezes:

```text
canonicalizer_version: semantic_block_v1
target_tokens: 512
min_tokens: 128
max_tokens: 640
alignment_quantum: 16
search_window_tokens: 64
alignment_policy: soft
padding: false
tail_policy: semantic_rebalance
```

These are initial v7 defaults and must be selected once on train/calibration
data before locked experiments. The formal CacheBlend-matched controlled arm
also reports fixed target sizes such as 256/512/1024 using the same tokenizer
and deterministic boundary implementation for every compared method.

### 2.3 Semantic and alignment objective

Around the target endpoint, the chunker considers paragraph ends, sentence
ends, structural breaks, and aligned endpoints. For a candidate endpoint `t`:

\[
Score(t)
=w_pS_{paragraph}(t)
+w_sS_{sentence}(t)
-w_lD_{length}(t)
-w_fF_{fragment}(t)
-w_tP_{tail}(t).
\]

One length and fragmentation definition is:

\[
D_{length}(t)=\frac{|N_t-M|}{M},
\qquad
F_{fragment}(t)=
\frac{(Q_{align}-(N_t\bmod Q_{align}))\bmod Q_{align}}
{Q_{align}}.
\]

Paragraph and sentence integrity receive higher priority than the soft
fragmentation term. A high-quality semantic endpoint may therefore be selected
even when it is not a multiple of 16.

Alignment is motivated primarily by paged GPU KV storage and transfers. Packed
CPU/SSD tensor files and summaries store actual rows and do not require this
alignment. Nevertheless, the canonical boundaries are tier-independent:
moving a Source among GPU, CPU, and SSD never invokes the chunker again.

### 2.4 Short-tail handling

A `512/512/1` split uses little physical padding but creates an uneconomic
one-token Segment with its own metadata, summary, comparisons, load, and
scheduling events. The tail penalty therefore searches for a semantic
rebalance such as `512/384/129` when suitable. This example is not a mandatory
split; semantic boundaries may choose another deterministic partition.

For physical block size `B`, independent chunk lengths `N_i` have internal
fragmentation:

\[
W=\sum_i((B-(N_i\bmod B))\bmod B).
\]

This is unused physical capacity, not lost token content.

## 3. Stable canonicalization and exact lookup

Every retrieval occurrence follows exactly this order:

```text
retrieved document + revision
  -> deterministic text normalization
  -> model-specific tokenizer
  -> frozen canonical chunker
  -> Segment provenance and exact token IDs
  -> exact reuse-content key
  -> content-bucket lookup
  -> Source Variant candidates
```

Source lookup never searches for a historically cached chunk with an
approximately similar length. Approximate semantic retrieval chooses documents;
ProbeKV Source lookup within those documents is exact-token identity only.

The same raw document may produce different boundaries under Mistral and Qwen
because tokenization is model-specific. Their namespaces, Source Variants, and
manifests remain isolated.

## 4. One provenance layer and four KV reuse identity layers

v7 defines five conceptual objects. The first is data provenance; the next
four form the KV reuse identity hierarchy.

```text
Canonical Segment Provenance
  -> Reuse Content Bucket
  -> Source Variant
  -> KV Artifact
  -> Physical Replica
  -> mutable Physical Locator
```

They respectively answer:

1. Where did this Segment come from and how was it cut?
2. What exact model-token content is it?
3. Under which historical prefix and positions was this state produced?
4. In what KV representation and serialization is that state stored?
5. Which materialized copy exists, and where is it currently placed?

### 4.1 Canonicalizer signature and Segment provenance

`alignment_quantum` is a frozen canonicalizer parameter rather than a dynamic
runtime property:

\[
S_{chunker}=H(
S_{tokenizer},
S_{normalization},
chunker\ version,
M,N_{min},N_{max},
Q_{align},W,
semantic\ policy).
\]

Segment provenance is:

\[
ID_{provenance}=H(
document\ ID,
document\ revision,
S_{chunker},
segment\ ordinal,
token\ span,
content\ hash).
\]

The provenance identity supports dataset grouping, split isolation,
reproduction, and calibration auditing. It does not by itself decide whether
two KV contents are mathematically identical.

### 4.2 Model mathematics, tokenizer, and reuse content

`ModelMathSignature` contains the properties that define model mathematics:

```text
weights digest/revision
architecture and layer count
attention/Q/K/V geometry
RoPE and positional semantics
sliding-window semantics
weight transformation or quantization semantics
```

It excludes GPU type, CUDA version, storage tier, file layout, transient
handles, and runtime patch SHA.

`TokenizerSignature` covers tokenizer files, configuration, normalization,
special tokens, and prompt/template policy. Compatibility with the model math
signature is validated explicitly.

The reuse content key is:

\[
K_{reuse}=H(
S_{model-math},
S_{tokenizer},
N_C,
tokenIDs_C).
\]

Two canonicalizer versions that happen to emit exactly the same model tokens
may share a content bucket. A canonicalizer namespace mismatch may still make
the request ineligible for the paper's calibrated selector. That is a
calibration limitation, not a Transformer/KV correctness failure.

### 4.3 Source Variant identity

A content bucket may hold 1-16 canonical historical Source Variants. Each one
describes a distinct historical model state:

\[
ID_{variant}=H(
K_{reuse},
H_{historical-prefix},
H_{position-ids},
ID_{occurrence},
S_{model-math},
canonical\ origin).
\]

Required provenance includes:

```text
historical_prefix_token_hash and token count
source_position_start/end
position_ids_digest
source_occurrence_id
model_math_signature
origin=full_prefill
canonical_source_state_digest
```

Readable start/end fields do not replace the strict position-ID digest. Even
with pre-RoPE K storage, earlier hidden states and V depend on historical
attention and positions. A short-prefix and a long-prefix occurrence of the
same exact Segment are therefore separate Source Variants in the same content
bucket. Selective-repair results never become canonical Source Variants.

### 4.4 Canonical KV Artifact identity and digest domains

The v7 main protocol permits exactly one full-KV Artifact per Source Variant:

\[
ID_{artifact}=H(
ID_{variant},
S_{artifact-format},
artifact\ bytes\ digest).
\]

The canonical Artifact is lossless BF16, stores pre-RoPE K and raw V, and
covers layer/head geometry, RoPE semantics, serialization version, producer
runtime, and patch compatibility. Tier-specific packing and paged layouts are
Replica properties rather than additional Artifacts.

The Source Variant stores:

```text
canonical_source_state_digest
```

The unique Artifact stores:

```text
parent_source_state_digest
artifact_logical_digest
artifact_bytes_digest
fidelity_class: lossless
```

Digest domains are tagged and cannot be compared blindly. INT8 summaries are
selector metadata and are not full-KV Artifacts. Lossy or compressed full-KV
representations are outside v7; a future H4b protocol must introduce a new
explicit compatibility and calibration contract rather than silently adding a
second Artifact to v7.

### 4.5 Physical Replica and mutable locator

A Replica is a lifetime-scoped materialization of an Artifact. Its stable
identity is not a CUDA pointer, block ID, file descriptor, or pathname.

```text
replica_id
artifact_id
replica_generation
tier
placement_state
created_at/destroyed_at
derived_from_replica_id
```

The mutable locator contains GPU block IDs, a pinned-buffer locator, or an SSD
path/offset. Relocation inside one tier updates the locator and placement epoch.
A cross-tier copy normally creates a new destination Replica linked to the
source Replica; it does not change the Source Variant or Artifact identity.
Replica IDs are never reused, and generation and placement epochs increase
monotonically to prevent ABA errors.

## 5. Experiment compatibility is not runtime correctness

The paper freezes `alignment_quantum=16` and requires the formal vLLM block
size to be 16. Runtime qualification records:

```text
alignment_quantum
runtime_vllm_block_size
alignment_quantum_matches_runtime
experiment_contract_compatible
gpu_runtime_correctness
```

If the runtime block size is 32, the Segments and KV may still be mathematically
correct. The result is:

```text
alignment_quantum_matches_runtime=false
experiment_contract_compatible=false
gpu_runtime_correctness=not_evaluated or independently passed
h1_h2_execution_allowed=false
```

It is rejected from the frozen experiment because the storage/profile contract
changed, not because KV reuse is intrinsically incorrect.

## 6. Eligibility, preview, selection, binding, and admission

One `eligible` Boolean is forbidden. v7 separates why a request does not reuse.

### 6.1 SourceVariantEligibility

This checks the historical state before the selector:

```text
exact K_reuse match
model-math and tokenizer compatibility
canonical full-prefill origin
complete historical-prefix provenance
complete position/RoPE provenance
canonical state digest and read-only lifecycle
```

Failure means the Source Variant cannot be used as a valid historical state.

### 6.2 CalibrationEligibility

This checks whether the selector and conservative quality model cover the
request/Source pair:

```text
canonicalizer/calibration namespace
Segment length and semantic-boundary regime
historical prefix and position feature support
Source count, probe layer, summary format, and feature envelope
```

Failure means the KV might be mathematically valid, but ProbeKV cannot provide
the frozen quality guarantee. The Segment abstains and executes dense; the
Source remains stored.

### 6.3 ArtifactCompatibility and ReplicaFeasibilityPreview

Before Source selection, ProbeKV verifies the unique canonical Artifact and
reads versioned Replica metadata only. It enumerates currently plausible
Replica access routes for each eligible Source Variant. It does not yet lease,
copy, or bind a concrete Replica.

This preview is necessary because the Source selector minimizes safe total
cost. Moving every Replica check after selection would make it choose a
historical state without knowing whether it is on GPU, CPU, SSD, or not
loadable at all.

### 6.4 Source selection and freeze

The selector chooses a Source Variant, not a physical Artifact or Replica. It
uses current-state quality features plus the best conservative predicted access
plan. Once selected, scheduler and refined planner may not replace it with
`latest`, a default, or the second-ranked Source.

### 6.5 Artifact binding and RuntimeEligibility

After Source lock, ProbeKV revalidates the unique Artifact and preview, then
atomically binds a Replica. `RuntimeEligibility` has predicted and refined
states:

- predicted admission uses the current binding, profiles, and economic bound;
- refined admission uses actual ready time, scheduler progress, interference,
  post-ready blocking, actual boundary, and union-repair profile.

Runtime rejection makes the current Segment/request dense while retaining the
Source Variant and its valid canonical Artifact.

## 7. Shared and candidate-contingent cost accounting

At probe checkpoint `q`, all already incurred request-level work is charged
once:

\[
T_{sunk,q}=
T_{shared-probe,q}
+T_{shared-metadata,q}
+T_{summary,q}
+T_{compare-batch,q}
+T_{speculative-visible,q}.
\]

`T_compare-batch` covers every candidate actually compared. A vectorized
comparison batch is measured/profiled as a batch and is not assumed to equal
the sum of isolated candidate timings.

For Source Variant `s`, its unique Artifact `A(s)`, and feasible Replicas
`R(A(s))`, define future conditional cost:

\[
\widehat F_s^{upper}=
\min_{\rho\in R(A(s))}
\widehat F^{upper}(s,A(s),\rho),
\]

\[
\widehat F^{upper}(s,a,\rho)=
\widehat T_{visible-load,\rho}
+\widehat T_{post-ready-blocking,\rho}
+\widehat T_{interference,\rho}
+\widehat T_{repair-selection,s}
+\widehat T_{repair,s}
+\widehat T_{remaining,s}.
\]

The predicted request cost is:

\[
\widehat C_{request}^{upper}(s)
=T_{sunk,q}+\widehat F_s^{upper}.
\]

`T_sunk` is constant across candidates at one checkpoint and can be omitted
from the ranking implementation, but it must be restored for admission and
reported total cost.

The request-level probe budget is:

\[
T_{shared-probe}
+T_{shared-metadata}
+T_{summary}
+\sum_qT_{compare-batch,q}
\le 0.05T_{dense-reference}.
\]

It never multiplies a shared early forward by `K`.

Initial and refined costs retain the same request-arrival-to-first-token
endpoint and component identity. The first uses versioned conservative
predictions; the second substitutes actual incurred scheduler feedback and
boundary-conditioned future profiles without double charging load,
interference, or post-ready blocking.

## 8. Versioned previews and same-Source fallback

Each `PredictedAccessPlan` records:

```text
access_plan_id
source_variant_id
artifact_id and artifact_generation
replica_id and replica_generation
placement_epoch
pool_snapshot_id
scheduler_snapshot_id
profile_version
per-component predicted upper costs
created_at and valid_until
```

Global pool or scheduler snapshot IDs are audit context, not strict locks: an
unrelated cache event must not invalidate a plan. Binding checks the relevant
entity generations, placement, lease state, and resource reservation.

After Source Variant `A` is frozen:

1. If the preferred `A1/R1` plan is still valid, bind and use it.
2. If it is stale, acquire a selection lease and replan only among Replicas of
   Source `A`'s unique Artifact.
3. Charge replan latency, extra waiting, and wasted transfer to refined cost.
4. If Source `A` has no economically feasible plan, execute dense.
5. Never switch to Source `B` under the v7 main protocol.

A future Source-reselection protocol would require a new explicit policy and
would charge new comparison/waiting costs. It is not part of v7.

## 9. Multi-Segment request-level joint planning

The same early-layer pass serves all active Segments. Every Segment may lock a
Source Variant independently at a different checkpoint, but independent
standalone costs cannot decide final runtime admission because Segments share
HBM, PCIe, copy streams, CUDA streams, scheduler slots, and prefetch windows.

v7 therefore uses:

```text
per-Segment current-state Source ranking and Source freeze
  -> request-level joint Replica and execution planner
```

For locked Source `s_i`, the joint planner chooses:

```text
x_i: reuse or dense
A(s_i): unique canonical Artifact
rho_i: Replica
b_i: actual reuse boundary
q_i: repair ratio
sigma: load/schedule order
```

Its objective is:

\[
\min_{P}\quad
T_{sunk}
+T_{load}^{joint}(P)
+T_{blocking}^{joint}(P)
+T_{interference}^{joint}(P)
+T_{repair}^{union}(P)
+T_{remaining}^{joint}(P),
\]

subject to HBM, bandwidth, stream, readiness, simultaneous-quality, and
economic constraints, including:

\[
T_{reuse,total}\le\gamma T_{dense-reference,total}.
\]

The planner may accept all, some, or none of the locked Segments, and may use
per-Segment staggered boundaries. It may not substitute a different historical
Source Variant. It never enumerates the Cartesian product of Source candidates.
A small offline joint Source-combination Oracle may be reported only as an
upper bound on the cost of this tractability decision.

## 10. Prefix-first and multi-Segment request path

The serving path remains:

1. Native engine exact Prefix Cache consumes all reusable complete prefix
   blocks.
2. The exact but incomplete prefix tail is computed dense.
3. All remaining exact non-prefix repeated canonical Segments are represented;
   there is no fixed Segment-count whitelist.
4. Each Segment has an explicit path: selected for joint planning, abstained,
   rejected, or dense.
5. Accepted Segments contribute repair tokens to an absolute-position union
   mask; prefix rows and the mandatory suffix never enter Source repair.
6. Policy A (`causal_commit_wait`) and policy C
   (`immediate_staggered_closed_loop`) remain explicit execution policies for
   matched ablation. Neither silently changes Source identity or selection.

## 11. v7 state and audit schema

### 11.1 Request-level fields

```text
protocol_version=7
request_plan_id and generation
canonicalizer_signature
model_math_signature
tokenizer_signature
scheduler_snapshot_id
shared_probe_ms
shared_metadata_ms
shared_summary_ms
shared_compare_batch_ms
shared_resource_cost_ms
joint_load_ms
joint_blocking_ms
joint_interference_ms
joint_union_repair_ms
joint_total_cost_ms
experiment_contract_compatible
gpu_runtime_correctness
execution_mode
```

### 11.2 Segment/Source fields

```text
segment_plan_id
segment_provenance_id
reuse_content_key
stored_k / source_variant_eligible_k / calibration_eligible_k / compared_k
source_variant_ineligibility_reasons
calibration_ineligibility_reasons
locked_source_variant_id
source_lock_layer and reason
predicted_access_plan_id
canonical_artifact_compatible / replica_preview_count
bound_artifact_id / bound_replica_id
replica_generation / placement_epoch
predicted_runtime_admission
refined_runtime_admission
actual_reuse_boundary
repair_ratio and union-mask contribution
joint_rejection_reason
```

### 11.3 Lifecycle and resource fields

```text
Source Variant, Artifact, Replica, and Locator events
selection lease and execution lease
copy source/destination replica IDs
transferred and wasted bytes
eviction/purge reason
preview stale reason and same-Source replan count
profile and scheduler snapshot versions
```

## 12. Explicitly rejected designs

v7 does not allow:

- dynamically replacing `alignment_quantum` with the current runtime block
  size;
- calling alignment mismatch a KV mathematical correctness failure;
- padding all logical RAG Segments to 512 tokens;
- cutting a high-value sentence solely to satisfy a hard block multiple;
- fuzzy Source lookup by approximately matching Segment length or text;
- putting `canonicalizer_signature` into the fundamental exact-token reuse key;
- putting GPU type, CUDA version, storage tier, or transient handles into the
  reuse-content identity;
- treating a short-prefix and long-prefix KV occurrence as one Source Variant;
- treating CPU/GPU/SSD copies as different historical Sources;
- using a CUDA pointer or block ID as a persistent Replica identity;
- checking a concrete Artifact digest before an Artifact has been bound;
- hiding Replica access cost from the Source selector;
- multiplying a shared probe by the number of candidates;
- charging only the winning candidate's comparison after comparing many;
- switching to `latest`, default, or second-ranked Source after Source lock;
- running independent final admissions for resource-coupled Segments;
- mixing lossy full-KV INT8 with H1/H2 Source-selection conclusions;
- silently reinterpreting a v6 config, artifact, or qualification gate as v7.

## 13. Implementation phases and gates

### Phase A: protocol and compatibility scaffolding

1. Add `protocol_version: 7` config and schema without changing v3-v6 loaders.
2. Split model math, tokenizer, runtime, Artifact, and Replica signatures.
3. Add frozen reason enums and migration readers for old HistoricalSource
   records.
4. Add contract validation that reports alignment mismatch as experiment
   incompatibility, not runtime correctness failure.

Gate: all existing 277 v6 tests remain green and new schema-only tests pass.

### Phase B: canonical semantic chunker

1. Implement deterministic normalization and token-boundary provenance.
2. Implement semantic candidates, soft alignment score, and tail penalty.
3. Persist canonicalizer signature, document revision, ordinal, token span,
   exact IDs, and reuse-content key.
4. Re-tokenize independently for Mistral and Qwen.

Gate: deterministic reruns are byte-identical; paragraph/sentence, alignment,
short-tail, Unicode, special-token, and tokenizer-isolation tests pass.

### Phase C: identity-aware Source Store

1. Add Source Variant, one canonical KV Artifact, Physical Replica, and locator objects.
2. Preserve global byte capacity, 1-16 variants, value-density eviction,
   probation, leases, and model namespace lifecycle.
3. Make cross-tier copy create/audit Replicas without changing Source Variant.
4. Add canonical/logical/bytes digest domains and enforce the lossless-only v7 policy.

Gate: identity collision, ABA, copy, relocation, eviction, purge, corruption,
and model-isolation tests pass.

### Phase D: layered eligibility and versioned access preview

1. Implement `SourceVariantEligibility` and `CalibrationEligibility`.
2. Implement canonical Artifact compatibility and Replica feasibility previews.
3. Add predicted access-plan snapshots and atomic compare-and-bind.
4. Add same-Source-only replan and dense fallback.

Gate: stale previews never bind silently; unrelated pool activity does not
invalidate a plan; Source fallback never changes the locked variant.

### Phase E: cost accounting and joint planner

1. Split request-level sunk costs from Source-contingent future costs.
2. Measure/profile comparison batches rather than multiplying isolated costs.
3. Add request-level Multi-Segment resource, union-repair, boundary, and
   admission planning while preserving per-Segment Source freeze.
4. Retain A/C execution policies and actual scheduler feedback.

Gate: no double charging; no Cartesian Source enumeration; partial/all/dense
plans obey resource, quality, and `gamma` constraints.

### Phase F: no-GPU artifacts and new freeze

1. Run compile, all unit/cross-module tests, config validation, both A/C local
   simulations, and `git diff --check`.
2. Regenerate Mistral and Qwen canonical manifests, 140-job qualifications,
   150-case pilot manifests, and 9,720-job H1 lists.
3. Freeze and push a new v7 SHA. The old `fea69c9` artifacts remain archived.
4. Produce a v7 no-GPU readiness gate with `gpu_runtime_qualified=false` and
   `h1_h2_execution_allowed=false`.

Gate: no failures, no locked-test access, no fake timing accepted.

### Phase G: A800 qualification before any full H1

For Mistral and then Qwen:

1. Verify frozen SHA, environment, `alignment_quantum=16`, and actual vLLM
   block size 16.
2. Run native Prefix Cache plus non-prefix `r=1` sentinel.
3. Run canary and all 140 runtime jobs.
4. Run one four-Source, nine-ratio H1 sentinel.
5. Generate protocol-v7, schema-v3 per-model and joint gates.

Only after both models pass may the system emit
`ready_for_full_h1_pilot=true`; it must not start the full pilot automatically.

## 14. Required tests

At minimum, v7 adds tests for:

- canonical chunking determinism and model-specific tokenizer isolation;
- semantic boundary winning over alignment when appropriate;
- aligned boundary winning when semantic scores are equivalent;
- short-tail rebalance without padding or token loss;
- identical tokens from different chunker versions sharing a reuse bucket but
  failing a mismatched formal calibration namespace;
- different historical prefix/position provenance producing different Source
  Variants in one content bucket;
- rejecting a second full-KV Artifact while allowing Summary features outside
  the Artifact count;
- one Artifact owning multiple tier Replicas;
- locator mutation without identity mutation;
- cross-tier copy, generation monotonicity, and ABA protection;
- source-state, logical-artifact, and serialized-byte digest corruption;
- layered eligibility reasons and dense fallbacks;
- preview staleness, atomic binding, selection lease, same-Source replan, and
  prohibition of Source reselection;
- shared probe charged once for K=1/4/16;
- all compared candidates included in batch comparison cost;
- 1/2/5/10/37 and arbitrary Segment counts with explicit execution ownership;
- joint HBM/bandwidth contention, partial reuse, staggered boundaries, and
  union-repair cost;
- v3-v6 config/artifact readability and inability to unlock v7 gates.

## 15. Evidence and stop rules

All v7 development, qualification, and H1 sentinel outputs remain:

```text
paper_evidence=false
locked_test_accessed=false
```

The original scientific stop rules remain binding:

- insufficient Source sensitivity stops the multi-Source line;
- a selector that cannot beat Cache-Craft CFO by the preregistered margin stops
  the current-state selection line;
- comparison/probe overhead exceeding the request-level 5% budget invalidates
  the main online policy;
- refined reuse that fails the default `gamma=0.8` economic gate executes dense;
- quality or tail protection failure cannot be repaired by loosening thresholds
  on the locked test set;
- multi-Segment benefit limited to controlled synthetic traces cannot support
  the main end-to-end claim.

## 16. Next action

The next action is Phase A, not GPU rental. Implement v7 on a separate branch,
preserve `main@fea69c9` as the v6 execution point until v7 local gates and
regenerated artifacts pass, and only then replace the server execution SHA.
