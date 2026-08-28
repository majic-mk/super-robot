# ProbeKV

ProbeKV is a research harness for **target-model current-early-state-guided,
variant-specific safe-cost selection**. When the same exact non-prefix segment
has several canonical historical KV variants, fresh early state from the
current request selects only a conservatively safe and economically useful
source; uncertain decisions abstain and fall back to full recomputation.
Legacy strict abstention remains reproducible, while new final-layer selector
policies and bounded-overrun scheduling must be enabled explicitly.
Protocol v4 additionally enforces the runtime order
`selection -> load/schedule -> refined cost -> final admission`; protocol v3
remains available only for reproduction. Protocol v5 keeps checkpoint-level
early Source selection and makes preliminary prediction and refined admission
use one request-arrival-to-first-token component cost identity. Once selected,
the Source is locked; refinement can only accept reuse or fall back to full.
Protocol v6 adds a global 1-16-variant pool and plans every exact non-prefix
repeated segment in one request. It compares lightweight summaries within a
request-level 5% budget and locks Sources independently during the shared early
pass. The current main policy is A (`causal_commit_wait`); C
(`immediate_staggered_closed_loop`) is retained as an explicit execution-matched
ablation. The removed shadow-dense policy is not accepted by configuration.
Both A and C use actual multi-source scheduler feedback and per-segment refined
boundaries. The old common-boundary configuration remains separately available
for reproduction. v3-v5 remain unchanged.
The legacy complete-generate H1 case runner is retained only for CB1-CB3 and
old-protocol reproduction; it cannot produce protocol-v6 H1 labels. The v6 H1
worker uses the same layer-resumable executor as the capability-gated online
adapter and hard-stops when any Source has `r=1 != dense`. Patch mode
`probekv_v6_staggered_runtime` remains the frozen legacy v6 runtime. The
explicit `probekv_v6_prefix_hardened_runtime` mode appends native Prefix Cache
shadow support without changing that legacy protocol, and adds concrete
layer-resumable hooks for the Mistral `llama.py` path and the
Qwen2 `qwen2.py` path. Both adapters passed their frozen 140-job A800 matrices
at commit `6618068`; any later code commit and the new prefix-hardened patch
must be requalified before H1. H1 now requires a schema-v2 qualification gate
bound to the exact code, model, patch/tree, 140-job audit, GPU UUID and native
Prefix Cache block-metadata sentinel.

Protocol v7 is a separately versioned path. It adds deterministic
semantic-block canonical Segments, exact model/tokenizer-scoped content
buckets, historical Source Variant identity, exactly one lossless BF16
pre-RoPE KV Artifact per Variant, and versioned GPU/pinned-CPU/SSD Replicas.
Source selection reads lightweight Summaries, not full-KV Artifacts. A locked
Source may replan among Replicas of its Artifact but may never switch Variant.
The request-level joint planner uses per-Segment staggered boundaries and may
accept all, some, or none of the locked Segments. v6 gates cannot authorize v7.

Protocol v8 is another explicit path and does not reinterpret v7 results. It
uses training-free current-state Residual-K Source selection instead of a
learned/calibrated selector. CFO only orders budgeted comparisons, online
CacheBlend repair is fixed at 15%, and early Source lock remains enabled at
completed-depth checkpoints. Selection reads exact BF16 K-only states through
bounded GPU scratch and never transfers full-KV Artifacts to compare candidates.
One logical Artifact may move through SSD, pinned CPU and GPU Replicas without
becoming three permanent copies. Profile freeze must precede Profile-bound
140-job A800 qualification; v7 and v8 Gates cannot authorize each other.

Protocol v8 schema-v6 is an explicit runtime successor while schema-v5 remains
readable. It replaces per-Segment additive Gate2/Gate3 timing with one joint
request critical path, separates selection/admission/preparation/commit state,
permits resource-safe frozen-winner preparation while Gate2 is deferred, and
returns partial ready-subset Gate3 decisions. Full-KV transfer always requires
a physical Replica lease plus a unified HBM reservation.

Mistral-7B-Instruct-v0.3 is the CacheBlend qualification and secondary model;
Qwen2.5-7B-Instruct is the formal primary model. Llama 3.1 is deferred until
the Qwen end-to-end gain is at least 5%.

This repository implements the complete local validation layer:
source invariants, RoPE round-trip, safe-ratio labeling, dynamic `L_probe`,
conformal upper budgets, dynamic `L_reuse`, total-cost admission, P0-P4/Dynamic
prefetch, strict/bounded atomic scheduling, grouped statistics, gates, audit
output, deterministic simulation, and an offline Hugging Face model H0 check.
It also contains the audited CacheBlend patchset, A800 CB0-CB3 gate runner,
official-data pilot builder and resumable non-paper H1 server worker. Server
measurements are not considered complete until their archived gate files pass.

## Quick start

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m probekv validate-config --config configs/local_smoke.json
python -m probekv simulate --config configs/local_smoke.json
python -m probekv simulate --config configs/local_system_v3.json
python -m probekv simulate --config configs/local_system_v4.json
python -m probekv simulate --config configs/local_system_v5.json
python -m probekv.cli --config configs/local_system_v6.json
python -m probekv.cli --config configs/local_system_v7_causal_wait.json
python -m probekv.cli --config configs/local_system_v7_immediate_staggered.json
python -m probekv.cli --config configs/local_system_v8_causal_wait.json
python -m probekv.cli --config configs/local_system_v8_immediate_staggered.json
python -m probekv.cli --config configs/local_system_v8_schema6_causal_wait.json
python -m probekv.cli --config configs/local_system_v8_schema6_immediate_staggered.json
python -m probekv local-e1e2 --config configs/local_e1e2.json --resume
```

Simulation artifacts are always marked `paper_evidence: false`. Formal timing
claims require the pinned CacheBlend stack on the config-frozen A800.

## Key files

- `configs/experiment_contract.yaml`: frozen research contract and all gates.
- `docs/UNIFIED_COST_ACCOUNTING.md`: v5 shared-cost and Source-lock protocol.
- `docs/V6_MULTI_SEGMENT.md`: v6 global-pool, arbitrary-count multi-segment and A/C staggered contract.
- `docs/PROBEKV_V7_NEXT_PHASE_PLAN.md`: v7 canonical-Segment, layered identity,
  single-Artifact/multi-Replica, versioned access-plan and request-level
  joint-planning contract. Its local implementation is testable; A800 evidence
  remains unqualified until both schema-v3 model gates pass.
- `docs/A800_V7_RUNBOOK.md`: immutable no-GPU handoff, two-model A800
  qualification and one-case H1 sentinel procedure.
- `docs/V7_IMPLEMENTATION_STATUS.md`: explicit local-complete versus
  server/GPU-pending boundary; never treat it as GPU evidence.
- `docs/PROBEKV_V8_PROTOCOL.md`: frozen training-free Residual-K, K-state,
  lease and two-stage Planner protocol.
- `docs/A800_V8_RUNBOOK.md`: Profile-before-qualification server sequence.
- `docs/PROBEKV_V8_SCHEMA6_PROTOCOL.md`: joint-timeline, orthogonal-state,
  speculative-winner and subset-Gate3 runtime contract.
- `docs/A800_V8_SCHEMA6_SENTINEL_RUNBOOK.md`: first four-hour Mistral sentinel.
- `docs/V8_IMPLEMENTATION_STATUS.md`: v8 local-complete/GPU-pending boundary.
- `docs/ARCHITECTURE.md`: end-to-end system explanation.
- `docs/NOVELTY_AUDIT.md`: frozen claim boundary, prior-art matrix and novelty gates.
- `docs/NOVELTY_AUDIT_SOURCES.tsv`: machine-readable primary-source audit index.
- `docs/LOCAL_VALIDATION.md`: commands and local/formal-server boundary.
- `docs/A100_RUNBOOK.md`: formal-server timing and evidence collection rules
  (legacy filename retained for stable links).
- `docs/A800_STAGE2_RUNBOOK.md`: exact CB0-CB3 and 150-case H1 pilot commands.
- `docs/A800_V6_RENTAL_CHECKLIST.md`: final no-GPU artifact gate, pinned server
  installation, model audit and mandatory A800 runtime-qualification boundary.
- `docs/LOCAL_E1E2.md`: complete local E1/E2 plumbing and artifacts.
- `docs/RAG_DATA.md`: real-dataset normalization and Source construction rules.
- `docs/E1_JOBS.md`: deterministic repair-grid sharding and result audit.
- `docs/CACHEBLEND_INTEGRATION.md`: the pinned-runtime shim contract and gates.
- `docs/TWO_STAGE_CLOSURE.md`: v4 state machine, timing feedback and Source lifecycle.
- `src/probekv/backend.py`: integration boundary for CacheBlend or SparseX.
- `tests/`: executable invariants and decision-policy tests.

## Important statistical correction

With only 200 locked cases per dataset, zero tail violations still cannot prove
a one-sided 95% upper bound of at most 1% on that dataset. The frozen contract
pre-registers the H3 tail gate on 600 pooled RAG cases per model, while retaining
per-dataset descriptive results. Increasing each dataset to at least 299 cases
is the alternative.

## Version-control workflow

The GitHub repository is the source of truth; experiment servers run an exact
commit and should not contain uncommitted code edits.

1. Develop and test on a feature branch.
2. Push the branch and let the CPU correctness workflow pass.
3. Record the selected commit SHA in the frozen experiment configuration.
4. On the formal A800 server, fetch and checkout that exact SHA.
5. Keep datasets, model weights, papers, credentials and raw experiment output
   outside Git. Every result row already records the code and environment hash.

Before a formal paper run:

```bash
python scripts/server/verify_paper_environment.py \
  --contract configs/experiment_contract.yaml
git rev-parse HEAD
git status --short
```

For protocol v6, passing the generic environment check is not sufficient.
`scripts/server/verify_v6_runtime_qualification.py` must also report both
`gpu_runtime_qualified: true` and `h1_h2_execution_allowed: true`. The concrete
schema-v2 gate additionally requires `native_prefix_cache_qualified: true` and
hashes of the job manifest, runtime audit and Prefix Cache audit. The concrete
dual-model source hooks are implemented and statically audited. Model-specific
H1 manifests must be rebuilt with `prepare_v6_h1_model_data.py`; Mistral token
IDs and content hashes must never be reused by Qwen.

For protocol v7, use `configs/a800_server_lock_v7.json`, patch mode
`probekv_v7_single_artifact_runtime`, the v7 job/readiness scripts, and a
schema-v3 qualification gate. A v6 gate is deliberately rejected. Full H1 is
not started by the qualification scripts.
