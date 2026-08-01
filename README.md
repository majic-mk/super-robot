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
request-level 5% budget, locks Sources independently during the shared early
pass, then uses actual multi-source scheduler feedback to choose one common
reuse boundary and an economical subset of segments. v3-v5 remain unchanged.
The real CacheBlend path is split into an H1-only case runner and a
capability-gated online adapter; the case runner cannot silently satisfy the
online closed-loop contract.

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
python -m probekv local-e1e2 --config configs/local_e1e2.json --resume
```

Simulation artifacts are always marked `paper_evidence: false`. Formal timing
claims require the pinned CacheBlend stack on the config-frozen A800.

## Key files

- `configs/experiment_contract.yaml`: frozen research contract and all gates.
- `docs/UNIFIED_COST_ACCOUNTING.md`: v5 shared-cost and Source-lock protocol.
- `docs/V6_MULTI_SEGMENT.md`: v6 global-pool, multi-segment and common-boundary contract.
- `docs/ARCHITECTURE.md`: end-to-end system explanation.
- `docs/NOVELTY_AUDIT.md`: frozen claim boundary, prior-art matrix and novelty gates.
- `docs/NOVELTY_AUDIT_SOURCES.tsv`: machine-readable primary-source audit index.
- `docs/LOCAL_VALIDATION.md`: commands and local/formal-server boundary.
- `docs/A100_RUNBOOK.md`: formal-server timing and evidence collection rules
  (legacy filename retained for stable links).
- `docs/A800_STAGE2_RUNBOOK.md`: exact CB0-CB3 and 150-case H1 pilot commands.
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
