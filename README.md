# ProbeKV

ProbeKV is a research harness for **current-state-guided historical KV source
selection**. When the same non-prefix segment has several canonical historical
KV versions, early fresh state from the current request selects the source with
the lowest conservative downstream repair cost.

This repository currently implements the complete local validation layer:
source invariants, RoPE round-trip, safe-ratio labeling, dynamic `L_probe`,
conformal upper budgets, dynamic `L_reuse`, total-cost admission, P0-P4/Dynamic
prefetch, A-only/B-only/Hybrid scheduling, grouped statistics, gates, audit
output, deterministic simulation, and an offline Hugging Face model H0 check.

## Quick start

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m probekv validate-config --config configs/local_smoke.json
python -m probekv simulate --config configs/local_smoke.json
python -m probekv local-e1e2 --config configs/local_e1e2.json --resume
```

Simulation artifacts are always marked `paper_evidence: false`. Formal timing
claims require the pinned CacheBlend stack on A100.

## Key files

- `configs/experiment_contract.yaml`: frozen research contract and all gates.
- `docs/ARCHITECTURE.md`: end-to-end system explanation.
- `docs/LOCAL_VALIDATION.md`: commands and local/A100 boundary.
- `docs/A100_RUNBOOK.md`: timing and evidence collection rules.
- `docs/LOCAL_E1E2.md`: complete local E1/E2 plumbing and artifacts.
- `docs/RAG_DATA.md`: real-dataset normalization and Source construction rules.
- `docs/E1_JOBS.md`: deterministic repair-grid sharding and result audit.
- `docs/CACHEBLEND_INTEGRATION.md`: the remaining pinned-runtime shim contract.
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
4. On the A100 server, fetch and checkout that exact SHA.
5. Keep datasets, model weights, papers, credentials and raw experiment output
   outside Git. Every result row already records the code and environment hash.

Before an A100 paper run:

```bash
python scripts/server/verify_a100_environment.py
git rev-parse HEAD
git status --short
```
