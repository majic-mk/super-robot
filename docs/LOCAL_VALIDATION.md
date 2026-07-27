# Local validation boundary

Local validation proves software and small-model correctness. It does not
produce A100 timing claims.

## Automated checks

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m probekv validate-config --config configs/local_smoke.json
python -m probekv simulate --config configs/local_smoke.json
python scripts/audit_environment.py --output artifacts/local_validation/environment.json
```

For a locally cached Hugging Face causal LM:

```powershell
python scripts/local_model_h0.py `
  --model C:\path\to\snapshot `
  --output artifacts/local_validation/model_h0.json
```

The model check verifies:

- two equal-length but different prefixes produce different KV for the same
  exact segment, demonstrating why `S2` cannot be constructed from `S1`;
- canonical tensor save/load is bit-exact;
- probe-summary reads do not mutate the source;
- deterministic greedy generation repeats exactly.

## Current workstation finding

The RTX 5070 Ti Laptop GPU reports compute capability `sm_120`. The installed
PyTorch 2.4.1+cu121 exposes compiled architectures only through `sm_90`, so
GPU tensor execution is not valid in this environment. CPU model checks remain
valid. A separate modern environment can later be installed, but it must not
replace the pinned CacheBlend/A100 paper environment.

## Cannot be completed locally

- CacheBlend's pinned vLLM/CUDA performance reproduction;
- A100 TTFT, throughput, P95/P99, HBM and copy/compute interference;
- full 7B/8B dataset scans and paper gates H1-H5;
- cross-framework baselines whose official artifact needs the server stack.
