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

python scripts/local_reference_probe.py `
  --model C:\path\to\snapshot `
  --output artifacts/local_validation/reference_probe.json
```

The model check verifies:

- three mutually different prefixes `P`, `A` and `B`, with three different
  natural token lengths, are independently prefixed to the same exact segment
  `C`; `P|C` is the current
  request and only `A|C`, `B|C` are historical Sources;
- both historical Sources differ from the current state, and `A|C` differs
  from `B|C`, demonstrating why one Source cannot be copied from another;
- no prefix is truncated or padded to equal length; this exercises the same
  simultaneous context and position changes that Cache-Craft handles with
  position correction;
- current-to-current self-comparison is allowed only as a zero-drift hook
  sanity check and is never presented to the Source selector;
- canonical tensor save/load is bit-exact;
- probe-summary reads do not mutate the source;
- deterministic greedy generation repeats exactly.
- the reference backend captures pre-RoPE K, V, hidden and query projections
  at every layer through the 25% probe ceiling.

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
