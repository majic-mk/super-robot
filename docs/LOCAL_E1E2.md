# Local E1/E2 closed-loop validation

## Scope

This workflow validates experiment plumbing before formal server allocation. It covers
group-isolated manifests, repair-grid labeling, per-layer probe observations,
train/calibration/test discipline, quantile budget prediction, conformal cost
intervals, dynamic early exit, abstention, audit artifacts and resume.

The included fixture backend uses synthetic labels and timings. Its outputs are
always marked `paper_evidence: false`; H1 and H2 can only be accepted using real
model and CacheBlend measurements on the locked data.

## Run

```powershell
$env:PYTHONPATH = "src"
python -m probekv local-e1e2 `
  --config configs/local_e1e2.json `
  --output artifacts/local_e1e2 `
  --resume
```

The main 32-layer policy checks every layer from 1 through 8. The original
`{1,2,4,6,8}` policy remains an explicit sparse-checkpoint ablation.

## Artifacts

- `manifest.json` and `case_manifest.jsonl`: group split and token-hash audit.
- `ratio_measurements.jsonl`: every source, reuse layer and repair ratio.
- `safe_budget_labels.jsonl`: monotonic-envelope `r_safe` labels.
- `probe_observations.jsonl`: per-layer K/V/hidden/query and metadata features.
- `calibration_report.json`: fit counts, corrections and test-lock assertion.
- `decisions.jsonl`: candidate intervals, selection or abstention and regret.
- `ledger.json`: stage fingerprints used for deterministic resume.
- `summary.json`: software-path diagnostics, never a paper gate result.

Optional reporting smoke test:

```powershell
python scripts/plot_local_e1e2.py `
  --input artifacts/local_e1e2 `
  --output artifacts/local_e1e2/figures
```

## Normalized real-data input

Dataset-specific preparation must produce JSONL rows matching
`examples/manifest_input.example.jsonl`. `segment_token_ids` must come from the
exact target model tokenizer and revision. Build and validate the manifest with:

```powershell
python -m probekv build-manifest `
  --input data/normalized/cases.jsonl `
  --output artifacts/manifests/cases.jsonl `
  --model-signature "model@revision"
```

The builder hashes token IDs rather than display text and rejects cross-split
group leakage. Raw datasets and generated manifests remain outside Git.

## Hugging Face reference state

`HuggingFaceReferenceStateBackend` performs slow canonical full-prefill and
extracts segment K, V, hidden and query-projection tensors. It is intended for
small-model correctness checks. It intentionally does not implement selective
repair and cannot produce CacheBlend performance evidence.

The reference backend captures K before RoPE for source comparison while also
retaining the exact cached K/V tensors. It rejects separately tokenized
prefix/segment pairs when their concatenation is not tokenizer-boundary stable.
