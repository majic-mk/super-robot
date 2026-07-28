# A800 Stage 2 runbook

Stage 2 is a non-paper server pilot.  It must use the exact ProbeKV Git SHA,
CacheBlend base commit, tracked patch hashes, Mistral revision and A800 UUID
recorded in the artifact directory.  Never place SSH credentials in a command,
configuration file, shell history, Git repository or artifact.

## Local freeze

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

Commit and push the clean branch before using the server.  The server checks
out that exact commit; it is not a development worktree.

## Server directories and patches

Use data-disk paths for the ProbeKV checkout, CacheBlend trees, Hugging Face
cache, environments and artifacts.

```bash
export PYTHONPATH="$PROBEKV_SRC/src"
bash scripts/server/prepare_cacheblend.sh "$STAGE2_ROOT/src/CacheBlend-cb0" cb0
python scripts/server/verify_cacheblend_patch.py \
  --cacheblend "$STAGE2_ROOT/src/CacheBlend-cb0" \
  --mode cb0 \
  --output "$ARTIFACTS/cb0_patch.json"

bash scripts/server/prepare_cacheblend.sh \
  "$STAGE2_ROOT/src/CacheBlend-probekv" probekv
python scripts/server/verify_cacheblend_patch.py \
  --cacheblend "$STAGE2_ROOT/src/CacheBlend-probekv" \
  --mode probekv \
  --output "$ARTIFACTS/probekv_patch.json"
```

Install the editable vLLM package from the `CacheBlend-probekv/vllm_blend`
tree before CB1-CB3 and H1.  Capture the environment after installation.

## Data freeze

Download only official train artifacts:

```bash
python scripts/server/download_h1_datasets.py \
  --registry configs/h1_official_datasets.json \
  --output "$STAGE2_ROOT/data/official"
```

For each dataset, run `scripts/prepare_rag_data.py` with the corresponding
`train_path`, official URL, repository revision and license from
`official_sources.json`.  Use:

- tokenizer `mistralai/Mistral-7B-Instruct-v0.3`;
- model revision `c170c708c41dac9275d15a8fff4eca08d52bab71`;
- construction `both`;
- seed `20260726`;
- no record limit.

Combine the three prepared `cases.jsonl` files:

```bash
python scripts/build_h1_pilot_manifest.py \
  --dataset-manifest "$MUSIQUE_CASES" \
  --dataset-manifest "$TWOWIKI_CASES" \
  --dataset-manifest "$HOTPOT_CASES" \
  --output "$ARTIFACTS/manifest" \
  --per-dataset 50 \
  --natural-target 25 \
  --seed 20260726 \
  --model-revision c170c708c41dac9275d15a8fff4eca08d52bab71
```

The command must produce exactly 150 pilot cases and report that locked test
data was not accessed.

## CB0-CB3 gate

Run the patched official example in the CB0 environment:

```bash
python scripts/server/run_cb0_patched.py \
  --cacheblend "$STAGE2_ROOT/src/CacheBlend-cb0" \
  --output "$ARTIFACTS/cb0"
```

Build and run the three-dataset CB1-CB3 grid:

```bash
python scripts/build_cb_gate_jobs.py \
  --manifest "$ARTIFACTS/manifest/h1_pilot_cases.jsonl" \
  --output "$ARTIFACTS/cb-gates"

python scripts/server/run_cacheblend_h1_pilot.py \
  --manifest "$ARTIFACTS/manifest/h1_pilot_cases.jsonl" \
  --jobs "$ARTIFACTS/cb-gates/cb_gate_jobs.jsonl" \
  --output "$ARTIFACTS/cb-gates/runtime" \
  --environment "$ARTIFACTS/environment.json" \
  --patch-provenance "$ARTIFACTS/probekv_patch.json" \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  --pass all \
  --max-hours 3

python scripts/server/validate_cb1_cb3.py \
  --jobs "$ARTIFACTS/cb-gates/cb_gate_jobs.jsonl" \
  --results "$ARTIFACTS/cb-gates/runtime/results-all.jsonl" \
  --output "$ARTIFACTS/cb-gates/stage_gate.json"
```

Do not start H1 unless CB0, CB1, CB2 and CB3 all pass.  In particular, every
`r=1` endpoint must have exact generated token IDs and logit relative-L2 at
most `1e-4`.

## H1 sessions

```bash
python scripts/build_e1_jobs.py \
  --manifest "$ARTIFACTS/manifest/h1_pilot_cases.jsonl" \
  --config configs/a800_h1_pilot.json \
  --output "$ARTIFACTS/h1/jobs" \
  --splits pilot \
  --anchor-fraction 0.20
```

Session 1 runs the primary layer after the CB gates, with its remaining
wall-clock budget.  Session 2 first resumes the primary pass, then runs anchors.
Use `--resume` whenever the corresponding result file already exists.

```bash
python scripts/server/run_cacheblend_h1_pilot.py \
  --manifest "$ARTIFACTS/manifest/h1_pilot_cases.jsonl" \
  --jobs "$ARTIFACTS/h1/jobs/jobs.jsonl" \
  --output "$ARTIFACTS/h1/runtime" \
  --environment "$ARTIFACTS/environment.json" \
  --patch-provenance "$ARTIFACTS/probekv_patch.json" \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  --pass primary \
  --max-hours 5

python scripts/server/run_cacheblend_h1_pilot.py \
  --manifest "$ARTIFACTS/manifest/h1_pilot_cases.jsonl" \
  --jobs "$ARTIFACTS/h1/jobs/jobs.jsonl" \
  --output "$ARTIFACTS/h1/runtime" \
  --environment "$ARTIFACTS/environment.json" \
  --patch-provenance "$ARTIFACTS/probekv_patch.json" \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  --pass primary \
  --max-hours 8 \
  --resume

python scripts/server/run_cacheblend_h1_pilot.py \
  --manifest "$ARTIFACTS/manifest/h1_pilot_cases.jsonl" \
  --jobs "$ARTIFACTS/h1/jobs/jobs.jsonl" \
  --output "$ARTIFACTS/h1/runtime" \
  --environment "$ARTIFACTS/environment.json" \
  --patch-provenance "$ARTIFACTS/probekv_patch.json" \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  --pass anchors \
  --max-hours 8
```

Merge and analyze without `--allow-test`:

```bash
python scripts/merge_e1_results.py \
  --jobs "$ARTIFACTS/h1/jobs/jobs.jsonl" \
  --result-dir "$ARTIFACTS/h1/runtime" \
  --output "$ARTIFACTS/h1/analysis" \
  --total-layers 32 \
  --bootstrap-iterations 10000
```

The primary H1 decision is valid when all 5,400 primary rows are accounted for.
The 4,320 anchor rows may remain explicitly pending after the second hard stop,
but must be completed before formal E1.  Release the cloud instance from the
provider console after copying the artifact directory.

