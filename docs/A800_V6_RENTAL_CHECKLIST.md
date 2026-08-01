# A800 v6 rental-readiness checklist

This checklist separates three claims that must never be conflated:

1. **Artifact-ready without a GPU**: the exact code, Python stack, model
   snapshot, CacheBlend patch tree and 140-job manifest are present and bound
   by hashes.
2. **Ready only for a short bring-up rental**: item 1 passed, so CUDA build and
   integration failures can be diagnosed without risking an H1/H2 run.
3. **Ready to rent for qualification**: the concrete layer-resumable engine
   hook exists in source and the immutable 140-job worker can exercise it.
4. **Ready for H1/H2 or paper timing**: the concrete layer-resumable
   CacheBlend/vLLM engine has passed all 140 jobs, `r=1` dense equivalence and
   CUDA timing gates.

Passing item 1 does not imply item 3 or 4. At the current repository revision the
v6 contracts, controller, adapters and CacheBlend multi-region mask patch are
implemented, but the concrete pinned-vLLM layer-resumable engine hook is still
missing. This is a source-implementation blocker that an A800 alone does not
solve; after implementation it must also be qualified on A800. The hard gate deliberately reports
`h1_h2_execution_allowed: false` until a real A800 audit proves otherwise.

## Frozen inputs

- Python 3.10; PyTorch 2.2.1 + CUDA 12.1; xformers 0.0.25; vLLM 0.4.1.
  The previously successful A800 versions of NumPy, Transformers, Tokenizers,
  Hugging Face Hub, Ray, CMake and Ninja are also pinned in
  `requirements/server-tools.txt`; do not replace them with latest versions.
- CacheBlend commit
  `b72d7945e6d6306f12be66520196e0f081fa2b0c` with patch mode
  `probekv_v6_multiregion`.
- `mistralai/Mistral-7B-Instruct-v0.3` revision
  `c170c708c41dac9275d15a8fff4eca08d52bab71`.
- One A800 80 GB, compute capability 8.0, at least 110 GiB host RAM, 16 CPU
  cores and 250 GiB free on the data disk.
- The machine-readable source of truth is
  `configs/a800_server_lock.json`.

## No-GPU preparation

Use only data-disk paths. Do not put credentials in commands, scripts, Git or
artifact JSON.

```bash
export PROBEKV_SRC=/data/src/ProbeKV
export STAGE_ROOT=/data/probekv-stage
export FROZEN_SHA=<commit-pushed-to-GitHub>

git -C "$PROBEKV_SRC" fetch origin
git -C "$PROBEKV_SRC" checkout --detach "$FROZEN_SHA"
bash "$PROBEKV_SRC/scripts/server/setup_a800_env.sh" \
  "$PROBEKV_SRC" "$STAGE_ROOT"
source "$STAGE_ROOT/envs/probekv-py310/bin/activate"

export HF_HOME="$STAGE_ROOT/hf"
export PYTHONPATH="$PROBEKV_SRC/src"
python "$PROBEKV_SRC/scripts/server/download_model_snapshot.py" \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  --cache-dir "$STAGE_ROOT/hf" \
  --output "$STAGE_ROOT/artifacts/v6_setup/model_audit.json"

bash "$PROBEKV_SRC/scripts/server/run_v6_no_gpu_preflight.sh" \
  "$PROBEKV_SRC" \
  "$STAGE_ROOT" \
  "$FROZEN_SHA" \
  "$STAGE_ROOT/artifacts/v6_setup/model_audit.json" \
  "$STAGE_ROOT/artifacts/v6_setup/cacheblend_patch.json"
```

The last command must finish with `artifact_preparation_ready: true`. At the
current revision it may report `gpu_rental_ready_for_runtime_bringup: true`,
but intentionally keeps `gpu_rental_ready_for_runtime_qualification`,
`gpu_runtime_qualified` and `h1_h2_execution_allowed` false until the concrete
engine hook exists. Therefore a rental at this point is suitable only for a
short bring-up/debug session, not for H1/H2 production.

## First A800 session: qualification only

First run the existing hardware/stack gate:

```bash
cd "$PROBEKV_SRC"
python scripts/server/verify_paper_environment.py \
  --contract configs/experiment_contract.yaml \
  --output "$STAGE_ROOT/artifacts/v6_a800/hardware_stack.json"
```

The concrete worker must then consume the frozen
`v6_no_gpu_preflight/jobs/jobs.jsonl` and produce a runtime audit bound to the
same `code_commit`, `job_digest`, model revision and CacheBlend patch hash. It
must prove all capabilities listed in `a800_server_lock.json`, complete all
140 jobs, preserve canonical Source digests, and pass:

```text
r=1 generated token IDs == dense generated token IDs
max first-32-token teacher-forced logit relative-L2 <= 1e-4
```

Validate that audit with:

```bash
python scripts/server/verify_v6_runtime_qualification.py \
  --server-lock configs/a800_server_lock.json \
  --job-manifest "$STAGE_ROOT/artifacts/v6_no_gpu_preflight/jobs/manifest.json" \
  --runtime-audit "$STAGE_ROOT/artifacts/v6_a800/runtime_audit.json" \
  --output "$STAGE_ROOT/artifacts/v6_a800/runtime_gate.json"
```

Only `gpu_runtime_qualified: true` and `h1_h2_execution_allowed: true` authorize
H1/H2. A contract-only adapter, fake runtime, partial 140-job run, host timer,
different SHA, different model or different patch is rejected.

## Stop conditions

- Stop before GPU work if the repository is dirty or any hash differs.
- Stop after environment build if vLLM is not exactly 0.4.1.
- Stop H1/H2 if the concrete engine hook is absent or any one of the 140 jobs
  fails.
- Never relabel local simulation, CB1-CB3 or qualification timing as paper
  evidence.
