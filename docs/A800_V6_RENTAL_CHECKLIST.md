# A800 v6 dual-model rental-readiness checklist

Four states must remain separate:

1. CPU artifacts are frozen and hash-bound.
2. Concrete Mistral and Qwen runtime sources exist.
3. Renting an A800 for runtime qualification is allowed.
4. H1/H2 is allowed only after real CUDA qualification.

The repository now implements the concrete source hooks under patch mode
`probekv_v6_staggered_runtime`. This is not a claim that they already work on
an A800. Before GPU execution, the final gate must report:

```text
artifact_preparation_ready = true
mistral_runtime_source_ready = true
qwen_runtime_source_ready = true
gpu_rental_ready_for_runtime_qualification = true
gpu_runtime_qualified = false
h1_h2_execution_allowed = false
failures = []
```

## Frozen inputs

- Python 3.10, PyTorch 2.2.1+cu121, xformers 0.0.25, vLLM 0.4.1.
- CacheBlend commit `b72d7945e6d6306f12be66520196e0f081fa2b0c`.
- Mistral revision `c170c708c41dac9275d15a8fff4eca08d52bab71`.
- Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Mistral adapter `mistral_cacheblend_llama_v041`.
- Qwen adapter `qwen2_5_vllm041`.
- One A800 80GB, compute capability 8.0, 16 CPU cores and 110GiB RAM.
- Transformers remains 4.40.2; no silent stack upgrade is allowed.

## 130GB storage policy

The old 250GiB single-disk gate is removed. The server must have at least:

- 70GiB free across unique writable filesystems;
- 50GiB free on the largest writable filesystem;
- 15GiB free on the system filesystem.

At 90GiB or more free, retain both selective snapshots. At 70-89GiB, qualify
Mistral first, preserve audits/results, purge only its verified regenerable HF
snapshot, then download Qwen in CPU-only mode. Below 70GiB, stop before renting
a GPU and do not delete user files.

The Mistral downloader excludes `consolidated.safetensors`; only the three HF
weight shards are allowed. Qwen uses only its four weight shards. One venv and
one final CacheBlend tree are retained, pip caching is disabled, build objects
are removed after installation, datasets stream, and full-dataset KV is never
stored.

The 70/50GiB admission thresholds are evaluated and archived before model
download. After download, a separate steady-state audit enforces the 15GiB
system-filesystem reserve; it does not reapply the pre-download thresholds to
space intentionally occupied by the frozen snapshots.

## CPU-only preparation

```bash
export PROBEKV_SRC=/data/src/ProbeKV
export STAGE_ROOT=/data/probekv-stage
export FROZEN_SHA=<pushed-clean-commit>

git -C "$PROBEKV_SRC" fetch origin
git -C "$PROBEKV_SRC" checkout --detach "$FROZEN_SHA"
bash "$PROBEKV_SRC/scripts/server/setup_a800_env.sh" \
  "$PROBEKV_SRC" "$STAGE_ROOT"
source "$STAGE_ROOT/envs/probekv-py310/bin/activate"

# A reused server may select one already verified Python 3.10 environment:
# export PROBEKV_PYTHON_BIN=/absolute/stage/envs/existing/bin/python
# export PROBEKV_ENV_DIR=/absolute/stage/envs/existing
# export PROBEKV_NVCC_BIN=/usr/local/cuda/bin/nvcc

# Use `both` when storage.json selects dual_model_resident. In sequential mode
# use `mistral` now and `qwen` after Mistral qualification and verified purge.
bash "$PROBEKV_SRC/scripts/server/prepare_dual_model_snapshots.sh" \
  "$PROBEKV_SRC" "$STAGE_ROOT" both

bash "$PROBEKV_SRC/scripts/server/run_v6_no_gpu_preflight.sh" \
  "$PROBEKV_SRC" "$STAGE_ROOT" "$FROZEN_SHA" \
  "$STAGE_ROOT/artifacts/model_audits/model_audit_mistral.json" \
  "$STAGE_ROOT/artifacts/model_audits/model_audit_qwen.json" \
  "$STAGE_ROOT/artifacts/v6_setup/cacheblend_patch.json"
```

`PROBEKV_ENV_DIR` is accepted only below the selected stage's `envs/`
directory. This avoids keeping a second multi-gigabyte environment solely for
renaming consistency.

The preflight compiles sources, runs all tests, validates the experiment
contract, runs both A and C local v6 configurations, audits storage and runtime
sources, and writes:

```text
jobs_mistral/jobs_mistral.jsonl
jobs_mistral/manifest_mistral.json
jobs_qwen/jobs_qwen.jsonl
jobs_qwen/manifest_qwen.json
readiness.json
```

Each matrix contains 140 non-paper qualification jobs and binds the ProbeKV
commit, CacheBlend base/patch/tree, model revision, tokenizer hash,
config/contract/server-lock hashes, adapter and job digest.

## GPU sequence

First rent: Mistral only, at most four hours. Run hardware/stack gate, then A
and C `1 Segment, K=1, r=1` sentinels, then all 140 Mistral jobs. Stop on any
token, first-32-logit, RoPE, mask, Source digest or CUDA-event failure.

Second rent: Qwen only after the Mistral runtime passed and the Qwen handoff is
complete. Run dense smoke, A/C r=1 sentinels and all 140 Qwen jobs. Also run a
non-zero native Prefix Cache sentinel and verify that every layer's pre-RoPE
prefix shadow is present, read-only and absolute-position aligned. Only a full
pass may unlock Qwen H1/H2.

The runtime audit must prove:

```text
r=1 generated token IDs == dense generated token IDs
max first-32-token teacher-forced logit relative-L2 <= 1e-4
canonical Source digests unchanged
all 140 jobs completed, failed = 0
```

Validate the audit with `scripts/server/verify_v6_runtime_qualification.py`.
Fake timing, a different SHA, partial jobs, a different model/adapter or a
different CacheBlend tree is rejected. Qualification remains
`paper_evidence:false`; formal matched-stack measurements start only afterward.
