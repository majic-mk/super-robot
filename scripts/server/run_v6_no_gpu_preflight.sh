#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 /absolute/ProbeKV /absolute/data/root COMMIT /mistral/audit.json /qwen/audit.json /patch/audit.json" >&2
  exit 2
fi

repo="$1"
data_root="$2"
expected_commit="$3"
mistral_audit="$4"
qwen_audit="$5"
patch_audit="$6"
for path in "$repo" "$data_root" "$mistral_audit" "$qwen_audit" "$patch_audit"; do
  case "$path" in /*) ;; *) echo "all paths must be absolute: $path" >&2; exit 2;; esac
done

cd "$repo"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "expected ProbeKV $expected_commit, found $actual_commit" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "no-GPU handoff requires a clean ProbeKV worktree" >&2
  exit 1
fi

output="$data_root/artifacts/v6_no_gpu_preflight"
mkdir -p "$output"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_contract.py \
  --contract configs/experiment_contract.yaml \
  --output "$output/contract.json"
python -m probekv.cli \
  --config configs/local_system_v6.json \
  --output "$output/local_v6"
python -m probekv.cli \
  --config configs/local_system_v6_immediate_staggered.json \
  --output "$output/local_v6_immediate_staggered"
python scripts/server/plan_server_storage.py \
  --stage-root "$data_root" --system-root / \
  --output "$output/storage.json"
python scripts/server/audit_v6_runtime_sources.py \
  --repo "$repo" --output "$output/runtime_sources.json"
python scripts/server/build_v6_a800_jobs.py \
  --config configs/v6_a800_microbench.json \
  --contract configs/experiment_contract.yaml \
  --server-lock configs/a800_server_lock.json \
  --model-key mistral --model-audit "$mistral_audit" \
  --patch-audit "$patch_audit" --output "$output/jobs_mistral"
python scripts/server/build_v6_a800_jobs.py \
  --config configs/v6_a800_microbench.json \
  --contract configs/experiment_contract.yaml \
  --server-lock configs/a800_server_lock.json \
  --model-key qwen --model-audit "$qwen_audit" \
  --patch-audit "$patch_audit" --output "$output/jobs_qwen"
python scripts/server/verify_v6_dual_model_no_gpu_readiness.py \
  --repo "$repo" \
  --data-root "$data_root" \
  --expected-commit "$expected_commit" \
  --server-lock configs/a800_server_lock.json \
  --config configs/v6_a800_microbench.json \
  --contract configs/experiment_contract.yaml \
  --storage-audit "$output/storage.json" \
  --runtime-source-audit "$output/runtime_sources.json" \
  --mistral-jobs "$output/jobs_mistral/jobs_mistral.jsonl" \
  --mistral-manifest "$output/jobs_mistral/manifest_mistral.json" \
  --mistral-model-audit "$mistral_audit" \
  --qwen-jobs "$output/jobs_qwen/jobs_qwen.jsonl" \
  --qwen-manifest "$output/jobs_qwen/manifest_qwen.json" \
  --qwen-model-audit "$qwen_audit" \
  --patch-audit "$patch_audit" \
  --output "$output/readiness.json"

echo "Dual-model sources and artifacts are ready for A800 runtime qualification; H1/H2 remain blocked until the real GPU qualification passes."
