#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 /absolute/ProbeKV /absolute/data/root COMMIT /model/audit.json /patch/audit.json" >&2
  exit 2
fi

repo="$1"
data_root="$2"
expected_commit="$3"
model_audit="$4"
patch_audit="$5"
for path in "$repo" "$data_root" "$model_audit" "$patch_audit"; do
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
python scripts/server/build_v6_a800_jobs.py \
  --config configs/v6_a800_microbench.json \
  --contract configs/experiment_contract.yaml \
  --server-lock configs/a800_server_lock.json \
  --output "$output/jobs"
python scripts/server/verify_v6_no_gpu_readiness.py \
  --repo "$repo" \
  --data-root "$data_root" \
  --expected-commit "$expected_commit" \
  --server-lock configs/a800_server_lock.json \
  --config configs/v6_a800_microbench.json \
  --contract configs/experiment_contract.yaml \
  --jobs "$output/jobs/jobs.jsonl" \
  --job-manifest "$output/jobs/manifest.json" \
  --model-audit "$model_audit" \
  --patch-audit "$patch_audit" \
  --output "$output/readiness.json"

echo "No-GPU artifacts are ready for short A800 bring-up; the concrete engine hook and runtime qualification still block H1/H2."
