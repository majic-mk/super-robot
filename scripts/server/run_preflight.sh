#!/usr/bin/env bash
set -euo pipefail

mode="${1:-gpu}"
if [[ "$mode" != "gpu" && "$mode" != "no-gpu" ]]; then
  echo "usage: $0 [gpu|no-gpu]" >&2
  exit 2
fi

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$workspace"

export PYTHONPATH="$workspace/src${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_contract.py \
  --contract configs/experiment_contract.yaml \
  --output artifacts/server_preflight/contract.json
python -m probekv local-e1e2 \
  --config configs/local_e1e2.json \
  --output artifacts/server_preflight/local_e1e2 \
  --resume
python scripts/server/capture_environment.py \
  --output artifacts/server_preflight/environment.json
if [[ "$mode" == "gpu" ]]; then
  python scripts/server/verify_paper_environment.py \
    --contract configs/experiment_contract.yaml \
    --output artifacts/server_preflight/paper_environment.json
else
  echo "GPU environment gate skipped explicitly; use run_v6_no_gpu_preflight.sh for protocol v6."
fi
