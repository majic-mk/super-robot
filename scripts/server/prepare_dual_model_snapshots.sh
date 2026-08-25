#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 /absolute/ProbeKV /absolute/stage/root both|mistral|qwen" >&2
  exit 2
fi
repo="$1"
stage_root="$2"
phase="$3"
case "$repo" in /*) ;; *) echo "ProbeKV path must be absolute" >&2; exit 2;; esac
case "$stage_root" in /*) ;; *) echo "stage root must be absolute" >&2; exit 2;; esac
case "$stage_root" in /|/root|/home) echo "stage root is too broad" >&2; exit 2;; esac
case "$phase" in both|mistral|qwen) ;; *) echo "invalid phase" >&2; exit 2;; esac

python="${PROBEKV_PYTHON:-$stage_root/envs/probekv-py310/bin/python}"
artifacts="$stage_root/artifacts/model_audits"
mkdir -p "$artifacts"
export HF_HOME="$stage_root/hf"
export HUGGINGFACE_HUB_CACHE="$stage_root/hf/hub"
export TRANSFORMERS_CACHE="$stage_root/hf/transformers"
export PIP_NO_CACHE_DIR=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

"$python" "$repo/scripts/server/plan_server_storage.py" \
  --stage-root "$stage_root" --system-root / \
  --output "$artifacts/storage.json"

download_mistral() {
  "$python" "$repo/scripts/server/download_model_snapshot.py" \
    --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
    --cache-dir "$stage_root/hf" \
    --output "$artifacts/model_audit_mistral.json"
}
download_qwen() {
  "$python" "$repo/scripts/server/download_model_snapshot.py" \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --cache-dir "$stage_root/hf" \
    --output "$artifacts/model_audit_qwen.json"
}

if [[ "$phase" == "both" || "$phase" == "mistral" ]]; then download_mistral; fi
if [[ "$phase" == "both" || "$phase" == "qwen" ]]; then download_qwen; fi

echo "Selective model snapshot phase completed: $phase"
