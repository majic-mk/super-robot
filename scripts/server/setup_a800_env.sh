#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/path/to/ProbeKV /absolute/data/root" >&2
  exit 2
fi

repo="$1"
stage_root="$2"
case "$repo" in /*) ;; *) echo "ProbeKV path must be absolute" >&2; exit 2;; esac
case "$stage_root" in /*) ;; *) echo "data root must be absolute" >&2; exit 2;; esac
case "$stage_root" in /|/root|/home) echo "data root is too broad" >&2; exit 2;; esac
if [[ ! -d "$repo/.git" ]]; then
  echo "ProbeKV path is not a Git checkout: $repo" >&2
  exit 1
fi

python_bin="${PROBEKV_PYTHON_BIN:-python3.10}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python 3.10 is required; set PROBEKV_PYTHON_BIN if necessary" >&2
  exit 1
fi
python_version="$($python_bin -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$python_version" != "3.10" ]]; then
  echo "expected Python 3.10, found $python_version" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -q 'release 12\.1'; then
  echo "CUDA toolkit 12.1 with nvcc is required before building vLLM" >&2
  exit 1
fi

mkdir -p "$stage_root"/{src,envs,hf,build,artifacts}
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
python "$repo/scripts/server/plan_server_storage.py" \
  --stage-root "$stage_root" --system-root / \
  --output "$stage_root/artifacts/storage.json"

env_dir="$stage_root/envs/probekv-py310"
cacheblend="$stage_root/src/CacheBlend-probekv-v6"
artifacts="$stage_root/artifacts/v6_setup"
mkdir -p "$artifacts"
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$python_bin" -m venv "$env_dir"
fi

export MAX_JOBS="${MAX_JOBS:-8}"
export HF_HOME="$stage_root/hf"
export HUGGINGFACE_HUB_CACHE="$stage_root/hf/hub"
export TRANSFORMERS_CACHE="$stage_root/hf/transformers"
export TORCH_EXTENSIONS_DIR="$stage_root/build/torch_extensions"
export XDG_CACHE_HOME="$stage_root/build/xdg"
export PIP_NO_CACHE_DIR=1

python="$env_dir/bin/python"
"$python" -m pip install --upgrade 'pip>=23.3,<25' 'setuptools>=68,<76' wheel
"$python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.2.1'
"$python" -m pip install --no-deps 'xformers==0.0.25'
"$python" -m pip install -r "$repo/requirements/server-tools.txt"
"$python" -m pip install --no-deps -e "$repo"

if ! "$python" "$repo/scripts/server/verify_cacheblend_patch.py" \
  --cacheblend "$cacheblend" \
  --mode probekv_v6_staggered_runtime \
  --output "$artifacts/cacheblend_patch.json" >/dev/null 2>&1; then
  if [[ -d "$cacheblend/.git" && -n "$(git -C "$cacheblend" status --porcelain)" ]]; then
    echo "existing CacheBlend tree is neither verified nor clean: $cacheblend" >&2
    exit 1
  fi
  bash "$repo/scripts/server/prepare_cacheblend.sh" \
    "$cacheblend" probekv_v6_staggered_runtime
  "$python" "$repo/scripts/server/verify_cacheblend_patch.py" \
    --cacheblend "$cacheblend" \
    --mode probekv_v6_staggered_runtime \
    --output "$artifacts/cacheblend_patch.json"
fi

"$python" -m pip install --no-build-isolation -e "$cacheblend/vllm_blend"
"$python" "$repo/scripts/server/capture_environment.py" \
  --output "$artifacts/environment.json"

rm -rf -- "$stage_root/build/torch_extensions" "$stage_root/build/xdg"

echo "A800 software tree prepared under $stage_root"
echo "Activate with: source $env_dir/bin/activate"
echo "Cache variables: HF_HOME=$HF_HOME TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
