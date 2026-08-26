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
python_bin="$(command -v "$python_bin")"
python_version="$($python_bin -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$python_version" != "3.10" ]]; then
  echo "expected Python 3.10, found $python_version" >&2
  exit 1
fi
nvcc_bin="${PROBEKV_NVCC_BIN:-}"
if [[ -z "$nvcc_bin" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_bin="$(command -v nvcc)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin=/usr/local/cuda/bin/nvcc
  fi
fi
if [[ -z "$nvcc_bin" || ! -x "$nvcc_bin" ]] || \
   ! "$nvcc_bin" --version | grep -q 'release 12\.1'; then
  echo "CUDA toolkit 12.1 with nvcc is required before building vLLM" >&2
  exit 1
fi
export PATH="$(dirname "$nvcc_bin"):$PATH"

mkdir -p "$stage_root"/{src,envs,hf,build,artifacts}
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$repo/scripts/server/plan_server_storage.py" \
  --stage-root "$stage_root" --system-root / \
  --output "$stage_root/artifacts/storage.json"

env_dir="${PROBEKV_ENV_DIR:-$stage_root/envs/probekv-py310}"
case "$env_dir" in
  "$stage_root"/envs/*) ;;
  *) echo "PROBEKV_ENV_DIR must remain inside $stage_root/envs" >&2; exit 2 ;;
esac
cacheblend="${PROBEKV_CACHEBLEND_TARGET:-$stage_root/src/CacheBlend-probekv-v6}"
case "$cacheblend" in
  "$stage_root"/src/*) ;;
  *) echo "PROBEKV_CACHEBLEND_TARGET must stay under $stage_root/src" >&2; exit 2;;
esac
artifacts="$stage_root/artifacts/v6_setup"
mkdir -p "$artifacts"
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$python_bin" -m venv "$env_dir"
fi
env_python_version="$($env_dir/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$env_python_version" != "3.10" ]]; then
  echo "selected ProbeKV environment is not Python 3.10: $env_dir" >&2
  exit 1
fi

export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${PROBEKV_CUDA_ARCH_LIST:-8.0}"
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
  --mode probekv_v6_prefix_hardened_runtime \
  --manifest "$repo/patches/cacheblend/manifest.json" \
  --output "$artifacts/cacheblend_patch.json" >/dev/null 2>&1; then
  if [[ -d "$cacheblend/.git" && -n "$(git -C "$cacheblend" status --porcelain)" ]]; then
    echo "existing CacheBlend tree is neither verified nor clean: $cacheblend" >&2
    exit 1
  fi
  bash "$repo/scripts/server/prepare_cacheblend.sh" \
    "$cacheblend" probekv_v6_prefix_hardened_runtime
  "$python" "$repo/scripts/server/verify_cacheblend_patch.py" \
    --cacheblend "$cacheblend" \
    --mode probekv_v6_prefix_hardened_runtime \
    --manifest "$repo/patches/cacheblend/manifest.json" \
    --output "$artifacts/cacheblend_patch.json"
fi

if [[ -n "${PROBEKV_PREBUILT_VLLM_SOURCE:-}" ]]; then
  site_packages="$($python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  "$python" "$repo/scripts/server/install_prebuilt_vllm_extensions.py" \
    --donor "$PROBEKV_PREBUILT_VLLM_SOURCE" \
    --target "$cacheblend" \
    --site-packages "$site_packages" \
    --cuobjdump "$(dirname "$nvcc_bin")/cuobjdump" \
    --output "$artifacts/vllm_install.json"
  echo "Prebuilt vLLM extension dynamic loading is deferred to the A800 gate."
else
  "$python" -m pip install --no-build-isolation -e "$cacheblend/vllm_blend"
fi
"$python" "$repo/scripts/server/capture_environment.py" \
  --repo "$repo" \
  --output "$artifacts/environment.json"

rm -rf -- "$stage_root/build/torch_extensions" "$stage_root/build/xdg"

echo "A800 software tree prepared under $stage_root"
echo "Activate with: source $env_dir/bin/activate"
echo "Cache variables: HF_HOME=$HF_HOME TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
