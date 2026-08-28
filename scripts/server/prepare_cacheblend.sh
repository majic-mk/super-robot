#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/path/to/CacheBlend [cb0|probekv|probekv_closed_loop|probekv_v6_multiregion|probekv_v6_staggered_runtime|probekv_v6_prefix_hardened_runtime|probekv_v7_single_artifact_runtime|probekv_v8_training_free_residual_k|probekv_v8_schema6_joint_cfo]" >&2
  exit 2
fi

target="$1"
mode="${2:-probekv}"
expected_commit="b72d7945e6d6306f12be66520196e0f081fa2b0c"
repository="${PROBEKV_CACHEBLEND_SOURCE:-https://github.com/YaoJiayi/CacheBlend.git}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
patch_dir="$repo_root/patches/cacheblend"
python_bin="${PROBEKV_PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python is required; set PROBEKV_PYTHON_BIN if necessary" >&2
  exit 1
fi
python_bin="$(command -v "$python_bin")"

case "$mode" in
  cb0)
    patches=("0001-cb0-fix-suffix-length.patch")
    ;;
  probekv|probekv_closed_loop)
    patches=(
      "0001-cb0-fix-suffix-length.patch"
      "0002-probekv-segment-repair-mask.patch"
    )
    ;;
  probekv_v6_multiregion)
    patches=(
      "0001-cb0-fix-suffix-length.patch"
      "0002-probekv-segment-repair-mask.patch"
      "0003-probekv-multiregion-union-mask.patch"
    )
    ;;
  probekv_v6_staggered_runtime)
    patches=(
      "0001-cb0-fix-suffix-length.patch"
      "0002-probekv-segment-repair-mask.patch"
      "0003-probekv-multiregion-union-mask.patch"
      "0004-probekv-staggered-resumable-dual-model.patch"
    )
    ;;
  probekv_v6_prefix_hardened_runtime)
    patches=(
      "0001-cb0-fix-suffix-length.patch"
      "0002-probekv-segment-repair-mask.patch"
      "0003-probekv-multiregion-union-mask.patch"
      "0004-probekv-staggered-resumable-dual-model.patch"
      "0005-probekv-native-prefix-shadow.patch"
    )
    ;;
  probekv_v7_single_artifact_runtime|probekv_v8_training_free_residual_k)
    patches=(
      "0001-cb0-fix-suffix-length.patch"
      "0002-probekv-segment-repair-mask.patch"
      "0003-probekv-multiregion-union-mask.patch"
      "0004-probekv-staggered-resumable-dual-model.patch"
      "0005-probekv-native-prefix-shadow.patch"
      "0006-probekv-v7-conservative-repair-rounding.patch"
    )
    ;;
  probekv_v8_schema6_joint_cfo)
    patches=(
      "0001-cb0-fix-suffix-length.patch"
      "0002-probekv-segment-repair-mask.patch"
      "0003-probekv-multiregion-union-mask.patch"
      "0004-probekv-staggered-resumable-dual-model.patch"
      "0005-probekv-native-prefix-shadow.patch"
      "0006-probekv-v7-conservative-repair-rounding.patch"
      "0007-probekv-schema6-cfo-post-rope-hook.patch"
    )
    ;;
  *)
    echo "unsupported CacheBlend patch mode: $mode" >&2
    exit 2
    ;;
esac

case "$target" in
  /*) ;;
  *) echo "target must be an absolute path" >&2; exit 2 ;;
esac

if [[ -d "$target/.git" ]]; then
  if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
    echo "refusing to change a dirty CacheBlend worktree: $target" >&2
    exit 1
  fi
  git -C "$target" fetch origin
else
  if [[ -e "$target" ]]; then
    echo "target exists but is not a Git repository: $target" >&2
    exit 1
  fi
  if [[ -d "$repository" ]]; then
    if [[ ! -d "$repository/.git" ]]; then
      echo "local CacheBlend source is not a Git repository: $repository" >&2
      exit 1
    fi
    if ! git -C "$repository" cat-file -e "$expected_commit^{commit}"; then
      echo "local CacheBlend source lacks the frozen commit" >&2
      exit 1
    fi
  fi
  git clone --no-hardlinks "$repository" "$target"
fi

git -C "$target" checkout --detach "$expected_commit"
actual_commit="$(git -C "$target" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "CacheBlend commit mismatch: $actual_commit" >&2
  exit 1
fi

for patch_name in "${patches[@]}"; do
  patch_path="$patch_dir/$patch_name"
  if [[ ! -f "$patch_path" ]]; then
    echo "missing tracked CacheBlend patch: $patch_path" >&2
    exit 1
  fi
  git -C "$target" apply --unidiff-zero --check "$patch_path"
  git -C "$target" apply --unidiff-zero "$patch_path"
done

git -C "$target" diff --check
git -C "$target" add -u
patched_tree="$(git -C "$target" write-tree)"
patch_sha256="$(
  for patch_name in "${patches[@]}"; do
    cat "$patch_dir/$patch_name"
  done | sha256sum | awk '{print $1}'
)"

PROBEKV_CB_COMMIT="$actual_commit" \
PROBEKV_CB_MODE="$mode" \
PROBEKV_CB_PATCH_SHA="$patch_sha256" \
PROBEKV_CB_TREE="$patched_tree" \
"$python_bin" - <<'PY'
import json
import os

print(json.dumps({
    "cacheblend_commit": os.environ["PROBEKV_CB_COMMIT"],
    "patch_mode": os.environ["PROBEKV_CB_MODE"],
    "cacheblend_patch_sha256": os.environ["PROBEKV_CB_PATCH_SHA"],
    "cacheblend_tree": os.environ["PROBEKV_CB_TREE"],
}, sort_keys=True))
PY
