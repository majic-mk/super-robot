#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/CacheBlend" >&2
  exit 2
fi

target="$1"
expected_commit="b72d7945e6d6306f12be66520196e0f081fa2b0c"
repository="https://github.com/YaoJiayi/CacheBlend.git"

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
  git clone "$repository" "$target"
fi

git -C "$target" checkout --detach "$expected_commit"
actual_commit="$(git -C "$target" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "CacheBlend commit mismatch: $actual_commit" >&2
  exit 1
fi
echo "CacheBlend pinned at $actual_commit"
