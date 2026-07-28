"""Verify and archive provenance for an externally patched CacheBlend tree."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.cacheblend_patch import (
    combined_patch_sha256,
    load_patch_manifest,
    patch_files_for_mode,
)
from probekv.io import atomic_write_json


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacheblend", required=True)
    parser.add_argument("--mode", required=True, choices=("cb0", "probekv"))
    parser.add_argument("--manifest", default="patches/cacheblend/manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cacheblend = Path(args.cacheblend).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_patch_manifest(manifest_path)
    actual_commit = _git(cacheblend, "rev-parse", "HEAD")
    if actual_commit != manifest["base_commit"]:
        raise RuntimeError("CacheBlend base commit mismatch")
    patch_paths = patch_files_for_mode(manifest_path, args.mode)
    for patch_path in patch_paths:
        subprocess.check_call(
            [
                "git",
                "-C",
                str(cacheblend),
                "apply",
                "--reverse",
                "--check",
                str(patch_path),
            ]
        )
    subprocess.check_call(["git", "-C", str(cacheblend), "diff", "--check"])
    subprocess.check_call(["git", "-C", str(cacheblend), "add", "-u"])
    tree = _git(cacheblend, "write-tree")
    payload = {
        "cacheblend_commit": actual_commit,
        "patch_mode": args.mode,
        "patches": [path.name for path in patch_paths],
        "cacheblend_patch_sha256": combined_patch_sha256(patch_paths),
        "cacheblend_tree": tree,
        "innovation_claim": False,
    }
    atomic_write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
