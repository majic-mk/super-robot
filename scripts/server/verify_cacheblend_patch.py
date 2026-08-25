"""Verify and archive provenance for an externally patched CacheBlend tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
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


def _git_with_env(repo: Path, env, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
        env=env,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacheblend", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "cb0",
            "probekv",
            "probekv_closed_loop",
            "probekv_v6_multiregion",
            "probekv_v6_staggered_runtime",
        ),
    )
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
    subprocess.check_call(["git", "-C", str(cacheblend), "diff", "--check"])
    with tempfile.TemporaryDirectory(prefix="probekv-cacheblend-verify-") as root:
        root_path = Path(root)
        expected_repo = root_path / "expected"
        subprocess.check_call(
            [
                "git",
                "clone",
                "--quiet",
                "--no-hardlinks",
                str(cacheblend),
                str(expected_repo),
            ]
        )
        subprocess.check_call(
            [
                "git",
                "-C",
                str(expected_repo),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "--quiet",
                "--detach",
                manifest["base_commit"],
            ]
        )
        for patch_path in patch_paths:
            subprocess.check_call(
                [
                    "git",
                    "-C",
                    str(expected_repo),
                    "apply",
                    "--unidiff-zero",
                    "--whitespace=nowarn",
                    str(patch_path),
                ]
            )
        subprocess.check_call(
            ["git", "-C", str(expected_repo), "diff", "--check"]
        )
        subprocess.check_call(
            ["git", "-C", str(expected_repo), "add", "-u"]
        )
        expected_tree = _git(expected_repo, "write-tree")

        alternate_index = root_path / "actual.index"
        index_env = dict(os.environ)
        index_env["GIT_INDEX_FILE"] = str(alternate_index)
        subprocess.check_call(
            ["git", "-C", str(cacheblend), "read-tree", "HEAD"],
            env=index_env,
        )
        subprocess.check_call(
            ["git", "-C", str(cacheblend), "add", "-u"],
            env=index_env,
        )
        tree = _git_with_env(cacheblend, index_env, "write-tree")
        if tree != expected_tree:
            raise RuntimeError(
                "patched CacheBlend tree does not equal the ordered patchset"
            )
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
