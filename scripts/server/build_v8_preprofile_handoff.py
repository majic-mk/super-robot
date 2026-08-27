from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.cacheblend_patch import combined_patch_sha256, patch_files_for_mode
from probekv.io import write_json
from probekv.v8_a800_jobs import build_v8_preprofile_manifest


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/a800_server_lock_v8.json")
    parser.add_argument("--patch-manifest", default="patches/cacheblend/manifest.json")
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--policy", choices=("causal_commit_wait", "immediate_staggered_closed_loop"), default="causal_commit_wait")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    lock = load(Path(args.lock).resolve())
    patch_audit = load(Path(args.patch_audit).resolve())
    model_audit = load(Path(args.model_audit).resolve())
    model = lock["models"][args.model_key]
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    patch_paths = patch_files_for_mode(
        Path(args.patch_manifest).resolve(), lock["stack"]["cacheblend_patch_mode"]
    )
    patch_sha = combined_patch_sha256(patch_paths)
    if patch_audit.get("cacheblend_patch_sha256") != patch_sha:
        raise ValueError("patch audit differs from the v8 patch manifest")
    manifest = build_v8_preprofile_manifest(
        code_commit=code_commit,
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_hash=model_audit["tokenizer_hash"],
        adapter_name=model["adapter_name"],
        selection_execution_policy=args.policy,
        checkpoint_depths=model["completed_depths"],
        cacheblend_patch_sha256=patch_sha,
        cacheblend_tree=patch_audit["cacheblend_tree"],
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / ("preprofile_%s_%s.json" % (args.model_key, args.policy)), manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
