"""Download and audit the exact frozen Hugging Face model revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file


def audit_snapshot(snapshot: Path, model_id: str, revision: str) -> dict:
    resolved_revision = snapshot.resolve().name
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if len(revision) == 40 and resolved_revision != revision:
        missing.append("resolved snapshot revision %s" % revision)
    tokenizer_assets = tuple(
        name
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer.model.v3")
        if (snapshot / name).is_file()
    )
    if not tokenizer_assets:
        missing.append("tokenizer.json|tokenizer.model|tokenizer.model.v3")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected_weight_names = tuple(sorted(set(index.get("weight_map", {}).values())))
        if not expected_weight_names:
            missing.append("non-empty model.safetensors.index.json weight_map")
        for name in expected_weight_names:
            if not (snapshot / name).is_file():
                missing.append(name)
        weights = [snapshot / name for name in expected_weight_names if (snapshot / name).is_file()]
    else:
        weights = sorted(snapshot.glob("*.safetensors"))
        if not weights:
            missing.append("*.safetensors")
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    selected_hashes = {
        name: sha256_file(snapshot / name)
        for name in required + (("model.safetensors.index.json",) if index_path.is_file() else ())
        if (snapshot / name).is_file()
    }
    return {
        "schema_version": 1,
        "paper_evidence": False,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "weight_files": [path.name for path in weights],
        "tokenizer_assets": list(tokenizer_assets),
        "selected_file_sha256": selected_hashes,
        "missing": missing,
        "complete": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            cache_dir=str(Path(args.cache_dir).resolve()),
            local_files_only=args.local_files_only,
        )
    )
    audit = audit_snapshot(snapshot, args.model_id, args.revision)
    atomic_write_json(Path(args.output).resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
