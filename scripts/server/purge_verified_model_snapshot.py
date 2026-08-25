"""Remove one regenerable HF snapshot after qualification, never user files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from probekv.model_adapters import MISTRAL_SPEC, QWEN_SPEC


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
        return True
    except ValueError:
        return False


def purge_snapshot(stage_root: Path, audit_path: Path, expected_model_id: str) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("complete") is not True:
        raise ValueError("only a previously complete audited snapshot may be purged")
    if audit.get("model_id") != expected_model_id:
        raise ValueError("audit model identity differs from purge request")
    allowed = {MISTRAL_SPEC.model_id, QWEN_SPEC.model_id}
    if expected_model_id not in allowed:
        raise ValueError("model is outside the frozen ProbeKV set")
    hf_root = Path(os.path.abspath(stage_root / "hf"))
    snapshot = Path(os.path.abspath(audit["snapshot_path"]))
    if not _inside(snapshot, hf_root) or "snapshots" not in snapshot.parts:
        raise ValueError("snapshot is outside the ProbeKV HF cache")
    if snapshot == hf_root or not snapshot.is_dir():
        raise ValueError("snapshot purge target is missing or too broad")
    repository_root = snapshot.parent.parent
    blob_targets = set()
    for path in snapshot.rglob("*"):
        if path.is_file() or path.is_symlink():
            target = path.resolve()
            if _inside(target, repository_root / "blobs"):
                blob_targets.add(target)
    other_targets = set()
    snapshots_root = repository_root / "snapshots"
    for other in snapshots_root.glob("*"):
        if Path(os.path.abspath(other)) == snapshot:
            continue
        for path in other.rglob("*"):
            if path.is_file() or path.is_symlink():
                other_targets.add(path.resolve())
    removable_blobs = sorted(blob_targets - other_targets)
    shutil.rmtree(snapshot)
    removed_bytes = 0
    for blob in removable_blobs:
        if blob.is_file() and _inside(blob, repository_root / "blobs"):
            removed_bytes += blob.stat().st_size
            blob.unlink()
    return {
        "model_id": expected_model_id,
        "snapshot_removed": str(snapshot),
        "unshared_blobs_removed": len(removable_blobs),
        "removed_bytes": removed_bytes,
        "recoverable_by_redownload": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    result = purge_snapshot(
        Path(args.stage_root), Path(args.audit), args.model_id
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
