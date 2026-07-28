from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


def load_patch_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("base_commit") != (
        "b72d7945e6d6306f12be66520196e0f081fa2b0c"
    ):
        raise ValueError("unexpected CacheBlend base commit")
    modes = manifest.get("patches")
    if not isinstance(modes, dict) or not {"cb0", "probekv"}.issubset(modes):
        raise ValueError("patch manifest must define cb0 and probekv modes")
    return manifest


def patch_files_for_mode(manifest_path: Path, mode: str) -> Tuple[Path, ...]:
    manifest = load_patch_manifest(manifest_path)
    if mode not in manifest["patches"]:
        raise ValueError("unknown CacheBlend patch mode: %s" % mode)
    root = manifest_path.parent
    paths = tuple(root / str(name) for name in manifest["patches"][mode])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing CacheBlend patches: %s" % ", ".join(missing))
    return paths


def combined_patch_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()
