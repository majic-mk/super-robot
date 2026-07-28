from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


_HUNK_HEADER = re.compile(
    r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@"
)


def validate_unified_diff(path: Path) -> None:
    """Reject malformed hunk counts before a patch reaches the server."""
    lines = path.read_text(encoding="utf-8").splitlines()
    hunk_count = 0
    index = 0
    while index < len(lines):
        match = _HUNK_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue
        hunk_count += 1
        expected_old = int(match.group(1) or 1)
        expected_new = int(match.group(2) or 1)
        actual_old = 0
        actual_new = 0
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.startswith("@@ ") or line.startswith("diff --git "):
                break
            if line.startswith("\\"):
                index += 1
                continue
            if not line or line[0] not in (" ", "+", "-"):
                raise ValueError(
                    "invalid unified diff line in %s: %r" % (path, line)
                )
            if line[0] in (" ", "-"):
                actual_old += 1
            if line[0] in (" ", "+"):
                actual_new += 1
            index += 1
        if (actual_old, actual_new) != (expected_old, expected_new):
            raise ValueError(
                "malformed hunk in %s: expected %d/%d old/new lines, "
                "observed %d/%d"
                % (
                    path,
                    expected_old,
                    expected_new,
                    actual_old,
                    actual_new,
                )
            )
    if hunk_count == 0:
        raise ValueError("patch contains no unified diff hunks: %s" % path)


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
    for path in paths:
        validate_unified_diff(path)
    return paths


def combined_patch_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()
