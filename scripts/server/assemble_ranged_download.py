"""Atomically assemble and verify an ordered ranged download."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(
    parts: list[Path],
    output: Path,
    total_bytes: int,
    regular_part_bytes: int,
) -> None:
    if len(parts) < 2:
        raise ValueError("at least two range parts are required")
    expected_last = total_bytes - regular_part_bytes * (len(parts) - 1)
    if expected_last <= 0 or expected_last > regular_part_bytes:
        raise ValueError("part geometry does not match total byte count")
    expected_sizes = [regular_part_bytes] * (len(parts) - 1) + [
        expected_last
    ]
    actual_sizes = [part.stat().st_size for part in parts]
    if actual_sizes != expected_sizes:
        raise ValueError(
            "range part sizes mismatch: expected %s, observed %s"
            % (expected_sizes, actual_sizes)
        )
    temporary = output.with_suffix(output.suffix + ".assembling")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    if temporary.stat().st_size != total_bytes:
        raise RuntimeError("assembled file byte count mismatch")
    os.replace(str(temporary), str(output))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-bytes", required=True, type=int)
    parser.add_argument("--regular-part-bytes", required=True, type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-git-blob-sha1")
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    part_directory = Path(args.parts).resolve()
    parts = sorted(
        path
        for path in part_directory.iterdir()
        if path.is_file() and path.name.startswith("part")
    )
    output = Path(args.output).resolve()
    assemble(
        parts,
        output,
        args.total_bytes,
        args.regular_part_bytes,
    )
    sha256 = sha256_file(output)
    blob_sha1 = git_blob_sha1(output)
    if args.expected_sha256 and sha256 != args.expected_sha256:
        raise RuntimeError("assembled SHA256 mismatch")
    if (
        args.expected_git_blob_sha1
        and blob_sha1 != args.expected_git_blob_sha1
    ):
        raise RuntimeError("assembled Git blob identity mismatch")
    audit = {
        "parts": [str(path) for path in parts],
        "part_sizes": [path.stat().st_size for path in parts],
        "output": str(output),
        "total_bytes": output.stat().st_size,
        "sha256": sha256,
        "git_blob_sha1": blob_sha1,
        "verified": True,
        "paper_evidence": False,
        "evidence_class": "data_preparation",
    }
    atomic_write_json(Path(args.audit).resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
