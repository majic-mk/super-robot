"""Relocate an already-built vLLM editable install to an audited patch tree.

This is for low-memory no-GPU preparation only.  It reuses CUDA extensions
built from the same frozen CacheBlend base commit and changes only the editable
Python source mapping.  The A800 runner re-hashes the extensions before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def command(cwd: Path, *values: str) -> str:
    return subprocess.check_output(values, cwd=str(cwd), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--from-cacheblend", required=True)
    parser.add_argument("--to-cacheblend", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Keep the virtual-environment entry path. Resolving its symlink would make
    # Python use the base interpreter's site-packages instead of the venv.
    python = Path(args.python).absolute()
    source = Path(args.from_cacheblend).resolve()
    target = Path(args.to_cacheblend).resolve()
    audit = json.loads(Path(args.patch_audit).read_text(encoding="utf-8"))
    expected_commit = str(audit["cacheblend_commit"])
    if command(source, "git", "rev-parse", "HEAD") != expected_commit:
        raise ValueError("compiled source does not use the frozen CacheBlend commit")
    if command(target, "git", "rev-parse", "HEAD") != expected_commit:
        raise ValueError("target source does not use the frozen CacheBlend commit")
    if command(target, "git", "write-tree") != audit["cacheblend_tree"]:
        raise ValueError("target source tree differs from the patch audit")

    source_package = source / "vllm_blend" / "vllm"
    target_package = target / "vllm_blend" / "vllm"
    extensions = sorted(source_package.glob("*.so"))
    if not extensions:
        raise FileNotFoundError("compiled editable source has no vLLM extensions")
    copied: dict[str, str] = {}
    for extension in extensions:
        destination = target_package / extension.name
        shutil.copy2(extension, destination)
        copied[extension.name] = sha256(destination)

    purelib = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
        ).strip()
    )
    finders = sorted(purelib.glob("__editable___vllm_*_finder.py"))
    if len(finders) != 1:
        raise RuntimeError("expected exactly one vLLM editable finder")
    finder = finders[0]
    old_root = str((source / "vllm_blend").resolve())
    new_root = str((target / "vllm_blend").resolve())
    contents = finder.read_text(encoding="utf-8")
    if old_root not in contents:
        raise RuntimeError("editable finder does not point to --from-cacheblend")
    atomic_write(finder, contents.replace(old_root, new_root))

    resolved = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.util; print(importlib.util.find_spec('vllm').origin)",
        ],
        text=True,
    ).strip()
    expected_origin = str((target_package / "__init__.py").resolve())
    if str(Path(resolved).resolve()) != expected_origin:
        raise RuntimeError("relocated editable finder did not resolve the target vLLM")

    result = {
        "schema_version": 6,
        "cacheblend_commit": expected_commit,
        "cacheblend_tree": audit["cacheblend_tree"],
        "runtime_vllm_origin": expected_origin,
        "compiled_extension_sha256": copied,
        "rebuilt": False,
        "paper_evidence": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
