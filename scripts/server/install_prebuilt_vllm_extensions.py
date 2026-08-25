"""Reuse audited vLLM extensions when a CPU-only container cannot compile them."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file


FROZEN_COMMIT = "b72d7945e6d6306f12be66520196e0f081fa2b0c"
ALLOWED_PATCHED_SUFFIXES = (".py",)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def parse_sm_arches(cuobjdump_output: str) -> set[str]:
    return set(re.findall(r"\.sm_([0-9]+)\.cubin", cuobjdump_output))


def changed_files(repo: Path) -> tuple[str, ...]:
    output = _git(repo, "diff", "HEAD", "--name-only")
    return tuple(line for line in output.splitlines() if line)


def install_prebuilt_extensions(
    donor: Path,
    target: Path,
    site_packages: Path,
    cuobjdump: Path,
) -> dict:
    donor = donor.resolve()
    target = target.resolve()
    site_packages = site_packages.resolve()
    if donor == target:
        raise ValueError("prebuilt donor and patched target must differ")
    for repo in (donor, target):
        if _git(repo, "rev-parse", "HEAD") != FROZEN_COMMIT:
            raise RuntimeError("CacheBlend extension tree has the wrong base commit")
        native_changes = [
            name for name in changed_files(repo)
            if not name.endswith(ALLOWED_PATCHED_SUFFIXES)
        ]
        if native_changes:
            raise RuntimeError(
                "prebuilt extension reuse forbids native/build changes: %s"
                % ",".join(native_changes)
            )
    donor_package = donor / "vllm_blend" / "vllm"
    target_package = target / "vllm_blend" / "vllm"
    binaries = []
    for stem in ("_C", "_moe_C"):
        matches = sorted(donor_package.glob(stem + "*.so"))
        if len(matches) != 1:
            raise RuntimeError("expected one prebuilt %s extension" % stem)
        source = matches[0]
        dump = subprocess.check_output(
            [str(cuobjdump), "-lelf", str(source)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        arches = parse_sm_arches(dump)
        if arches != {"80"}:
            raise RuntimeError(
                "prebuilt extension must contain only frozen A800 sm_80, found %r"
                % sorted(arches)
            )
        destination = target_package / source.name
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError("prebuilt extension digest changed during copy")
        binaries.append({
            "name": source.name,
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "sm_arches": sorted(arches),
        })
    site_packages.mkdir(parents=True, exist_ok=True)
    pth = site_packages / "00-probekv-cacheblend-v6.pth"
    source_root = target / "vllm_blend"
    pth.write_text(
        "import sys; sys.path.insert(0, %r)\n" % str(source_root),
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "paper_evidence": False,
        "install_mode": "audited_prebuilt_native_python_patch",
        "cacheblend_commit": FROZEN_COMMIT,
        "donor": str(donor),
        "target": str(target),
        "target_python_changes": list(changed_files(target)),
        "pth": str(pth),
        "binaries": binaries,
        "gpu_qualification_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = install_prebuilt_extensions(
        Path(args.donor),
        Path(args.target),
        Path(args.site_packages),
        Path(args.cuobjdump),
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
