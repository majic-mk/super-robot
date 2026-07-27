"""Fail-fast audit for the pinned A100 paper environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def output(command):
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return getattr(error, "output", "") or str(error)


def package_version(name):
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version
    try:
        return version(name)
    except Exception:
        return None


def main() -> int:
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "git_status": output(["git", "status", "--short"]),
        "nvidia_smi": output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "torch": package_version("torch"),
        "vllm": package_version("vllm"),
        "xformers": package_version("xformers"),
    }
    failures = []
    if "A100" not in record["nvidia_smi"]:
        failures.append("paper timing requires an NVIDIA A100")
    if record["torch"] != "2.2.1":
        failures.append("expected torch==2.2.1 for the pinned CacheBlend stack")
    if record["vllm"] != "0.4.1":
        failures.append("expected vllm==0.4.1 for the pinned CacheBlend stack")
    if record["git_status"]:
        failures.append("paper run must start from a clean worktree")
    record["valid"] = not failures
    record["failures"] = failures
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
