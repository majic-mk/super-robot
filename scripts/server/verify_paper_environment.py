"""Fail-fast audit for the config-driven formal paper environment."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml


def command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(command),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return getattr(error, "output", "") or str(error)


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version
    try:
        return version(name)
    except Exception:
        return None


def versions_match(expected: Any, observed: Any, allow_local_suffix: bool = False) -> bool:
    expected_text = str(expected)
    observed_text = str(observed)
    if allow_local_suffix and "+" not in expected_text:
        observed_text = observed_text.split("+", 1)[0]
    return observed_text == expected_text


def _torch_record() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch_cuda": torch.version.cuda,
            "torch_device_count": torch.cuda.device_count(),
        }
    except Exception as error:
        return {
            "torch_cuda": None,
            "torch_device_count": 0,
            "torch_error": "%s: %s" % (type(error).__name__, error),
        }


def _gpu_records() -> List[Dict[str, Any]]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    records = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            memory_mib = int(parts[1])
        except ValueError:
            continue
        records.append(
            {
                "name": parts[0],
                "memory_mib": memory_mib,
                "uuid": parts[2],
                "compute_capability": parts[3],
            }
        )
    return records


def collect_environment() -> Dict[str, Any]:
    nvcc = command_output(["nvcc", "--version"])
    nvcc_match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc)
    record: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "gpus": _gpu_records(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvcc": nvcc,
        "nvcc_cuda": nvcc_match.group(1) if nvcc_match else None,
        "torch": package_version("torch"),
        "vllm": package_version("vllm"),
        "xformers": package_version("xformers"),
    }
    record.update(_torch_record())
    return record


def evaluate_environment(
    contract: Mapping[str, Any], record: Mapping[str, Any]
) -> List[str]:
    failures = []
    expected_hardware = contract["hardware"]["primary"]
    expected_stack = contract["stacks"]["primary"]
    gpus = list(record.get("gpus", ()))
    if len(gpus) != int(expected_hardware["gpu_count"]):
        failures.append(
            "expected %d GPU, found %d"
            % (int(expected_hardware["gpu_count"]), len(gpus))
        )
    for gpu in gpus:
        if re.search(str(expected_hardware["gpu_name_regex"]), gpu["name"]) is None:
            failures.append("unexpected GPU name: %s" % gpu["name"])
        if int(gpu["memory_mib"]) < int(expected_hardware["minimum_memory_mib"]):
            failures.append("GPU memory is below the configured minimum")
        if str(gpu["compute_capability"]) != str(
            expected_hardware["compute_capability"]
        ):
            failures.append("unexpected GPU compute capability")
    if int(record.get("torch_device_count", 0)) != int(
        expected_hardware["gpu_count"]
    ):
        failures.append("PyTorch GPU count does not match the hardware contract")
    for package in ("torch", "vllm", "xformers"):
        expected_key = "pytorch" if package == "torch" else package
        if not versions_match(
            expected_stack[expected_key],
            record.get(package),
            allow_local_suffix=(package == "torch"),
        ):
            failures.append(
                "expected %s==%s, found %s"
                % (package, expected_stack[expected_key], record.get(package))
            )
    if str(record.get("torch_cuda")) != str(expected_stack["cuda"]):
        failures.append(
            "expected torch CUDA %s, found %s"
            % (expected_stack["cuda"], record.get("torch_cuda"))
        )
    if str(record.get("nvcc_cuda")) != str(expected_stack["cuda"]):
        failures.append(
            "expected nvcc CUDA %s, found %s"
            % (expected_stack["cuda"], record.get("nvcc_cuda"))
        )
    if record.get("git_status"):
        failures.append("paper run must start from a clean worktree")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    contract = yaml.safe_load(Path(args.contract).read_text(encoding="utf-8"))
    record = collect_environment()
    failures = evaluate_environment(contract, record)
    result = dict(record)
    result["contract"] = str(Path(args.contract).resolve())
    result["valid"] = not failures
    result["failures"] = failures
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
