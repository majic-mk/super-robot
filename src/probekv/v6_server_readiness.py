from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _command_output(command: list[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return (getattr(error, "output", "") or str(error)).strip()


def _memory_total_gib() -> float:
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size / float(1024**3)
        except (OSError, ValueError, TypeError):
            pass
    return 0.0


def collect_no_gpu_host(repo: Path, data_root: Path) -> Dict[str, Any]:
    """Collect only CPU/storage/software facts; no CUDA device is required."""

    nvcc = _command_output(["nvcc", "--version"])
    nvcc_match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc)
    disk = shutil.disk_usage(str(data_root))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "host_memory_gib": _memory_total_gib(),
        "data_root": str(data_root.resolve()),
        "data_disk_free_gib": disk.free / float(1024**3),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"], repo),
        "git_status": _command_output(["git", "status", "--porcelain"], repo),
        "nvcc_cuda": nvcc_match.group(1) if nvcc_match else None,
        "packages": {
            "torch": _package_version("torch"),
            "xformers": _package_version("xformers"),
            "vllm": _package_version("vllm"),
            "numpy": _package_version("numpy"),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "huggingface-hub": _package_version("huggingface-hub"),
            "ray": _package_version("ray"),
            "cmake": _package_version("cmake"),
            "ninja": _package_version("ninja"),
        },
    }


def _version_matches(expected: Any, observed: Any, local_suffix: bool = False) -> bool:
    expected_text = str(expected)
    observed_text = str(observed)
    if local_suffix:
        observed_text = observed_text.split("+", 1)[0]
    return observed_text == expected_text


def evaluate_no_gpu_readiness(
    lock: Mapping[str, Any],
    job_manifest: Mapping[str, Any],
    host: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    patch_audit: Mapping[str, Any],
    *,
    expected_code_commit: str,
    actual_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    """Evaluate the last gate that is meaningful before an A800 is attached.

    Passing this gate means the immutable code, model, patched backend, Python
    environment and job manifest can be handed to a GPU instance.  It does not
    claim that the layer-resumable vLLM engine has passed CUDA qualification.
    """

    failures: List[str] = []
    platform_lock = lock.get("platform", {})
    stack = lock.get("stack", {})
    model = lock.get("model", {})
    runtime = lock.get("runtime", {})

    observed_python = str(host.get("python", ""))
    if not observed_python.startswith(str(platform_lock.get("python_major_minor")) + "."):
        failures.append(
            "Python must be %s.x, found %s"
            % (platform_lock.get("python_major_minor"), observed_python)
        )
    if int(host.get("cpu_count", 0)) < int(platform_lock.get("minimum_cpu_count", 0)):
        failures.append("CPU count is below the server lock")
    if float(host.get("host_memory_gib", 0.0)) < float(
        platform_lock.get("minimum_host_memory_gib", 0.0)
    ):
        failures.append("host memory is below the server lock")
    if float(host.get("data_disk_free_gib", 0.0)) < float(
        platform_lock.get("minimum_data_disk_free_gib", 0.0)
    ):
        failures.append("data-disk free space is below the server lock")
    if str(host.get("git_commit")) != str(expected_code_commit):
        failures.append("checked-out ProbeKV commit does not match the requested SHA")
    if host.get("git_status"):
        failures.append("ProbeKV worktree must be clean")
    if str(host.get("nvcc_cuda")) != str(stack.get("pytorch_cuda")):
        failures.append("nvcc CUDA does not match the pinned stack")

    packages = host.get("packages", {})
    for package, key in (
        ("torch", "pytorch"),
        ("xformers", "xformers"),
        ("vllm", "vllm"),
        ("numpy", "numpy"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("huggingface-hub", "huggingface-hub"),
        ("ray", "ray"),
        ("cmake", "cmake"),
        ("ninja", "ninja"),
    ):
        if not _version_matches(
            stack.get(key),
            packages.get(package),
            local_suffix=(package == "torch"),
        ):
            failures.append(
                "expected %s==%s, found %s"
                % (package, stack.get(key), packages.get(package))
            )

    if job_manifest.get("protocol_version") != 6 or job_manifest.get("paper_evidence") is not False:
        failures.append("job manifest is not a protocol-v6 non-paper manifest")
    if int(job_manifest.get("jobs", -1)) != 140:
        failures.append("v6 A800 manifest must contain exactly 140 jobs")
    if str(job_manifest.get("code_commit")) != str(expected_code_commit):
        failures.append("job manifest was built from another ProbeKV commit")
    if job_manifest.get("git_clean") is not True:
        failures.append("job manifest was not frozen from a clean worktree")
    for manifest_key, hash_key in (
        ("jobs_sha256", "jobs_sha256"),
        ("config_sha256", "config_sha256"),
        ("contract_sha256", "contract_sha256"),
        ("server_lock_sha256", "server_lock_sha256"),
    ):
        if str(job_manifest.get(manifest_key)) != str(actual_hashes.get(hash_key)):
            failures.append("job manifest %s does not match the checked file" % manifest_key)
    manifest_model = job_manifest.get("model", {})
    if manifest_model.get("model_id") != model.get("model_id") or manifest_model.get("revision") != model.get("revision"):
        failures.append("job manifest model identity does not match the server lock")
    manifest_runtime = job_manifest.get("runtime", {})
    if manifest_runtime.get("backend") != runtime.get("backend"):
        failures.append("job manifest runtime backend does not match the server lock")

    if model_audit.get("complete") is not True:
        failures.append("model snapshot audit is incomplete")
    if model_audit.get("model_id") != model.get("model_id") or model_audit.get("revision") != model.get("revision"):
        failures.append("downloaded model identity does not match the server lock")

    if patch_audit.get("cacheblend_commit") != stack.get("cacheblend_commit"):
        failures.append("CacheBlend base commit does not match the server lock")
    if patch_audit.get("patch_mode") != stack.get("cacheblend_patch_mode"):
        failures.append("CacheBlend patch mode does not match the server lock")
    manifest_cacheblend = job_manifest.get("cacheblend", {})
    if manifest_cacheblend.get("patch_sha256") != patch_audit.get("cacheblend_patch_sha256"):
        failures.append("CacheBlend patch digest differs from the frozen job manifest")
    if not patch_audit.get("cacheblend_tree"):
        failures.append("CacheBlend patched tree identity is missing")

    artifact_ready = not failures
    runtime_status = str(runtime.get("implementation_status", "unknown"))
    concrete_engine_ready = runtime_status.startswith(
        "concrete_engine_hook_complete"
    )
    return {
        "schema_version": 1,
        "stage": "v6_no_gpu_readiness",
        "paper_evidence": False,
        "expected_code_commit": expected_code_commit,
        "artifact_preparation_ready": artifact_ready,
        "gpu_rental_ready_for_runtime_bringup": artifact_ready,
        "gpu_rental_ready_for_runtime_qualification": (
            artifact_ready and concrete_engine_ready
        ),
        "gpu_rental_ready_for_h1_h2": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "runtime_implementation_status": runtime_status,
        "blocking_source_implementation": (
            None
            if concrete_engine_ready
            else "concrete layer-resumable CacheBlend/vLLM engine hook"
        ),
        "next_required_gates": [
            "A800-hardware-and-stack",
            "v6-concrete-engine-hook",
            "v6-r1-dense-equivalence",
            "v6-CUDA-timing-and-140-job-bringup",
        ],
        "failures": failures,
    }
