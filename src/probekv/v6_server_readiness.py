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


def _nvcc_executable() -> str:
    """Resolve nvcc even when a non-interactive SSH shell has a minimal PATH."""

    explicit = os.environ.get("PROBEKV_NVCC_BIN")
    if explicit:
        return explicit
    detected = shutil.which("nvcc")
    if detected:
        return detected
    cuda_default = Path("/usr/local/cuda/bin/nvcc")
    return str(cuda_default) if cuda_default.is_file() else "nvcc"


def collect_no_gpu_host(repo: Path, data_root: Path) -> Dict[str, Any]:
    """Collect only CPU/storage/software facts; no CUDA device is required."""

    nvcc = _command_output([_nvcc_executable(), "--version"])
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
    if not model and lock.get("models"):
        model = next(iter(lock["models"].values()))
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
    legacy_minimum = platform_lock.get("minimum_data_disk_free_gib")
    if legacy_minimum is not None and float(
        host.get("data_disk_free_gib", 0.0)
    ) < float(legacy_minimum):
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
    if manifest_model.get("adapter_name") not in (None, model.get("adapter_name")):
        failures.append("job manifest adapter does not match the server lock")
    if model_audit.get("tokenizer_hash") and manifest_model.get(
        "tokenizer_hash"
    ) != model_audit.get("tokenizer_hash"):
        failures.append("job manifest tokenizer hash differs from model audit")
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
    manifest_tree = manifest_cacheblend.get("tree")
    if manifest_tree not in (None, "pending-patch-audit") and manifest_tree != patch_audit.get("cacheblend_tree"):
        failures.append("CacheBlend tree differs from the frozen job manifest")

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


def evaluate_dual_model_no_gpu_readiness(
    lock: Mapping[str, Any],
    job_manifests: Mapping[str, Mapping[str, Any]],
    host: Mapping[str, Any],
    storage: Mapping[str, Any],
    model_audits: Mapping[str, Mapping[str, Any]],
    patch_audit: Mapping[str, Any],
    runtime_source_audit: Mapping[str, Any],
    *,
    expected_code_commit: str,
    actual_hashes_by_model: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Final CPU-only gate for the sequential Mistral/Qwen handoff."""

    failures: List[str] = []
    models = lock.get("models", {})
    expected_keys = {"mistral", "qwen"}
    if set(models) != expected_keys:
        failures.append("server lock must define exactly Mistral and Qwen")
    if storage.get("storage_ready") is not True:
        failures.extend("storage: %s" % value for value in storage.get("failures", ()))
    per_model: Dict[str, Any] = {}
    for key in sorted(expected_keys):
        if key not in job_manifests or key not in model_audits:
            failures.append("missing %s manifest or model audit" % key)
            continue
        derived_lock = dict(lock)
        derived_lock["model"] = models.get(key, {})
        result = evaluate_no_gpu_readiness(
            derived_lock,
            job_manifests[key],
            host,
            model_audits[key],
            patch_audit,
            expected_code_commit=expected_code_commit,
            actual_hashes=actual_hashes_by_model[key],
        )
        per_model[key] = result
        failures.extend("%s: %s" % (key, value) for value in result["failures"])
    source_ready = runtime_source_audit.get("runtime_source_ready") is True
    if not source_ready:
        failures.extend(
            "runtime-source: %s" % value
            for value in runtime_source_audit.get("failures", ("audit failed",))
        )
    artifact_ready = not failures
    return {
        "schema_version": 2,
        "stage": "v6_dual_model_no_gpu_readiness",
        "paper_evidence": False,
        "expected_code_commit": expected_code_commit,
        "storage_mode": storage.get("storage_mode"),
        "artifact_preparation_ready": artifact_ready,
        "mistral_runtime_source_ready": source_ready and "mistral" in per_model,
        "qwen_runtime_source_ready": source_ready and "qwen" in per_model,
        "gpu_rental_ready_for_runtime_qualification": artifact_ready and source_ready,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "per_model": per_model,
        "failures": failures,
    }
