from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


GIB = float(1024**3)


def collect_storage(paths: Iterable[Path], system_root: Path) -> Dict[str, Any]:
    """Measure unique filesystems so two paths on one disk are not double-counted."""

    filesystems: Dict[str, Dict[str, Any]] = {}
    for raw in tuple(paths) + (system_root,):
        path = raw.resolve()
        path.mkdir(parents=True, exist_ok=True)
        stat = path.stat()
        key = str(getattr(stat, "st_dev", path.anchor))
        usage = shutil.disk_usage(path)
        filesystems.setdefault(
            key,
            {
                "path": str(path),
                "total_gib": usage.total / GIB,
                "free_gib": usage.free / GIB,
                "writable": os.access(path, os.W_OK),
            },
        )
    system_usage = shutil.disk_usage(system_root.resolve())
    writable = [row for row in filesystems.values() if row["writable"]]
    return {
        "schema_version": 1,
        "filesystems": list(filesystems.values()),
        "combined_free_gib": sum(row["free_gib"] for row in writable),
        "largest_writable_free_gib": max(
            (row["free_gib"] for row in writable), default=0.0
        ),
        "system_free_gib": system_usage.free / GIB,
    }


def evaluate_storage_plan(record: Mapping[str, Any]) -> Dict[str, Any]:
    combined = float(record.get("combined_free_gib", 0.0))
    largest = float(record.get("largest_writable_free_gib", 0.0))
    system = float(record.get("system_free_gib", 0.0))
    failures = []
    if combined < 70.0:
        failures.append("combined free space is below 70 GiB")
    if largest < 50.0:
        failures.append("largest writable filesystem is below 50 GiB free")
    if system < 15.0:
        failures.append("system filesystem reserve is below 15 GiB")
    if failures:
        mode = "stop_before_gpu_rental"
    elif combined >= 90.0:
        mode = "dual_model_resident"
    else:
        mode = "sequential_mistral_then_qwen"
    return {
        **dict(record),
        "storage_ready": not failures,
        "storage_mode": mode,
        "single_model_steady_budget_gib": 55,
        "dual_model_steady_budget_gib": 75,
        "build_peak_budget_gib": 90,
        "system_reserve_gib": 15,
        "failures": failures,
    }


def assert_safe_regenerable_cache(target: Path, stage_root: Path) -> Path:
    resolved = target.resolve()
    root = stage_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("cleanup target must remain inside ProbeKV stage") from error
    if resolved == root or resolved.parent == root:
        raise ValueError("cleanup target is too broad")
    return resolved
