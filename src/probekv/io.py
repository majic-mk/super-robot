from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: %r" % (value,))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact after a successful stage.

    Experiment stages can be interrupted by preemption or a workstation
    shutdown.  Writing to a sibling temporary file first prevents a truncated
    JSON file from being mistaken for a completed stage during resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            handle.write("\n")


def git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def environment_manifest(
    cwd: Path,
    seed: int,
    evidence_class: str,
    model_signature: str,
    data_manifest_hash: str,
    config_hash: str,
) -> Dict[str, Any]:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(cwd),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "evidence_class": evidence_class,
        "paper_evidence": evidence_class == "paper_measurement",
        "model_signature": model_signature,
        "data_manifest_hash": data_manifest_hash,
        "config_hash": config_hash,
    }
    stable_fields = {
        key: value for key, value in payload.items() if key != "timestamp_utc"
    }
    stable = json.dumps(stable_fields, sort_keys=True).encode("utf-8")
    payload["environment_hash"] = hashlib.sha256(stable).hexdigest()
    return payload


def try_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Best-effort Parquet output; JSONL remains the dependency-free record."""
    try:
        import pandas as pd  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(str(path), index=False)
        return True
    except (ImportError, ModuleNotFoundError, ValueError, OSError):
        return False
