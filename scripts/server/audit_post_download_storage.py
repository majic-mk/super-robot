"""Verify steady-state storage after selective model downloads."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from probekv.io import atomic_write_json


GIB = float(1024**3)


def audit_post_download_storage(stage_root: Path, system_root: Path) -> dict:
    stage = stage_root.resolve()
    system = system_root.resolve()
    stage_usage = shutil.disk_usage(stage)
    system_usage = shutil.disk_usage(system)
    system_free = system_usage.free / GIB
    failures = []
    if system_free < 15.0:
        failures.append("system filesystem reserve is below 15 GiB")
    return {
        "schema_version": 1,
        "stage_root": str(stage),
        "stage_filesystem_free_gib": stage_usage.free / GIB,
        "system_free_gib": system_free,
        "system_reserve_gib": 15,
        "steady_state_ready": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--system-root", default="/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_post_download_storage(
        Path(args.stage_root), Path(args.system_root)
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["steady_state_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
