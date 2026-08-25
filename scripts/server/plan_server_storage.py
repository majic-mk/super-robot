"""Audit the 130GB host before any model download or GPU rental."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.server_storage import collect_storage, evaluate_storage_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--system-root", default="/")
    parser.add_argument("--additional-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = [Path(args.stage_root), *(Path(value) for value in args.additional_root)]
    result = evaluate_storage_plan(
        collect_storage(roots, Path(args.system_root))
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["storage_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
