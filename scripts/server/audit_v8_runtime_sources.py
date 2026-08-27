from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.runtime_source_audit import audit_v8_runtime_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_v8_runtime_sources(Path(args.repo).resolve())
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["runtime_source_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
