"""Validate CacheBlend v5 closed-loop audit records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.closed_loop_audit import audit_cacheblend_closed_loop
from probekv.io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.records)
        .resolve()
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    result = audit_cacheblend_closed_loop(rows)
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
