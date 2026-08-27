from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_a800_jobs import build_v8_qualification_matrix_index


def main() -> int:
    parser = argparse.ArgumentParser()
    for key in (
        "mistral-causal-wait",
        "mistral-immediate-staggered",
        "qwen-causal-wait",
        "qwen-immediate-staggered",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        key.replace("-", "_"): Path(getattr(args, key.replace("-", "_"))).resolve()
        for key in (
            "mistral-causal-wait",
            "mistral-immediate-staggered",
            "qwen-causal-wait",
            "qwen-immediate-staggered",
        )
    }
    result = build_v8_qualification_matrix_index(
        {
            key: json.loads(path.read_text(encoding="utf-8"))
            for key, path in paths.items()
        }
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
