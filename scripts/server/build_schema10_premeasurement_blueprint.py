"""Build the immutable premeasurement blueprint from real development traces.

The input JSON is deliberately explicit. This command never tokenizes a prompt,
creates synthetic traces, or fills a runtime measurement SHA. It is intended to
be run on the server after the development partition and model audit files have
been transferred, before the first CUDA measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_single_request_sentinel import prepare_manifest
from probekv.v8_schema10_event_log import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON with trace_set, dispatches and binding")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite an immutable blueprint")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    for key in ("trace_set", "dispatches", "binding"):
        if key not in source:
            raise ValueError("blueprint input is missing " + key)
    binding = source["binding"]
    if any(binding.get(k) in (None, "", "pending", "pending-tokenizer-audit")
           for k in ("code_commit", "patch_sha256", "config_sha256", "model_signature",
                     "model_revision", "tokenizer_hash")):
        raise ValueError("blueprint must bind real code/model/tokenizer inputs")
    binding = {**binding, "runtime_measurement_sha256": None}
    blueprint = prepare_manifest(trace_set=source["trace_set"], dispatches=source["dispatches"],
                                binding=binding, backend_factory=source.get("backend_factory"),
                                measurement_pending=True)
    blueprint["input_sha256"] = digest_json(source)
    unsigned = {k: v for k, v in blueprint.items() if k != "manifest_sha256"}
    blueprint["manifest_sha256"] = digest_json(unsigned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, blueprint)
    print(json.dumps({"output": str(args.output), "manifest_sha256": blueprint["manifest_sha256"],
                      "measurement_pending": True, "online_trace_execution_allowed": False,
                      "gpu_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
