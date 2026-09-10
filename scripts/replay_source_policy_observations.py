"""Replay candidate source policies from a complete d1/d2 observation bundle.

No model load, CUDA execution, pool mutation or runtime admission occurs here.
"""
import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.source_policy_replay import replay_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if target.exists():
        raise FileExistsError("retain previous evidence; choose a new output path")
    # Read once so the verified bytes are exactly the parsed bytes.
    import hashlib
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise ValueError("input file digest mismatch")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    report = replay_observation(json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys))
    atomic_write_json(target, {"input_file_sha256": args.input_sha256, "report": report})
    print(json.dumps({"output": str(target), "output_file_sha256": sha256_file(target),
                      "gpu_execution_allowed": False}))


if __name__ == "__main__":
    main()
