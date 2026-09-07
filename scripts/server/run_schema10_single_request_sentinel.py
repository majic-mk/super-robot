"""Validate or execute an immutable manifest; validation never imports vLLM."""
import argparse
import json
from pathlib import Path

from probekv.v8_schema10_single_request_sentinel import run_manifest, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--instance-confirmed", action="store_true")
    parser.add_argument("--hourly-yuan", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if not args.execute:
        print(json.dumps({"manifest_valid": True, "gpu_started": False,
                          "ready_for_single_request_gpu_sentinel": False,
                          "pending": manifest.get("readiness", {}).get("pending", [])}))
        return
    if args.hourly_yuan is None or not args.output_dir:
        parser.error("execution requires --hourly-yuan and --output-dir")
    print(json.dumps(run_manifest(manifest, output_dir=args.output_dir, hourly_yuan=args.hourly_yuan,
        instance_confirmed=args.instance_confirmed, repository=Path(__file__).resolve().parents[2], resume=args.resume)))


if __name__ == "__main__": main()
