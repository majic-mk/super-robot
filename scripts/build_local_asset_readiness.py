"""Build a dual-model asset readiness report without opening model weights."""
import argparse
import json
from pathlib import Path

try:
    from .audit_local_assets import audit_asset_root
except ImportError:  # direct script execution
    from audit_local_assets import audit_asset_root


def build_readiness(models, partition):
    rows = []
    for name, root in models.items():
        expected = {"mistral": "Mistral-7B-Instruct-v0.3", "qwen": "Qwen2.5-7B-Instruct"}.get(name)
        rows.append({"model": name, **audit_asset_root(root, partition=partition, expected_model_id=expected)})
    return {"kind": "probekv_local_asset_readiness_v1", "models": rows,
            "all_models_ready": bool(rows) and all(row["ready"] for row in rows),
            "paper_evidence": False, "locked_test_accessed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral-root", required=True)
    parser.add_argument("--qwen-root", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("fresh readiness output required")
    report = build_readiness({"mistral": args.mistral_root, "qwen": args.qwen_root}, args.partition)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_models_ready": report["all_models_ready"],
                      "paper_evidence": False, "locked_test_accessed": False}))
    return 0 if report["all_models_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
