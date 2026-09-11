"""Audit local model/tokenizer/development inputs without loading a model.

The report is intentionally fail-closed: historical audit JSON is not treated as
the asset it describes, and missing inputs never receive placeholder hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_TOKENIZER = ("tokenizer.json", "tokenizer_config.json")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_asset_root(root, *, partition=None, expected_model_id=None):
    root = Path(root).resolve()
    result = {"root": str(root), "exists": root.is_dir(), "tokenizer": {},
              "model_config": None, "development_partition": None,
              "ready": False, "paper_evidence": False, "locked_test_accessed": False}
    if not root.is_dir():
        return result
    for name in REQUIRED_TOKENIZER:
        path = root / name
        result["tokenizer"][name] = {"exists": path.is_file(),
                                      "sha256": _sha(path) if path.is_file() else None}
    config = root / "config.json"
    result["model_config"] = {"exists": config.is_file(),
                               "sha256": _sha(config) if config.is_file() else None}
    if config.is_file() and expected_model_id:
        try:
            config_value = json.loads(config.read_text(encoding="utf-8"))
            result["model_config"]["model_type"] = config_value.get("model_type")
            result["model_config"]["_name_or_path"] = config_value.get("_name_or_path")
            result["model_config"]["identity_checked"] = (
                expected_model_id.lower() in str(config_value.get("_name_or_path", "")).lower()
                or expected_model_id.lower().split("/")[-1] in str(config_value.get("_name_or_path", "")).lower())
        except (ValueError, OSError):
            result["model_config"]["identity_checked"] = False
    if partition is not None:
        path = Path(partition).resolve()
        entry = {"path": str(path), "exists": path.is_file(),
                 "sha256": _sha(path) if path.is_file() else None}
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                entry["kind"] = value.get("kind")
                entry["locked_test_accessed"] = value.get("locked_test_accessed")
                entry["entries"] = len(value.get("entries", [])) if isinstance(value.get("entries"), list) else None
                entry["valid_shape"] = (value.get("kind") == "source_policy_capture_partition_v1"
                                         and value.get("locked_test_accessed") is False
                                         and isinstance(value.get("entries"), list) and bool(value["entries"]))
            except (OSError, ValueError, TypeError):
                entry["valid_shape"] = False
        result["development_partition"] = entry
    result["ready"] = (all(v["exists"] for v in result["tokenizer"].values())
                       and result["model_config"]["exists"]
                       and (not expected_model_id or result["model_config"].get("identity_checked", False))
                       and (partition is None or result["development_partition"].get("valid_shape", False)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--partition")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("fresh audit output required")
    report = audit_asset_root(args.model_root, partition=args.partition)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "ready": report["ready"],
                      "paper_evidence": False, "locked_test_accessed": False}))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
