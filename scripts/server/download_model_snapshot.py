"""Selectively download, smoke-test and audit a frozen HF model revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.model_adapters import (
    MISTRAL_SPEC,
    MODEL_SPECS,
    QWEN_SPEC,
    tokenizer_assets_hash,
)


MODEL_ALLOWLISTS = {
    MISTRAL_SPEC.model_id: (
        "config.json", "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "tokenizer.json", "tokenizer.model", "tokenizer.model.v3",
        "tokenizer_config.json", "special_tokens_map.json",
    ),
    QWEN_SPEC.model_id: (
        "config.json", "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
    ),
}


def audit_snapshot(snapshot: Path, model_id: str, revision: str) -> dict:
    resolved_revision = snapshot.resolve().name
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if len(revision) == 40 and resolved_revision != revision:
        missing.append("resolved snapshot revision %s" % revision)
    tokenizer_assets = tuple(
        name for name in
        ("tokenizer.json", "tokenizer.model", "tokenizer.model.v3")
        if (snapshot / name).is_file()
    )
    if not tokenizer_assets:
        missing.append("tokenizer.json|tokenizer.model|tokenizer.model.v3")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected_weights = tuple(sorted(set(index.get("weight_map", {}).values())))
        if not expected_weights:
            missing.append("non-empty model.safetensors.index.json weight_map")
        missing.extend(name for name in expected_weights if not (snapshot / name).is_file())
        weights = [snapshot / name for name in expected_weights if (snapshot / name).is_file()]
    else:
        weights = sorted(snapshot.glob("*.safetensors"))
        if not weights:
            missing.append("*.safetensors")
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    selected_hashes = {
        name: sha256_file(snapshot / name)
        for name in required + (("model.safetensors.index.json",) if index_path.is_file() else ())
        if (snapshot / name).is_file()
    }
    forbidden = [
        name for name in ("consolidated.safetensors",)
        if (snapshot / name).is_file()
    ]
    spec = MODEL_SPECS.get(model_id)
    adapter_errors = []
    tokenizer_hash = None
    config = {}
    if spec is not None and (snapshot / "config.json").is_file():
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        try:
            spec.validate_config(config)
        except ValueError as error:
            adapter_errors.append(str(error))
        try:
            tokenizer_hash = tokenizer_assets_hash(snapshot)
        except ValueError as error:
            adapter_errors.append(str(error))
    missing.extend("forbidden file: %s" % name for name in forbidden)
    missing.extend("adapter: %s" % value for value in adapter_errors)
    return {
        "schema_version": 2,
        "paper_evidence": False,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "weight_files": [path.name for path in weights],
        "tokenizer_assets": list(tokenizer_assets),
        "selected_file_sha256": selected_hashes,
        "adapter_name": spec.adapter_name if spec is not None else None,
        "tokenizer_hash": tokenizer_hash,
        "model_contract": {
            "architecture": spec.architecture,
            "num_layers": spec.num_layers,
            "num_attention_heads": spec.num_attention_heads,
            "num_kv_heads": spec.num_kv_heads,
            "rope_theta": spec.rope_theta,
            "rope_scaling": spec.rope_scaling,
            "sliding_window": spec.sliding_window,
            "configured_sliding_window": config.get("sliding_window"),
            "use_sliding_window": spec.use_sliding_window,
            "qkv_bias": spec.qkv_bias,
        } if spec is not None else None,
        "forbidden_files": forbidden,
        "missing": missing,
        "complete": not missing,
    }


def cpu_tokenizer_config_smoke(snapshot: Path, model_id: str) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    token_ids = tokenizer.encode("ProbeKV smoke", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer smoke produced no tokens")
    expected = MODEL_SPECS[model_id]
    if config.architectures is None or expected.architecture not in config.architectures:
        raise RuntimeError("AutoConfig architecture differs from frozen adapter")
    return {
        "passed": True,
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "smoke_token_count": len(token_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    if args.model_id not in MODEL_ALLOWLISTS:
        raise ValueError("model has no frozen selective-download allowlist")
    snapshot = Path(snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        cache_dir=str(Path(args.cache_dir).resolve()),
        local_files_only=args.local_files_only,
        allow_patterns=list(MODEL_ALLOWLISTS[args.model_id]),
    ))
    audit = audit_snapshot(snapshot, args.model_id, args.revision)
    try:
        audit["cpu_tokenizer_config_smoke"] = cpu_tokenizer_config_smoke(
            snapshot, args.model_id
        )
    except Exception as error:
        audit["cpu_tokenizer_config_smoke"] = {
            "passed": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        audit["missing"].append("CPU tokenizer/config smoke")
        audit["complete"] = False
    atomic_write_json(Path(args.output).resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
