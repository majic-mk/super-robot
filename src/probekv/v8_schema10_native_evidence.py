"""Recompute native primitive results from saved observations, never passed flags."""
import json
from pathlib import Path

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest
from .v8_schema10_native_validation import validate_correctness_observation


def read_signed_json(path, signature_key):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in row.items() if k != signature_key}
    if row.get(signature_key) != digest_json(unsigned):
        raise ValueError("native observation digest mismatch: " + str(path))
    return row


def load_saved_logits(root, row):
    import torch
    name = row.get("logits_path")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("native logit locator must be a local artifact filename")
    path = Path(root) / name
    if path.is_symlink() or file_digest(path) != row.get("logits_sha256"):
        raise ValueError("native raw logit digest mismatch")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
            or list(tensor.shape) != row.get("logits_shape") or tensor.shape[0] < 32
            or not torch.isfinite(tensor).all()):
        raise ValueError("native raw logit geometry/values invalid")
    return tensor.float()


def verify_native_primitive_directory(directory):
    import torch
    root = Path(directory)
    if (root / "failed.json").exists():
        raise ValueError("failed run cannot become successful native evidence")
    manifest = read_signed_json(root / "manifest.json", "manifest_sha256")
    if (manifest.get("stage") != "native_correctness_diagnostic"
            or manifest.get("paper_evidence") is not False
            or manifest.get("locked_test_accessed") is not False):
        raise ValueError("unexpected native diagnostic stage/scope")
    prefix = read_signed_json(root / "prefix.json", "raw_observation_sha256")
    hook = read_signed_json(root / "k_hook.json", "raw_observation_sha256")
    validate_correctness_observation("native_prefix", prefix)
    validate_correctness_observation("k_hook", hook)
    for row in (prefix, hook):
        if row.get("source_provenance") != manifest["native_runtime"]["source_provenance"]:
            raise ValueError("native primitive belongs to another code/model runtime")
    cfo = json.loads((root / "cfo.json").read_text(encoding="utf-8"))
    validate_correctness_observation("cfo", cfo)
    arm_root = root / "combined-r1"
    names = ("dense_free", "reuse_free", "dense_teacher", "reuse_teacher",
             "native_prefix_teacher", "resumable_prefix_teacher")
    arms = {name: read_signed_json(arm_root / (name + ".json"), "raw_observation_sha256") for name in names}
    target_hash = digest_json(manifest["diagnostic_requests"]["target"]["token_ids"])
    teacher_hash = digest_json(manifest["diagnostic_requests"]["teacher_token_ids"])
    boundary = manifest.get("diagnostic_reuse_boundary", 2)
    for name, row in arms.items():
        if (row.get("request_tokens_sha256") != target_hash
                or row.get("origin") != "real_cuda_execution" or row.get("fake_timing") is not False
                or name.endswith("teacher") and row.get("teacher_tokens_sha256") != teacher_hash):
            raise ValueError("native arm input/provenance mismatch")
        if name.startswith("reuse"):
            if (row.get("committed_segments") != {"C": boundary}
                    or row.get("whole_request_origin") != "selective_reuse"
                    or row.get("cached_prefix_tokens") != prefix["cached_prefix_tokens"]
                    or len(row.get("layer_rows", [])) != prefix["model_layers"]):
                raise ValueError("native reuse did not execute the declared Prefix/Source path")
            validate_correctness_observation("source_digest", row)
            validate_correctness_observation("absolute_mask", row)
            expected = list(range(prefix["cached_prefix_tokens"],
                                  len(manifest["diagnostic_requests"]["target"]["token_ids"])))
            if any(layer["active_positions"] != expected for layer in row["layer_rows"]):
                raise ValueError("r1 active mask differs from immutable request positions")
    tensors = {name: load_saved_logits(arm_root, arms[name]) for name in names if name.endswith("teacher")}
    left, right = tensors["dense_teacher"], tensors["reuse_teacher"]
    if left.shape != right.shape:
        raise ValueError("native dense/reuse logit shape mismatch")
    l2 = float((left - right).norm() / left.norm().clamp_min(1e-12))
    recomputed = dict(origin="real_cuda_execution", fake_timing=False,
        dense_token_ids=arms["dense_free"]["token_ids"], reuse_token_ids=arms["reuse_free"]["token_ids"],
        logit_relative_l2=l2, logit_token_count=left.shape[0])
    validate_correctness_observation("r1", recomputed)
    transfer = json.loads((root / "transfer.json").read_text(encoding="utf-8"))
    path = "SSD_STAGED_TO_GPU" if manifest["diagnostic_backing_tier"] == "ssd" else "CPU_PINNED_TO_GPU"
    winner = arms["reuse_teacher"]["source_id"]
    if (not transfer.get("events") or transfer.get("active_hbm_reserved_bytes") != 0
            or transfer.get("origin") != "real_cuda_execution" or transfer.get("fake_timing") is not False
            or not 0 <= transfer["staging_peak_bytes"] <= transfer["staging_capacity_bytes"]
            or any(e.get("path") != path or e.get("source_id") != winner or e.get("full_kv_bytes", 0) <= 0
                   for e in transfer["events"])):
        raise ValueError("native transfer path/resources mismatch")
    files = [root / n for n in ("manifest.json", "prefix.json", "k_hook.json", "cfo.json", "transfer.json")]
    files += [arm_root / (n + ".json") for n in names]
    files += [arm_root / arms[n]["logits_path"] for n in tensors]
    return {"binding": manifest["binding"], "raw_files_sha256": {
                str(p.relative_to(root)): file_digest(p) for p in files},
        "native_prefix_k_hook_r1_passed": True, "native_cfo_eager_streaming_passed": True,
        "logit_relative_l2_recomputed": l2, "raw_logits_bitwise_equal": torch.equal(left, right),
        "cached_prefix_tokens": prefix["cached_prefix_tokens"], "reuse_boundary": boundary,
        "transfer_path": path, "source_digest_validation": "device-capture attestations; backing not rehashed by verifier",
        "online_trace_execution_allowed": False, "gpu_runtime_qualified": False,
        "paper_evidence": False, "locked_test_accessed": False}
