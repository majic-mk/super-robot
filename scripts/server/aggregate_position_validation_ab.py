"""Matched host/device index-validation control; never online gain evidence."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from probekv.io import atomic_write_json
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest
from scripts.server.run_schema10_native_correctness import position_validation_pair_specs


def aggregate(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    specs = manifest["specs"]
    if specs != position_validation_pair_specs(len(specs) - 2):
        raise ValueError("paired schedule differs from excluded warmups/alternating order")
    rows = []
    for spec in specs:
        paths = {arm: root / ("pair-%02d-%s.json" % (spec["pair"], arm))
                 for arm in ("host", "device")}
        arms = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
        for arm in arms.values():
            if arm.get("origin") != "real_cuda_execution" or arm.get("fake_timing") is not False:
                raise ValueError("paired arm is not native execution evidence")
            claimed = arm.get("raw_observation_sha256")
            if claimed is not None and claimed != digest_json({k: v for k, v in arm.items()
                                                              if k != "raw_observation_sha256"}):
                raise ValueError("paired arm digest mismatch")
        host, device = arms["host"], arms["device"]
        for field in ("request_tokens_sha256", "cached_prefix_tokens", "prefix_cache_mode",
                      "sampling_signature", "diagnostic_completed_depth", "token_ids"):
            if field not in host or host[field] != device.get(field):
                raise ValueError("paired condition mismatch: " + field)
        if host["request_tokens_sha256"] != digest_json(manifest["request"]["token_ids"]):
            raise ValueError("request differs from preregistered tokens")
        if any(arm.get("source_id") is not None or arm.get("committed_segments") for arm in arms.values()):
            raise ValueError("fallback control must not use a Source or commit reuse")
        masks = lambda arm: [(r["layer"], r["expected_positions"]) for r in arm["layer_rows"]]
        if not host.get("layer_rows") or masks(host) != masks(device):
            raise ValueError("paired execution mask differs")
        times = {}
        for name, arm in arms.items():
            start, end = arm["diagnostic_start_ns"], arm["first_token_ns"]
            if type(start) is not int or type(end) is not int or end <= start:
                raise ValueError("invalid request timing endpoint")
            elapsed = (end - start) / 1e6
            if abs(elapsed - arm["first_token_host_ms"]) > 1e-6:
                raise ValueError("TTFT differs from actual endpoints")
            times[name] = elapsed
        rows.append({"pair": spec["pair"], "warmup": spec["warmup"],
            "host_ms": times["host"], "device_ms": times["device"],
            "host_minus_device_ms": times["host"] - times["device"],
            "arm_file_sha256": {name: file_digest(path) for name, path in paths.items()},
            "producer_embedded_digest_present": {name: "raw_observation_sha256" in arm
                                                  for name, arm in arms.items()}})
    measured = [row for row in rows if not row["warmup"]]
    report = {"kind": "position_validation_matched_control_v1",
        "code_commit": manifest["code_commit"], "patch_sha256": manifest["patch_sha256"],
        "manifest_digest": digest_json(manifest), "measured_pair_count": len(measured),
        "host_mean_ms": mean(r["host_ms"] for r in measured),
        "device_mean_ms": mean(r["device_ms"] for r in measured),
        "paired_mean_delta_ms": mean(r["host_minus_device_ms"] for r in measured),
        "paired_median_delta_ms": median(r["host_minus_device_ms"] for r in measured),
        "rows": rows, "scope": "resumable_dense_control_not_live_selector_or_reuse",
        "online_gain_proven": False, "paper_evidence": False}
    report["report_sha256"] = digest_json(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fresh aggregate output required")
    atomic_write_json(output, aggregate(args.input))
