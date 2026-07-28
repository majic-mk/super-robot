"""Record local hardware and framework compatibility without mutating it."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


def command_output(command):
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, universal_newlines=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return getattr(error, "output", "") or str(error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import torch

        record.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_visible": torch.cuda.is_available(),
                "torch_arch_list": list(torch.cuda.get_arch_list())
                if torch.cuda.is_available()
                else [],
            }
        )
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            target = "sm_%d%d" % (major, minor)
            record["gpu_name"] = torch.cuda.get_device_name(0)
            record["gpu_compute_capability"] = target
            record["framework_supports_gpu"] = target in record["torch_arch_list"]
        else:
            record["framework_supports_gpu"] = False
    except ImportError:
        record["torch"] = None
        record["framework_supports_gpu"] = False
    record["paper_performance_ready"] = False
    record["note"] = (
        "Local hardware is for correctness/smoke tests only; paper timing requires the config-frozen A800."
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
