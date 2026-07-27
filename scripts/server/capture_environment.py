"""Capture an auditable server environment record without installing anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command):
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, universal_newlines=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return getattr(error, "output", "") or str(error)


def redact(value: str) -> str:
    value = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", value)
    return re.sub(
        r"(?i)(token|password|passwd|secret|key)=([^&\s]+)", r"\1=***", value
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pip_freeze = redact(command_output([sys.executable, "-m", "pip", "freeze"]))
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "pip_freeze": pip_freeze.splitlines(),
        "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode("utf-8")).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output.resolve()), "git_commit": record["git_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
