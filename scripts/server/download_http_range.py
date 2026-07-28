"""Resume one absolute HTTP byte range without silently restarting it."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple


CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


def parse_content_range(value: str) -> Tuple[int, int, Optional[int]]:
    match = CONTENT_RANGE.fullmatch(value.strip())
    if not match:
        raise ValueError("invalid Content-Range: %s" % value)
    start, end, total = match.groups()
    return int(start), int(end), None if total == "*" else int(total)


def download_range(
    url: str,
    start: int,
    end: int,
    output: Path,
    retries: int,
) -> None:
    if start < 0 or end < start:
        raise ValueError("invalid absolute byte range")
    expected_bytes = end - start + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    current_bytes = output.stat().st_size if output.exists() else 0
    if current_bytes > expected_bytes:
        raise ValueError("existing part is longer than its requested range")
    failures = 0
    while current_bytes < expected_bytes:
        absolute_start = start + current_bytes
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ProbeKV-research-artifact/1.0",
                "Range": "bytes=%d-%d" % (absolute_start, end),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(
                        "range server returned HTTP %d" % response.status
                    )
                returned_start, returned_end, _ = parse_content_range(
                    response.headers.get("Content-Range", "")
                )
                if returned_start != absolute_start or returned_end > end:
                    raise RuntimeError(
                        "range response mismatch: requested %d-%d, got %d-%d"
                        % (
                            absolute_start,
                            end,
                            returned_start,
                            returned_end,
                        )
                    )
                before = current_bytes
                with output.open("ab") as handle:
                    shutil.copyfileobj(response, handle, 8 * 1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
                current_bytes = output.stat().st_size
                if current_bytes <= before or current_bytes > expected_bytes:
                    raise RuntimeError("range append made invalid progress")
                failures = 0
                print(
                    "%s %d/%d"
                    % (output.name, current_bytes, expected_bytes),
                    flush=True,
                )
        except Exception:
            failures += 1
            if failures > retries:
                raise
            time.sleep(min(30, 2 * failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()
    download_range(
        args.url,
        args.start,
        args.end,
        Path(args.output).resolve(),
        args.retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
