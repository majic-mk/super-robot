"""Download only official train artifacts and freeze their provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file


def _revision(repository: str) -> str:
    output = subprocess.check_output(
        ["git", "ls-remote", repository, "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    revision = output.split()[0]
    if len(revision) != 40:
        raise RuntimeError("could not freeze official repository revision")
    return revision


def _download(url: str, destination: Path, dataset: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    if dataset == "musique":
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError("install gdown for the official MuSiQue file") from error
        result = gdown.download(url, str(temporary), quiet=False, fuzzy=True)
        if not result:
            raise RuntimeError("gdown did not download MuSiQue")
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ProbeKV-research-artifact/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    os.replace(str(temporary), str(destination))


def _extract_member(archive: Path, member_suffix: str, destination: Path) -> None:
    with zipfile.ZipFile(str(archive)) as bundle:
        matches = [
            name
            for name in bundle.namelist()
            if name == member_suffix or name.endswith("/" + member_suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one %s in %s" % (member_suffix, archive)
            )
        with bundle.open(matches[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", default="configs/h1_official_datasets.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--use-transport",
        action="store_true",
        help="use the frozen byte-transport URL while retaining official provenance",
    )
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for dataset, spec in sorted(registry.items()):
        selected_url = (
            spec.get("transport_url", spec["train_url"])
            if args.use_transport
            else spec["train_url"]
        )
        suffix = ".json"
        if selected_url.split("?", 1)[0].endswith(".zip"):
            suffix = ".zip"
        elif spec["archive_member"].endswith(".jsonl"):
            suffix = ".zip"
        archive = output / ("%s-official%s" % (dataset, suffix))
        if not archive.exists():
            _download(selected_url, archive, dataset)
        if suffix == ".zip":
            raw_suffix = Path(spec["archive_member"]).suffix
            raw = output / ("%s-train%s" % (dataset, raw_suffix))
            if not raw.exists():
                _extract_member(archive, spec["archive_member"], raw)
        else:
            raw = archive
        if (
            spec.get("expected_train_bytes") is not None
            and raw.stat().st_size != int(spec["expected_train_bytes"])
        ):
            raise RuntimeError("%s train byte count mismatch" % dataset)
        raw_sha256 = sha256_file(raw)
        if (
            spec.get("expected_train_sha256")
            and raw_sha256 != spec["expected_train_sha256"]
        ):
            raise RuntimeError("%s train SHA256 mismatch" % dataset)
        if (
            spec.get("expected_archive_sha256")
            and sha256_file(archive) != spec["expected_archive_sha256"]
        ):
            raise RuntimeError("%s archive SHA256 mismatch" % dataset)
        if (
            spec.get("expected_git_blob_sha1")
            and _git_blob_sha1(raw) != spec["expected_git_blob_sha1"]
        ):
            raise RuntimeError("%s Git blob identity mismatch" % dataset)
        records[dataset] = {
            **spec,
            "official_repository_revision": _revision(
                spec["official_repository"]
            ),
            "download_sha256": sha256_file(archive),
            "train_path": str(raw),
            "train_sha256": raw_sha256,
            "download_transport_url": selected_url,
            "used_transport_mirror": selected_url != spec["train_url"],
            "source_split": "train",
            "paper_evidence": False,
        }
    payload = {
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "datasets": records,
        "locked_test_downloaded": False,
        "paper_evidence": False,
        "evidence_class": "data_preparation",
    }
    atomic_write_json(output / "official_sources.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
