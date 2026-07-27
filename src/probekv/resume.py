from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .io import atomic_write_json


class StageLedger:
    """Small auditable stage ledger for deterministic resume decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.state = {"schema_version": 1, "stages": {}}

    def completed(
        self, stage: str, fingerprint: str, required_outputs: Sequence[Path]
    ) -> bool:
        record = self.state.get("stages", {}).get(stage, {})
        return (
            record.get("fingerprint") == fingerprint
            and record.get("status") == "complete"
            and all(path.exists() for path in required_outputs)
        )

    def mark_complete(
        self,
        stage: str,
        fingerprint: str,
        outputs: Sequence[Path],
        details: Mapping[str, Any] = None,
    ) -> None:
        self.state.setdefault("stages", {})[stage] = {
            "status": "complete",
            "fingerprint": fingerprint,
            "outputs": [str(path) for path in outputs],
            "details": dict(details or {}),
        }
        atomic_write_json(self.path, self.state)
