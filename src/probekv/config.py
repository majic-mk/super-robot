from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .selector import default_probe_checkpoints


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    evidence_class: str
    seed: int
    cases: int
    total_layers: int
    online_kmax: int
    gamma: float
    probe_checkpoints: Tuple[int, ...]
    repair_ratios: Tuple[float, ...]
    output_dir: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        total_layers = int(raw.get("total_layers", 32))
        checkpoints = tuple(
            int(value)
            for value in raw.get(
                "probe_checkpoints", default_probe_checkpoints(total_layers)
            )
        )
        result = cls(
            name=str(raw["name"]),
            evidence_class=str(raw.get("evidence_class", "local_simulation")),
            seed=int(raw.get("seed", 20260726)),
            cases=int(raw.get("cases", 50)),
            total_layers=total_layers,
            online_kmax=int(raw.get("online_kmax", 4)),
            gamma=float(raw.get("gamma", 0.8)),
            probe_checkpoints=checkpoints,
            repair_ratios=tuple(float(value) for value in raw["repair_ratios"]),
            output_dir=str(raw.get("output_dir", "artifacts/local_smoke")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.evidence_class not in {
            "local_simulation",
            "server_pilot",
            "paper_measurement",
        }:
            raise ValueError("unsupported evidence_class")
        if self.cases <= 0:
            raise ValueError("cases must be positive")
        if not 1 <= self.online_kmax <= 4:
            raise ValueError("online_kmax must be in [1, 4]")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not self.probe_checkpoints:
            raise ValueError("probe checkpoints required")
        if self.probe_checkpoints[-1] > max(1, int(self.total_layers * 0.25)):
            raise ValueError("L_probe_max exceeds 25% of total layers")
        if any(not 0 <= ratio <= 1 for ratio in self.repair_ratios):
            raise ValueError("repair ratios must be in [0, 1]")


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return ExperimentConfig.from_mapping(json.load(handle))
