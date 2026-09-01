from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .v8_schema9_contracts import AbsoluteResidualThreshold


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class VariantAdmissionProfile:
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    source_score_trim_ratio: float
    thresholds: Tuple[AbsoluteResidualThreshold, ...]
    require_full_candidate_coverage_for_mismatch: bool = True
    materialization_budget_fraction: float = 0.02
    max_variants_per_content: int = 16
    canonical_source_provenance: str = "dense_exact"
    development_partition_sha256: str = ""
    frozen: bool = False
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        if not all(
            (
                self.code_commit,
                self.cacheblend_patch_sha256,
                self.model_id,
                self.model_revision,
                self.tokenizer_hash,
            )
        ):
            raise ValueError("VariantAdmissionProfile provenance is incomplete")
        if self.source_score_trim_ratio not in {0.0, 0.10, 0.15, 0.30, 0.40}:
            raise ValueError("source trim ratio is outside the schema9 grid")
        depths = tuple(row.completed_depth for row in self.thresholds)
        if depths != tuple(sorted(set(depths))) or not set(depths).issubset({1, 2}):
            raise ValueError("schema9 thresholds must uniquely cover sorted d1/d2")
        if not 0 <= self.materialization_budget_fraction <= 0.05:
            raise ValueError("materialization budget must be within 5% of dense")
        if not 1 <= self.max_variants_per_content <= 16:
            raise ValueError("schema9 supports 1-16 Variants per content")
        if self.canonical_source_provenance != "dense_exact":
            raise ValueError("schema9 canonical Sources require exact dense prefill")
        if self.frozen:
            if not self.development_partition_sha256:
                raise ValueError("frozen VariantAdmissionProfile needs development data")
            if self.profile_sha256 != _digest(self.payload(include_sha=False)):
                raise ValueError("VariantAdmissionProfile SHA differs")

    def threshold_for_depth(self, completed_depth: int) -> float:
        for row in self.thresholds:
            if row.completed_depth == completed_depth:
                return row.upper_residual
        raise KeyError("VariantAdmissionProfile has no threshold for this depth")

    def payload(self, *, include_sha: bool = True) -> Mapping[str, Any]:
        payload = {
            "protocol_version": 8,
            "schema_version": 9,
            "code_commit": self.code_commit,
            "cacheblend_patch_sha256": self.cacheblend_patch_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_hash": self.tokenizer_hash,
            "source_score_trim_ratio": self.source_score_trim_ratio,
            "thresholds": [row.__dict__ for row in self.thresholds],
            "require_full_candidate_coverage_for_mismatch": (
                self.require_full_candidate_coverage_for_mismatch
            ),
            "materialization_budget_fraction": self.materialization_budget_fraction,
            "max_variants_per_content": self.max_variants_per_content,
            "canonical_source_provenance": self.canonical_source_provenance,
            "development_partition_sha256": self.development_partition_sha256,
            "frozen": self.frozen,
        }
        if include_sha:
            payload["profile_sha256"] = self.profile_sha256
        return payload


def build_variant_admission_profile(
    **kwargs: Any,
) -> VariantAdmissionProfile:
    if not kwargs.get("frozen", False):
        return VariantAdmissionProfile(**kwargs)
    base = dict(kwargs)
    base.pop("profile_sha256", None)
    provisional = VariantAdmissionProfile(**{**base, "frozen": False})
    payload = dict(provisional.payload(include_sha=False))
    payload["frozen"] = True
    return VariantAdmissionProfile(
        **{**base, "frozen": True, "profile_sha256": _digest(payload)}
    )
