from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelSignature:
    weights_revision: str
    tokenizer_revision: str
    rope_signature: str
    dtype: str
    runtime_signature: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.weights_revision,
                self.tokenizer_revision,
                self.rope_signature,
                self.dtype,
                self.runtime_signature,
            )
        ):
            raise ValueError("all v6 model-signature components are required")

    def encode(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        return "model_signature_v1:%s" % encoded

    @classmethod
    def decode(cls, value: str) -> "ModelSignature":
        prefix = "model_signature_v1:"
        if not value.startswith(prefix):
            raise ValueError("v6 requires a structured model_signature_v1")
        payload = value[len(prefix):]
        payload += "=" * (-len(payload) % 4)
        try:
            raw = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        except Exception as error:
            raise ValueError("invalid structured model signature") from error
        required = {
            "weights_revision",
            "tokenizer_revision",
            "rope_signature",
            "dtype",
            "runtime_signature",
        }
        if set(raw) != required:
            raise ValueError("structured model signature fields are incomplete")
        return cls(**raw)


@dataclass(frozen=True)
class RuntimeModelSignature:
    model_id: str
    revision: str
    architecture: str
    tokenizer_hash: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    rope_theta: float
    rope_scaling: Any
    sliding_window: Optional[int]
    use_sliding_window: bool
    dtype: str
    runtime_patch_sha: str

    def __post_init__(self) -> None:
        required = (
            self.model_id, self.revision, self.architecture,
            self.tokenizer_hash, self.dtype, self.runtime_patch_sha,
        )
        if not all(required):
            raise ValueError("runtime model signature text fields are required")
        if min(self.num_layers, self.num_attention_heads, self.num_kv_heads) <= 0:
            raise ValueError("runtime model geometry must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    def encode(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
        return "model_signature_v2:%s" % encoded.rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "RuntimeModelSignature":
        prefix = "model_signature_v2:"
        if not value.startswith(prefix):
            raise ValueError("not a runtime model_signature_v2")
        payload = value[len(prefix):]
        payload += "=" * (-len(payload) % 4)
        try:
            raw = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
            return cls(**raw)
        except Exception as error:
            raise ValueError("invalid runtime model signature") from error


def validate_v6_model_signature(value: str) -> None:
    if value.startswith("model_signature_v2:"):
        RuntimeModelSignature.decode(value)
    else:
        ModelSignature.decode(value)
