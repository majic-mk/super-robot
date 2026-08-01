from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass


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


def validate_v6_model_signature(value: str) -> None:
    ModelSignature.decode(value)
