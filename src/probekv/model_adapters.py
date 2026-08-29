from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .resumable_prefill import LayerAdvanceResult
from .model_signature import RuntimeModelSignature


MISTRAL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


@dataclass(frozen=True)
class ResumableModelSpec:
    adapter_name: str
    model_id: str
    revision: str
    architecture: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    rope_theta: float
    rope_scaling: Any
    sliding_window: Any
    use_sliding_window: bool
    qkv_bias: bool
    checkpoints: Tuple[int, ...]
    max_context_tokens: int

    def validate_config(self, config: Mapping[str, Any]) -> None:
        observed_architectures = tuple(config.get("architectures", ()))
        if self.architecture not in observed_architectures:
            raise ValueError("model architecture does not match adapter")
        checks = {
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_kv_heads,
            "rope_theta": self.rope_theta,
        }
        for key, expected in checks.items():
            if config.get(key) != expected:
                raise ValueError("%s does not match adapter: %r" % (key, config.get(key)))
        if config.get("rope_scaling") != self.rope_scaling:
            raise ValueError("rope_scaling does not match adapter")
        observed_sliding = config.get("sliding_window")
        if self.adapter_name == "qwen2_5_vllm041":
            if config.get("use_sliding_window") is not False:
                raise ValueError("Qwen main model must disable sliding window")
        elif observed_sliding != self.sliding_window:
            raise ValueError("sliding_window does not match adapter")


MISTRAL_SPEC = ResumableModelSpec(
    adapter_name="mistral_cacheblend_llama_v041",
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    revision=MISTRAL_REVISION,
    architecture="MistralForCausalLM",
    num_layers=32,
    num_attention_heads=32,
    num_kv_heads=8,
    rope_theta=1000000.0,
    rope_scaling=None,
    sliding_window=None,
    use_sliding_window=False,
    qkv_bias=False,
    checkpoints=(1, 2, 4, 6, 8),
    max_context_tokens=32768,
)

QWEN_SPEC = ResumableModelSpec(
    adapter_name="qwen2_5_vllm041",
    model_id="Qwen/Qwen2.5-7B-Instruct",
    revision=QWEN_REVISION,
    architecture="Qwen2ForCausalLM",
    num_layers=28,
    num_attention_heads=28,
    num_kv_heads=4,
    rope_theta=1000000.0,
    rope_scaling=None,
    sliding_window=None,
    use_sliding_window=False,
    qkv_bias=True,
    checkpoints=(1, 2, 4, 5, 7),
    max_context_tokens=32768,
)

MODEL_SPECS = {spec.model_id: spec for spec in (MISTRAL_SPEC, QWEN_SPEC)}

# Historical v6/v7 manifests keep the original Mistral checkpoint at depth 6.
# Schema-v6 is an explicit new runtime contract and uses depth 5 instead.
MISTRAL_SCHEMA6_CHECKPOINTS = (1, 2, 4, 5, 8)
QWEN_SCHEMA6_CHECKPOINTS = (1, 2, 4, 5, 7)
MISTRAL_SCHEMA6_SPEC = replace(
    MISTRAL_SPEC,
    adapter_name="mistral_cacheblend_llama_v041_schema6",
    checkpoints=MISTRAL_SCHEMA6_CHECKPOINTS,
)
QWEN_SCHEMA6_SPEC = replace(
    QWEN_SPEC,
    adapter_name="qwen2_5_vllm041_schema6",
    checkpoints=QWEN_SCHEMA6_CHECKPOINTS,
)
SCHEMA6_MODEL_SPECS = {
    spec.model_id: spec for spec in (MISTRAL_SCHEMA6_SPEC, QWEN_SCHEMA6_SPEC)
}


def validate_schema6_checkpoint_contract(
    *, model_id: str, checkpoint_sources: Mapping[str, Sequence[int]]
) -> Tuple[int, ...]:
    spec = SCHEMA6_MODEL_SPECS.get(model_id)
    if spec is None:
        raise ValueError("unsupported schema-v6 model")
    expected = spec.checkpoints
    if not checkpoint_sources:
        raise ValueError("schema-v6 checkpoint validator requires sources")
    for source_name, checkpoints in checkpoint_sources.items():
        observed = tuple(int(value) for value in checkpoints)
        if observed != expected:
            raise ValueError(
                "schema-v6 checkpoints differ for %s: %r != %r"
                % (source_name, observed, expected)
            )
    return expected


def tokenizer_assets_hash(snapshot: Path) -> str:
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    )
    digest = hashlib.sha256()
    found = 0
    for name in names:
        path = snapshot / name
        if path.is_file():
            found += 1
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    if not found:
        raise ValueError("snapshot contains no tokenizer assets")
    return digest.hexdigest()


def runtime_model_signature(
    spec: ResumableModelSpec,
    *,
    tokenizer_hash: str,
    dtype: str,
    runtime_patch_sha: str,
) -> str:
    if not tokenizer_hash or not dtype or not runtime_patch_sha:
        raise ValueError("complete model signature inputs are required")
    return RuntimeModelSignature(
        model_id=spec.model_id,
        revision=spec.revision,
        architecture=spec.architecture,
        tokenizer_hash=tokenizer_hash,
        num_layers=spec.num_layers,
        num_attention_heads=spec.num_attention_heads,
        num_kv_heads=spec.num_kv_heads,
        rope_theta=spec.rope_theta,
        rope_scaling=spec.rope_scaling,
        sliding_window=spec.sliding_window,
        use_sliding_window=spec.use_sliding_window,
        dtype=dtype,
        runtime_patch_sha=runtime_patch_sha,
    ).encode()


class PinnedCacheBlendResumableAdapter:
    """Concrete bridge to methods added by the frozen CacheBlend patch.

    This class deliberately fails on an unpatched model instead of silently
    falling back to a monolithic ``generate()`` call.
    """

    def __init__(self, inner_model: Any, spec: ResumableModelSpec) -> None:
        self.inner_model = inner_model
        self.spec = spec
        self.adapter_name = spec.adapter_name
        self.total_layers = spec.num_layers
        for name in (
            "probekv_begin_prefill",
            "probekv_advance_prefill",
            "probekv_finish_prefill",
        ):
            if not callable(getattr(inner_model, name, None)):
                raise RuntimeError("patched model is missing %s" % name)

    def begin_prefill(
        self,
        *,
        token_ids: Sequence[int],
        absolute_positions: Sequence[int],
        attention_metadata: Any,
        working_kv: Any,
        model_signature: str,
    ) -> Tuple[Any, Any]:
        return self.inner_model.probekv_begin_prefill(
            token_ids,
            absolute_positions,
            attention_metadata,
            working_kv,
            model_signature,
        )

    def advance_layer(self, **kwargs: Any) -> LayerAdvanceResult:
        row = self.inner_model.probekv_advance_prefill(**kwargs)
        if isinstance(row, LayerAdvanceResult):
            return row
        if not row.get("union_mask_digest"):
            payload = json.dumps(
                list(kwargs["target_active_positions"]), separators=(",", ":")
            ).encode("ascii")
            row["union_mask_digest"] = hashlib.sha256(payload).hexdigest()
        return LayerAdvanceResult(**row)

    def finish_prefill(self, **kwargs: Any) -> Any:
        return self.inner_model.probekv_finish_prefill(**kwargs)

    def observe_pre_rope_k(
        self,
        *,
        completed_depth: int,
        hidden_states: Any,
        residual: Any,
        active_positions: Tuple[int, ...],
    ) -> Any:
        """Project exact K entering block ``completed_depth + 1``.

        The pinned runtime is single-GPU, so the local Q/K/V projection sizes
        equal the model geometry. No RoPE is applied here.
        """
        del active_positions
        if not 0 <= completed_depth < self.total_layers:
            raise ValueError("completed depth has no following Transformer block")
        layer = self.inner_model.layers[completed_depth]
        # vLLM's fused RMSNorm may update both ``hidden_states`` and
        # ``residual`` in place.  A Source-selection observation is read-only:
        # mutating the live prefill state here changes the input to the next
        # Transformer block and breaks the r=1 dense-equivalence endpoint.
        # Isolate the projection on private tensors rather than relying on a
        # particular RMSNorm implementation being out-of-place.
        observed_hidden = hidden_states.clone()
        observed_residual = None if residual is None else residual.clone()
        normalized = layer.input_layernorm(observed_hidden, observed_residual)
        if isinstance(normalized, tuple):
            normalized = normalized[0]
        projected = layer.self_attn.qkv_proj(normalized)
        if isinstance(projected, tuple):
            projected = projected[0]
        attention = layer.self_attn
        head_dim = int(getattr(attention, "head_dim"))
        q_size = int(
            getattr(attention, "q_size", self.spec.num_attention_heads * head_dim)
        )
        kv_size = int(
            getattr(attention, "kv_size", self.spec.num_kv_heads * head_dim)
        )
        if projected.shape[-1] < q_size + kv_size:
            raise RuntimeError("QKV projection is smaller than the pinned model geometry")
        key = projected[..., q_size : q_size + kv_size]
        return key.reshape(key.shape[0], self.spec.num_kv_heads, head_dim)


def model_audit_contract(spec: ResumableModelSpec) -> Mapping[str, Any]:
    return asdict(spec)
