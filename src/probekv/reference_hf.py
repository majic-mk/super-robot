from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReferenceLayerState:
    """Detached segment state from one full Hugging Face prefill."""

    cached_key: Any
    cached_value: Any
    key_pre_rope: Optional[Any]
    hidden: Any
    query: Optional[Any]


@dataclass(frozen=True)
class ReferencePrefill:
    prefix_token_count: int
    segment_token_ids: Tuple[int, ...]
    layer_states: Mapping[int, ReferenceLayerState]


def _legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _model_layers(model):
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise RuntimeError("unsupported model layout: transformer layers were not found")


class HuggingFaceReferenceStateBackend:
    """Slow correctness backend for canonical full-prefill state extraction.

    This deliberately does not implement ``RepairBackend``.  It can validate
    token boundaries and probe features, but it cannot stand in for CacheBlend
    selective repair or produce paper timing evidence.
    """

    paper_performance_evidence = False

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        local_files_only: bool = True,
    ) -> "HuggingFaceReferenceStateBackend":
        original_find_spec = importlib.util.find_spec

        def find_spec_without_bitsandbytes(name, *args, **kwargs):
            if name == "bitsandbytes" or name.startswith("bitsandbytes."):
                return None
            return original_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = find_spec_without_bitsandbytes
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            try:
                torch_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError("unsupported torch dtype: %s" % dtype) from error
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, local_files_only=local_files_only
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
            ).to(device)
        except ImportError as error:
            raise RuntimeError("torch and transformers are required") from error
        finally:
            importlib.util.find_spec = original_find_spec
        return cls(model, tokenizer, device)

    def tokenize_parts(self, prefix: str, segment: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        prefix_ids = tuple(
            int(token)
            for token in self.tokenizer.encode(prefix, add_special_tokens=True)
        )
        segment_ids = tuple(
            int(token)
            for token in self.tokenizer.encode(segment, add_special_tokens=False)
        )
        if not segment_ids:
            raise ValueError("tokenized segment must not be empty")
        joined_ids = tuple(
            int(token)
            for token in self.tokenizer.encode(
                prefix + segment, add_special_tokens=True
            )
        )
        if prefix_ids + segment_ids != joined_ids:
            raise ValueError(
                "prefix/segment tokenization is not boundary-stable; add an explicit "
                "delimiter or supply verified token IDs to prefill_ids"
            )
        return prefix_ids, segment_ids

    def prefill_ids(
        self,
        prefix_token_ids: Sequence[int],
        segment_token_ids: Sequence[int],
        capture_layers: Optional[Sequence[int]] = None,
    ) -> ReferencePrefill:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("torch is required") from error
        if not segment_token_ids:
            raise ValueError("segment_token_ids must not be empty")
        layers = _model_layers(self.model)
        wanted = set(
            range(len(layers)) if capture_layers is None else capture_layers
        )
        if any(layer < 0 or layer >= len(layers) for layer in wanted):
            raise ValueError("capture layer is outside the model")

        captured_queries: Dict[int, Any] = {}
        captured_keys: Dict[int, Any] = {}
        hooks = []
        for layer_index in sorted(wanted):
            attention = getattr(layers[layer_index], "self_attn", None)
            query_projection = getattr(attention, "q_proj", None)
            key_projection = getattr(attention, "k_proj", None)

            def capture_query(_module, _inputs, output, index=layer_index):
                captured_queries[index] = output.detach()

            def capture_key(_module, _inputs, output, index=layer_index):
                captured_keys[index] = output.detach()

            if query_projection is not None:
                hooks.append(query_projection.register_forward_hook(capture_query))
            if key_projection is not None:
                hooks.append(key_projection.register_forward_hook(capture_key))
        ids = torch.tensor(
            [list(prefix_token_ids) + list(segment_token_ids)],
            dtype=torch.long,
            device=self.device,
        )
        try:
            with torch.inference_mode():
                output = self.model(
                    ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
        cache = _legacy_cache(output.past_key_values)
        segment_length = len(segment_token_ids)
        states: Dict[int, ReferenceLayerState] = {}
        for layer_index in sorted(wanted):
            key, value = cache[layer_index][0], cache[layer_index][1]
            query = captured_queries.get(layer_index)
            key_pre_rope = captured_keys.get(layer_index)
            states[layer_index] = ReferenceLayerState(
                cached_key=key[:, :, -segment_length:, :].detach().cpu().clone(),
                cached_value=value[:, :, -segment_length:, :].detach().cpu().clone(),
                key_pre_rope=(
                    key_pre_rope[:, -segment_length:, :].cpu().clone()
                    if key_pre_rope is not None
                    else None
                ),
                hidden=output.hidden_states[layer_index + 1][
                    :, -segment_length:, :
                ].detach().cpu().clone(),
                query=(
                    query[:, -segment_length:, :].cpu().clone()
                    if query is not None
                    else None
                ),
            )
        return ReferencePrefill(
            len(prefix_token_ids), tuple(int(token) for token in segment_token_ids), states
        )

    @staticmethod
    def relative_l2(left, right) -> float:
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError("state tensors must have identical shapes")
        left_float = left.float()
        right_float = right.float()
        denominator = float(left_float.norm().item())
        numerator = float((left_float - right_float).norm().item())
        return numerator / max(denominator, 1e-12)

    @classmethod
    def compare_layer(
        cls, current: ReferenceLayerState, source: ReferenceLayerState
    ) -> Dict[str, Optional[float]]:
        return {
            "k_drift_pre_rope": (
                cls.relative_l2(current.key_pre_rope, source.key_pre_rope)
                if current.key_pre_rope is not None
                and source.key_pre_rope is not None
                else None
            ),
            "v_drift": cls.relative_l2(current.cached_value, source.cached_value),
            "hidden_drift": cls.relative_l2(current.hidden, source.hidden),
            "query_drift": (
                cls.relative_l2(current.query, source.query)
                if current.query is not None and source.query is not None
                else None
            ),
        }

    @staticmethod
    def save_canonical(path: Path, prefill: ReferencePrefill, metadata: Mapping[str, Any]) -> str:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("torch is required") from error
        payload = {
            "metadata": dict(metadata),
            "prefix_token_count": prefill.prefix_token_count,
            "segment_token_ids": prefill.segment_token_ids,
            "layer_states": dict(prefill.layer_states),
            "origin": "full_prefill",
            "exact": True,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest
