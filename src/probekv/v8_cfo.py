from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class CanonicalChunkOccurrence:
    content_id: str
    occurrence_ordinal: int
    provenance_position: int
    token_count: int

    def __post_init__(self) -> None:
        if not self.content_id or min(
            self.occurrence_ordinal, self.provenance_position
        ) < 0 or self.token_count <= 0:
            raise ValueError("invalid canonical chunk occurrence")

    @property
    def match_id(self) -> str:
        return "%s#%d" % (self.content_id, self.occurrence_ordinal)


@dataclass(frozen=True)
class SourceCFOMetadata:
    historical_prefix_chunk_occurrences: Tuple[CanonicalChunkOccurrence, ...]
    inter_mass_by_prefix_occurrence: Mapping[str, float]
    normalized_inter_by_layer: Tuple[float, ...]
    normalized_intra_by_layer: Tuple[float, ...]
    cci: float
    metadata_digest: str

    def __post_init__(self) -> None:
        identities = tuple(row.match_id for row in self.historical_prefix_chunk_occurrences)
        if len(set(identities)) != len(identities):
            raise ValueError("historical prefix occurrence identities must be unique")
        if set(self.inter_mass_by_prefix_occurrence) != set(identities):
            raise ValueError("CFO inter mass must cover every historical prefix occurrence")
        if not self.normalized_inter_by_layer or len(self.normalized_inter_by_layer) != len(
            self.normalized_intra_by_layer
        ):
            raise ValueError("CFO metadata requires matched layer-wise inter/intra values")
        values = (
            *self.inter_mass_by_prefix_occurrence.values(),
            *self.normalized_inter_by_layer,
            *self.normalized_intra_by_layer,
            self.cci,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("CFO metadata values must be finite and non-negative")
        if not 0 <= self.cci <= 1 or not self.metadata_digest:
            raise ValueError("invalid CCI or metadata digest")


@dataclass(frozen=True)
class CFOResult:
    prefix_overlap: float
    order_penalty: float
    adjusted_prefix_overlap: float
    cci: float
    cfo_raw: float
    cfo_operational: float


def _cci_from_means(a_bar: float, b_bar: float, epsilon: float) -> float:
    if min(a_bar, b_bar) < 0 or epsilon <= 0:
        raise ValueError("CCI inputs are invalid")
    if b_bar <= epsilon:
        return 1.0 if a_bar > epsilon else 0.5
    ratio = a_bar / b_bar
    if ratio >= 80:
        return 1.0
    return 1.0 / (1.0 + math.exp(-ratio))


def build_source_cfo_metadata(
    *,
    historical_prefix_chunk_occurrences: Sequence[CanonicalChunkOccurrence],
    inter_mass_by_layer_and_occurrence: Sequence[Mapping[str, float]],
    intra_mass_by_layer: Sequence[float],
    target_token_count: int,
    epsilon: float = 1e-12,
) -> SourceCFOMetadata:
    prefix = tuple(historical_prefix_chunk_occurrences)
    if target_token_count <= 0:
        raise ValueError("CFO metadata requires a target chunk")
    if len(inter_mass_by_layer_and_occurrence) != len(intra_mass_by_layer) or not intra_mass_by_layer:
        raise ValueError("CFO layer measurements are incomplete")
    expected = {row.match_id for row in prefix}
    if any(set(layer) != expected for layer in inter_mass_by_layer_and_occurrence):
        raise ValueError("each CFO layer must cover all prefix occurrences")
    if any(
        not math.isfinite(value) or value < 0
        for layer in inter_mass_by_layer_and_occurrence
        for value in layer.values()
    ) or any(not math.isfinite(value) or value < 0 for value in intra_mass_by_layer):
        raise ValueError("CFO attention masses must be finite and non-negative")
    occurrence_by_id = {row.match_id: row for row in prefix}
    normalized_inter = []
    for layer in inter_mass_by_layer_and_occurrence:
        normalized_inter.append(
            sum(
                layer[match_id]
                / (target_token_count * occurrence_by_id[match_id].token_count)
                for match_id in expected
            )
        )
    normalized_intra = [
        value / (target_token_count ** 2) for value in intra_mass_by_layer
    ]
    a_bar = sum(normalized_inter) / len(normalized_inter)
    b_bar = sum(normalized_intra) / len(normalized_intra)
    cci = _cci_from_means(a_bar, b_bar, epsilon)
    inter_aggregate = {
        match_id: sum(layer[match_id] for layer in inter_mass_by_layer_and_occurrence)
        / len(inter_mass_by_layer_and_occurrence)
        for match_id in expected
    }
    payload = {
        "prefix": [row.__dict__ for row in prefix],
        "inter": inter_aggregate,
        "normalized_inter": normalized_inter,
        "normalized_intra": normalized_intra,
        "cci": cci,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SourceCFOMetadata(
        prefix,
        inter_aggregate,
        tuple(normalized_inter),
        tuple(normalized_intra),
        cci,
        digest,
    )


def compute_cachecraft_cfo(
    metadata: SourceCFOMetadata,
    current_prefix_chunk_occurrences: Sequence[CanonicalChunkOccurrence],
    *,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
) -> CFOResult:
    if alpha < 0 or not math.isfinite(alpha) or epsilon <= 0:
        raise ValueError("CFO alpha/epsilon is invalid")
    current = tuple(current_prefix_chunk_occurrences)
    current_order = {row.match_id: index for index, row in enumerate(current)}
    historical = metadata.historical_prefix_chunk_occurrences
    total_inter = sum(metadata.inter_mass_by_prefix_occurrence.values())
    common = [row.match_id for row in historical if row.match_id in current_order]
    beta = (
        sum(metadata.inter_mass_by_prefix_occurrence[match_id] for match_id in common)
        / total_inter
        if total_inter > epsilon
        else 0.0
    )
    if len(common) < 2:
        order_penalty = 0.0
    else:
        current_positions = [current_order[match_id] for match_id in common]
        discordant = sum(
            current_positions[left] > current_positions[right]
            for left in range(len(current_positions))
            for right in range(left + 1, len(current_positions))
        )
        order_penalty = discordant / (len(common) * (len(common) - 1) / 2)
    beta_prime = beta * (1.0 - order_penalty)
    cfo_raw = alpha * metadata.cci * (1.0 - beta_prime)
    if not math.isfinite(cfo_raw):
        raise ValueError("CFO result is non-finite")
    return CFOResult(
        beta,
        order_penalty,
        beta_prime,
        metadata.cci,
        cfo_raw,
        min(1.0, max(0.0, cfo_raw)),
    )


@dataclass(frozen=True)
class StreamingAttentionMass:
    inter_mass_by_pair: Mapping[Tuple[str, str], float]
    intra_mass_by_occurrence: Mapping[str, float]


def streaming_qk_attention_mass(
    q: object,
    k: object,
    token_occurrence_ids: Sequence[str],
    *,
    scale: float,
    block_size: int = 64,
) -> StreamingAttentionMass:
    """Accumulate exact causal attention mass without materializing NxN.

    Inputs are post-RoPE tensors shaped ``[tokens, heads, head_dim]``.  GQA is
    reproduced by the standard contiguous Q-head to KV-head grouping.  A
    two-pass streaming log-sum-exp supplies the global softmax denominator;
    individual key blocks are never normalized independently.
    """

    import torch

    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError("streaming CFO attention requires torch tensors")
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("Q/K geometry is incompatible")
    token_count, q_heads, _ = q.shape
    kv_heads = int(k.shape[1])
    if token_count != len(token_occurrence_ids) or q_heads % kv_heads or block_size <= 0:
        raise ValueError("invalid CFO token/head/block geometry")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be finite and positive")
    device = q.device
    head_map = torch.arange(q_heads, device=device) // (q_heads // kv_heads)
    qf = q.float()
    kf = k.float()
    positions = torch.arange(token_count, device=device)
    inter: Dict[Tuple[str, str], float] = {}
    intra: Dict[str, float] = {}

    for q_start in range(0, token_count, block_size):
        q_end = min(token_count, q_start + block_size)
        q_block = qf[q_start:q_end]
        q_positions = positions[q_start:q_end]
        running_max = torch.full(
            (q_end - q_start, q_heads), -torch.inf, device=device, dtype=torch.float32
        )
        running_sum = torch.zeros_like(running_max)
        for k_start in range(0, token_count, block_size):
            k_end = min(token_count, k_start + block_size)
            expanded_k = kf[k_start:k_end].index_select(1, head_map)
            logits = torch.einsum("qhd,khd->qhk", q_block, expanded_k) * scale
            valid = positions[k_start:k_end].view(1, 1, -1) <= q_positions.view(-1, 1, 1)
            logits = logits.masked_fill(~valid, -torch.inf)
            block_max = logits.amax(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            running_sum = (
                running_sum * torch.exp(running_max - new_max)
                + torch.exp(logits - new_max.unsqueeze(-1)).masked_fill(~valid, 0).sum(dim=-1)
            )
            running_max = new_max
        log_denominator = running_max + running_sum.clamp_min(1e-30).log()

        for k_start in range(0, token_count, block_size):
            k_end = min(token_count, k_start + block_size)
            expanded_k = kf[k_start:k_end].index_select(1, head_map)
            logits = torch.einsum("qhd,khd->qhk", q_block, expanded_k) * scale
            valid = positions[k_start:k_end].view(1, 1, -1) <= q_positions.view(-1, 1, 1)
            probabilities = torch.exp(logits - log_denominator.unsqueeze(-1)).masked_fill(~valid, 0)
            probabilities = probabilities.mean(dim=1)
            for local_q, absolute_q in enumerate(range(q_start, q_end)):
                query_occurrence = token_occurrence_ids[absolute_q]
                for local_k, absolute_k in enumerate(range(k_start, k_end)):
                    if absolute_k > absolute_q:
                        continue
                    key_occurrence = token_occurrence_ids[absolute_k]
                    mass = float(probabilities[local_q, local_k].item())
                    if key_occurrence == query_occurrence:
                        intra[query_occurrence] = intra.get(query_occurrence, 0.0) + mass
                    else:
                        pair = (key_occurrence, query_occurrence)
                        inter[pair] = inter.get(pair, 0.0) + mass
    return StreamingAttentionMass(inter, intra)


def eager_qk_attention_mass(
    q: object,
    k: object,
    token_occurrence_ids: Sequence[str],
    *,
    scale: float,
) -> StreamingAttentionMass:
    """Small-context correctness oracle for the streaming CFO accumulator."""

    import torch

    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError("eager CFO attention requires torch tensors")
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("Q/K geometry is incompatible")
    token_count, q_heads, _ = q.shape
    kv_heads = int(k.shape[1])
    if token_count != len(token_occurrence_ids) or q_heads % kv_heads:
        raise ValueError("invalid CFO token/head geometry")
    head_map = torch.arange(q_heads, device=q.device) // (q_heads // kv_heads)
    expanded_k = k.float().index_select(1, head_map)
    logits = torch.einsum("qhd,khd->qhk", q.float(), expanded_k) * float(scale)
    positions = torch.arange(token_count, device=q.device)
    valid = positions.view(1, 1, -1) <= positions.view(-1, 1, 1)
    probabilities = torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)
    probabilities = probabilities.mean(dim=1)
    inter: Dict[Tuple[str, str], float] = {}
    intra: Dict[str, float] = {}
    for query_position in range(token_count):
        query_occurrence = token_occurrence_ids[query_position]
        for key_position in range(query_position + 1):
            key_occurrence = token_occurrence_ids[key_position]
            mass = float(probabilities[query_position, key_position].item())
            if key_occurrence == query_occurrence:
                intra[query_occurrence] = intra.get(query_occurrence, 0.0) + mass
            else:
                pair = (key_occurrence, query_occurrence)
                inter[pair] = inter.get(pair, 0.0) + mass
    return StreamingAttentionMass(inter, intra)


def _mass_max_abs_error(
    left: StreamingAttentionMass, right: StreamingAttentionMass
) -> float:
    errors = [
        abs(left.inter_mass_by_pair.get(key, 0.0) - right.inter_mass_by_pair.get(key, 0.0))
        for key in set(left.inter_mass_by_pair) | set(right.inter_mass_by_pair)
    ]
    errors.extend(
        abs(
            left.intra_mass_by_occurrence.get(key, 0.0)
            - right.intra_mass_by_occurrence.get(key, 0.0)
        )
        for key in set(left.intra_mass_by_occurrence) | set(right.intra_mass_by_occurrence)
    )
    return max(errors, default=0.0)


class CFOFullPrefillCollector:
    """One-pass post-RoPE Q/K hook used only for canonical full-prefill.

    The patched model calls this object immediately after RoPE.  Each layer is
    reduced to chunk-level attention masses before returning, so token-square
    attention matrices and full Q/K tensors are never retained.
    """

    def __init__(
        self,
        token_occurrence_ids: Sequence[str],
        *,
        expected_layers: int,
        block_size: int = 64,
        eager_reference: bool = False,
        eager_tolerance: float = 2e-5,
    ) -> None:
        if expected_layers <= 0 or block_size <= 0 or not token_occurrence_ids:
            raise ValueError("invalid CFO full-prefill collector contract")
        self.token_occurrence_ids = tuple(str(value) for value in token_occurrence_ids)
        self.expected_layers = int(expected_layers)
        self.block_size = int(block_size)
        self.eager_reference = bool(eager_reference)
        self.eager_tolerance = float(eager_tolerance)
        self.layer_masses: list[StreamingAttentionMass] = []
        self.eager_max_abs_error = 0.0
        self.ignored_nonprefill_calls = 0

    def __call__(self, **payload: Any) -> None:
        import torch

        q = payload.get("q")
        k = payload.get("k")
        positions = payload.get("positions")
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            raise TypeError("CacheBlend CFO hook did not provide Q/K tensors")
        token_count = len(self.token_occurrence_ids)
        if int(q.shape[0]) != token_count:
            self.ignored_nonprefill_calls += 1
            return
        if len(self.layer_masses) >= self.expected_layers:
            raise RuntimeError("CFO hook observed more full-prefill layers than expected")
        expected_positions = torch.arange(token_count, device=positions.device)
        if positions.ndim != 1 or not torch.equal(positions, expected_positions):
            raise RuntimeError("CFO full-prefill positions are not canonical absolute positions")
        q_heads = int(payload["q_heads"])
        kv_heads = int(payload["kv_heads"])
        head_dim = int(payload["head_dim"])
        q3 = q.reshape(token_count, q_heads, head_dim)
        k3 = k.reshape(token_count, kv_heads, head_dim)
        observed = streaming_qk_attention_mass(
            q3,
            k3,
            self.token_occurrence_ids,
            scale=float(payload["scale"]),
            block_size=self.block_size,
        )
        if self.eager_reference:
            reference = eager_qk_attention_mass(
                q3, k3, self.token_occurrence_ids, scale=float(payload["scale"])
            )
            self.eager_max_abs_error = max(
                self.eager_max_abs_error, _mass_max_abs_error(observed, reference)
            )
        self.layer_masses.append(observed)

    def finalize(
        self,
        *,
        prefix_occurrences: Sequence[CanonicalChunkOccurrence],
        target_occurrence: CanonicalChunkOccurrence,
    ) -> Tuple[SourceCFOMetadata, Mapping[str, Any]]:
        if len(self.layer_masses) != self.expected_layers:
            raise RuntimeError(
                "CFO hook captured %d/%d layers"
                % (len(self.layer_masses), self.expected_layers)
            )
        target_id = target_occurrence.match_id
        inter_by_layer = [
            {
                occurrence.match_id: layer.inter_mass_by_pair.get(
                    (occurrence.match_id, target_id), 0.0
                )
                for occurrence in prefix_occurrences
            }
            for layer in self.layer_masses
        ]
        intra_by_layer = [
            layer.intra_mass_by_occurrence.get(target_id, 0.0)
            for layer in self.layer_masses
        ]
        metadata = build_source_cfo_metadata(
            historical_prefix_chunk_occurrences=prefix_occurrences,
            inter_mass_by_layer_and_occurrence=inter_by_layer,
            intra_mass_by_layer=intra_by_layer,
            target_token_count=target_occurrence.token_count,
        )
        audit = {
            "expected_layers": self.expected_layers,
            "captured_layers": len(self.layer_masses),
            "post_rope_qk": True,
            "causal_mask": True,
            "gqa_mapping": True,
            "fp32_accumulation": True,
            "streaming_logsumexp": True,
            "eager_reference": self.eager_reference,
            "eager_max_abs_error": self.eager_max_abs_error,
            "eager_tolerance": self.eager_tolerance,
            "ignored_nonprefill_calls": self.ignored_nonprefill_calls,
            "passed": (
                len(self.layer_masses) == self.expected_layers
                and (
                    not self.eager_reference
                    or self.eager_max_abs_error <= self.eager_tolerance
                )
            ),
            "metadata_digest": metadata.metadata_digest,
        }
        return metadata, audit
