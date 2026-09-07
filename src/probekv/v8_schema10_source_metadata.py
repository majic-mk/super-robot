"""Publication/lookup validation shared by the physical store and selector."""
from dataclasses import asdict

from .v8_cfo import CanonicalChunkOccurrence, SourceCFOMetadata
from .v8_schema10_execution import digest_json


def read_cfo_metadata(payload):
    row = dict(payload)
    row["historical_prefix_chunk_occurrences"] = tuple(
        CanonicalChunkOccurrence(**x) for x in row["historical_prefix_chunk_occurrences"])
    metadata = SourceCFOMetadata(**row)
    expected = digest_json({"prefix": [asdict(x) for x in metadata.historical_prefix_chunk_occurrences],
        "inter": metadata.inter_mass_by_prefix_occurrence,
        "normalized_inter": list(metadata.normalized_inter_by_layer),
        "normalized_intra": list(metadata.normalized_intra_by_layer), "cci": metadata.cci})
    if metadata.metadata_digest != expected:
        raise ValueError("CFO metadata digest mismatch")
    return metadata


def validate_publication_metadata(metadata, *, token_count, num_layers):
    if not metadata.get("tokenizer_hash") or not metadata.get("runtime_compatibility"):
        raise ValueError("publication lacks tokenizer/runtime compatibility")
    tokens = metadata.get("token_ids", ())
    if len(tokens) != token_count or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("publication token identity does not match KV rows")
    try:
        cfo = read_cfo_metadata(metadata["cfo"])
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete canonical CFO metadata") from exc
    if len(cfo.normalized_inter_by_layer) != num_layers:
        raise ValueError("CFO metadata does not cover every model layer")
