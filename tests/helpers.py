from probekv.contracts import HistoricalSource, KVLocation, SourceOrigin


def canonical_source(source_id="s1", content_hash="hash-c", context_id="ctx-a"):
    return HistoricalSource(
        source_id=source_id,
        content_hash=content_hash,
        context_id=context_id,
        model_signature="model@revision",
        token_count=512,
        exact=True,
        origin=SourceOrigin.FULL_PREFILL,
        kv_location=KVLocation.PINNED_CPU,
        kv_handles=("layer-0",),
        probe_summary={1: (0.1, 0.2)},
    )
