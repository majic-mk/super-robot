# ProbeKV KV writeback kernel check (2026-09-09)

## Scope

This is a bounded external CacheBlend-patch experiment.  It replaces the
resumable status-2 indexed assignment

```python
key_old[active_positions] = key
value_old[active_positions] = value
```

with `index_copy_(0, active_positions, ...)`.  Selector policy, canonical
Source identity, repair mask, model inputs, Prefix policy, and the `r=1`
endpoint are unchanged.  The external patch was not merged into the ProbeKV
repository.

Patch provenance:

```text
base CacheBlend tree: 7c0187a48da19db4cd3ac8b3ecd795da761b9d2d
index-copy tree:      2695171c7d38c24eef43dd3603239d292fb48472
index-copy diff SHA:  4164410cfa4b301b22a01b66f1778e706cba3657abec3eb71b6c9440d4bd6ad7
position-cache model files: llama 99ccce..., qwen2 d1c591...
GPU: NVIDIA A800-SXM4-80GB
model: Mistral-7B-Instruct-v0.3
segment: 512 tokens; boundary 2; prefetch window 32; packed_slice
```

## Results

The run was real CUDA execution (`fake_timing=false`).  The `r=1` endpoint
passed exact token-ID equality and 32-token teacher-forced logit relative-L2
(`0.0`).  Fixed-15 rows from the five paired loop repetitions were:

```text
dense:       74.09 ms mean (approximately)
CacheBlend:  48.84 ms mean (approximately)
ProbeKV:     59.88 ms mean (approximately)
```

The previous position-state-cache run was approximately 60.61 ms, so this
kernel substitution saves only about 0.7 ms (about 1.2%).  It is therefore a
safe micro-optimization, not the explanation for the remaining gap to the
CacheBlend loop.

## Interpretation

The profiler still attributes the dominant residual work to model-layer
`aten::copy_`/`aten::_to_copy` and temporary tensor materialization.  The
patched resumable attention path expands Mistral GQA tensors and then makes
contiguous MHA views before the partial-bias call.  That path is a stronger
candidate for the next bounded experiment than another Source-pool or
selector change.  Any replacement must first pass the same `r=1` token/logit
checks and then fixed-15 checks; no performance claim is made from this
diagnostic alone.

This result does not establish that ProbeKV is slower because of Source
selection.  It isolates one writeback kernel and shows that its contribution
is small relative to the model-side GQA/attention path.

