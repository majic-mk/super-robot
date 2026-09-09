# Position-state cache diagnostic patch

This is a separately audited CacheBlend tree change used only for the next
single-segment Mistral diagnostic. It does not change Source selection,
repair ratio, masks, timing endpoints, or admission thresholds.

## Change

The patched Llama and Qwen resumable advance methods cache, per request and
per `(active_positions, target_active_positions)` tuple:

* the device `LongTensor` position state;
* the full prompt position range;
* the `org_pos` range.

The tuple is replaced whenever a repair commit changes the active subset.
The original tuple order, device, dtype, and layer bounds remain validated;
the cache is only an allocation/reuse optimization. The previous adapter
change that queries completed layer events before waiting remains in the
ProbeKV tree.

## Audit identity

Patch tree:
`7c0187a48da19db4cd3ac8b3ecd795da761b9d2d`

Source file SHA256:

* `llama.py`: `99ccce7877d946a885618b35d084aa828ddcf8bd4f3b1e76225ba51bab61f7ee`
* `qwen2.py`: `d1c59103f5840bf59c8b2f9b1986b04c75819367c33033dacad50dff9560bb67`
* `xformers.py` unchanged from the pinned tree:
  `7c2a6ac4c0a351b3d88ae0f581f70f3075a39939112ee302984da640705acedf`

The independent patch audit records source-diff SHA256
`8d98160c5565c611e3fc4fdc0d3c4efb1ac117d374c22fc9d54fafe19db01195`.

## A800 result

Mistral, 512-token zero-native-Prefix matched comparison, same fixed Source,
boundary 2, fixed15, packed working layout, prefetch window 32:

| Arm | Mean first-token wall-clock |
| --- | ---: |
| Dense | 74.330 ms |
| CacheBlend pinned normal loop | 49.447 ms |
| ProbeKV position-cache tree | 60.612 ms |

Compared with the previous stream-wait tree (65.089 ms), this is a 4.477 ms
reduction (6.88%). It is not CacheBlend parity and is not paper evidence.

The r=1 endpoint remained exact: free greedy token IDs matched and common
teacher-forced logit relative-L2 was 0.0 for both reuse arms. ProbeKV's
instrumented trace reported 44 stream synchronizations versus 102 previously;
the remaining copy/attention activity is still substantial and requires a
separate layerwise transfer/compute analysis.

The modified server tree is retained separately from the pinned original;
old failure and comparison artifacts are not overwritten.

## Remaining copy/compute diagnosis

The same trace was correlated by CPU operation and GPU memcpy kind. The
position-cache change removes the large per-layer Python position material
copy, but the prefill still contains approximately 1,589 `aten::to`, 1,501
`aten::copy_`, and 337 `_to_copy` CPU operations in the profiled request.
The profiler reports many device-to-device copies (the common 8-KB class is
not by itself proof of KV transfer) and only eight 6,656-byte pageable-to-GPU
copies. Therefore the remaining 10--11 ms gap cannot safely be labelled
“H2D loading”; it is a mixture of model-side indexed KV writeback, tensor
materialization, and attention execution.

The next experiment is a source-level operation split, not another blind
optimization: separately time (a) composite old-KV row installation, (b)
indexed active-row writeback, and (c) the attention/MLP layer body. Any
replacement of advanced indexing by contiguous-run copies must preserve the
absolute union mask and pass the existing r=1 token/logit gate.
