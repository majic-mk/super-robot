# A800 Stage 1 report — 2026-07-28

## Decision

**Strict Stage 1 status: failed; E1 remains prohibited.**

Hardware acceptance, the frozen software environment, ProbeKV preflight, the
Mistral-v0.3 H0 checks, and a minimally corrected CacheBlend diagnostic all
passed. The only failed gate is the plan's strict requirement that the pinned,
unmodified CacheBlend `example/blend.py` complete 10 samples.

At commit `b72d7945e6d6306f12be66520196e0f081fa2b0c`, the example enters the
xFormers CacheBlend path but raises `KeyError: suffix_len`. Its backend reads
`cache_fuse_metadata["suffix_len"]` for `status == 1`, while `blend.py` does not
initialize that field.

## Accepted results

- Formal hardware: one NVIDIA A800-SXM4-80GB, compute capability 8.0, with no
  other GPU process at acceptance or completion.
- Host allocation: 16 cgroup CPU cores, approximately 120 GiB memory, and a
  300 GB data volume.
- Storage smoke: approximately 3.1 GB/s sequential direct read and 366 MB/s
  sequential direct write. SSD-tier experiments remain eligible by the
  pre-registered read threshold, but read and write must be modeled separately.
- Pinned stack: PyTorch `2.2.1+cu121`, CUDA toolkit `12.1`, vLLM `0.4.1`, and
  xFormers `0.0.25`.
- ProbeKV commit: `21cdcde978536445e3b425a798dc8b453ed5a46b`.
- CacheBlend commit: `b72d7945e6d6306f12be66520196e0f081fa2b0c`.
- ProbeKV: compilation, 92 unit tests, contract validation, and the non-paper
  local E1/E2 simulation passed.
- Formal H0 model: `mistralai/Mistral-7B-Instruct-v0.3`, revision
  `c170c708c41dac9275d15a8fff4eca08d52bab71`.
- H0 prefixes are independent and naturally unequal: current P = 21 tokens,
  historical A = 15, and historical B = 12.
- Canonical KV save/load, loaded-cache next-token logits, source read-only
  hashes, and repeated greedy generation are exact.
- Model-state RoPE de-rotate/re-rotate relative error is
  `9.296705658231579e-17`, below the `1e-5` threshold.
- The first eight layers successfully expose query, pre-RoPE K, hidden state,
  cached K, and cached V; both historical Sources differ from the current
  request.

All timings in this stage are diagnostic and carry
`paper_performance_evidence: false`.

## CacheBlend result

The first unmodified attempt intentionally forced offline resolution and
confirmed that this vLLM path still requires the Hugging Face file-list API.
The second unmodified attempt used the available mirror, loaded Mistral,
entered the xFormers backend, and then failed at the missing `suffix_len`
field. Therefore the unmodified CB0 gate did not pass.

A diagnostic copy added exactly:

```python
cache_fuse_metadata["suffix_len"] = len(q_ids + s_end)
```

It completed all 10 cached and all 10 full paths without OOM, CUDA, or KV-shape
errors. Six output pairs were text-identical; the remaining four differed,
including two substantive answer differences. This diagnostic establishes
that the remaining runtime path works, but it is not labeled as an original
CB0 pass and its TTFT is not paper evidence.

## Required next action

Before CB1–CB3 or E1:

1. Pre-register the one-line CacheBlend fork correction and its rationale.
2. Commit the patch in the matched CacheBlend fork.
3. Rerun CB0 from that exact fork and retain both the original-failure and
   corrected-run evidence.
4. Run CacheBlend, the Cache-Craft-style baseline, and ProbeKV on that same
   patched stack.

The downloaded evidence bundle is under
`artifacts/a800_stage1_20260728/`; `SHA256SUMS.txt` verifies all 30 server
artifacts.
