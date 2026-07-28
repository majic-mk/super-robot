# Formal server execution runbook

## Stop before expensive runs

Run E0 first. Do not launch E1 until full-prefill identity, RoPE alignment,
CacheBlend quality, timing stability and memory telemetry pass.

The repository's local pipeline and `CacheBlendBackend` adapter are ready, but
the pinned CacheBlend runtime shim is intentionally not simulated. Complete the
CB0-CB3 milestones in `CACHEBLEND_INTEGRATION.md` before claiming E0 complete.

## Environment

1. Clone CacheBlend and checkout
   `b72d7945e6d6306f12be66520196e0f081fa2b0c`.
2. Preserve its vLLM 0.4.1 / PyTorch 2.2.1 / xFormers 0.0.25 /
   CUDA 12.1 stack.
3. Record `nvidia-smi`, package lock, model revision, tokenizer revision and
   repository dirty state.
4. Use the config-frozen single NVIDIA A800-SXM4-80GB for ProbeKV and every
   directly compared baseline. Use pinned CPU memory. Add SSD only after
   measuring sequential read bandwidth of at least 3 GB/s.

Initial server preparation:

```bash
python -m pip install -e . -r requirements/ci.txt -r requirements/analysis.txt
bash scripts/server/run_preflight.sh
bash scripts/server/prepare_cacheblend.sh /absolute/external/path/CacheBlend
```

The CacheBlend path should be outside this repository, as should model weights,
datasets and raw results.

Before launching E1, build and archive the immutable job manifest described in
`E1_JOBS.md`. A server worker may change attempts after a crash, but it must not
change a job's case, Source, layer or repair ratio.

## Required timing boundaries

Use CUDA events around GPU work and monotonic host timestamps around I/O.
Record, separately:

- current probe compute;
- comparison;
- speculative and winner KV transfer;
- copy/compute overlap and interference;
- repair;
- fallback recompute;
- decode start and TTFT;
- transferred and wasted bytes;
- peak allocated/reserved HBM.

The admission calculation must use the measured total visible path, not kernel
repair time alone.

## Evidence hygiene

Every output row needs case/segment group, split, seed, model and tokenizer
revision, code commit, environment hash, source origin, storage tier, K,
concurrency, gamma, `L_probe`, `L_reuse`, selection/abstention reason and timing
breakdown. Keep failed/OOM/reset runs with an explicit status rather than
silently deleting them.
