# ProbeKV local validation report

Date: 2026-07-27

Workstation: Windows 11 / RTX 5070 Ti Laptop 12 GB / 32 GB RAM

## Result

All validations executable in the current installed environment pass. The
remaining GPU-framework check is blocked by environment compatibility, not by
a failing ProbeKV invariant.

| Check | Result | Evidence |
|---|---:|---|
| Python compile | pass | `python -m compileall -q src scripts tests` |
| Unit/integration/property tests | 87/87 pass | source, RoPE, labels, probe, calibration, RAG/E1 orchestration, manifest isolation, CacheBlend adapter, resume, prefetch, scheduler, statistics, CLI |
| Frozen contract audit | pass | 360 main RAG cells; 1620 profile cells without SSD |
| Exact H3 sample audit | pass | minimum 299 zero-violation cases; pooled RAG has 600/model |
| Local simulation | pass | JSONL and Parquet emitted, all rows marked non-paper evidence |
| Real local causal-LM H0 | pass | cached TinyLlama-1.1B, CPU, 22 layers |
| Real per-layer reference probe | pass | pre-RoPE K, V, hidden and query captured for layers 1-5 |
| Local E1/E2 closed loop | pass | 60 fixtures; 10,800 ratio rows; 1,920 probe rows; selection and abstention both exercised |
| RAG manifest reference probe | pass | HotPot-shaped tokenizer fixture; 1 current + 4 Sources; 5 layers; 20 real observations |
| E1 four-shard orchestration | pass | 2,664 jobs; no missing/duplicate results; 296 labels; synthetic H1 remains non-paper |
| Python wheel build | pass | `probekv-0.1.0-py3-none-any.whl` |
| Current PyTorch GPU compatibility | fail/precondition | installed cu121 supports through `sm_90`; GPU is `sm_120` |

## Real-model observations

Two different, equal-token-length prefixes were independently prefixed to the
same 13-token segment. The same absolute token positions were therefore used.

- Layer 0 C-KV difference: `0.0` for K and V. This is expected because the first
  layer's K/V projection has not yet mixed preceding context.
- Layer 21 mean absolute difference: `0.275337` for K and `0.183128` for V.
- Raw current-state drift selected the matching A-conditioned source:
  source A `0.0`, source B `0.458465`.
- Canonical tensor save/load was bit-exact.
- The next-token logits from original versus reloaded full KV cache were exact.
- Computing summaries did not mutate canonical tensors.
- Two greedy generations were token-identical.

The new reference-state backend also captured every layer through the 25%
ceiling on the 22-layer model (layers 1-5). For the different historical
context, pre-RoPE K drift grew from `0.0` at layer 1 to `0.1400` at layer 5;
hidden drift grew from `0.0976` to `0.3202`. Matching-source drift remained
exactly zero, and query plus pre-RoPE K hooks were present at every layer.

This directly confirms the concern about `S1=KV(C|A)` and `S2=KV(C|B)`: the
deep-layer state of C carries historical-context influence. S2 cannot be made
by copying S1 and relabelling it; both must come from independent full prefills.

## Environment blocker

`nvidia-smi` reports compute capability `12.0` (`sm_120`). The installed
PyTorch is 2.4.1+cu121 and its compiled architecture list ends at `sm_90`.
WSL has no pip/PyTorch, passwordless sudo is unavailable, and outbound WSL
package download timed out. A modern isolated CUDA 12.8 PyTorch environment is
therefore still needed for local GPU tensor execution. This does not change the
paper rule: A100 measurements must use the pinned CacheBlend stack.

## Interpretation of synthetic gates

The deterministic simulation exists to exercise abstention, admission,
prefetch, scheduling and gate code. Its H1/H2/H4 booleans are not empirical
evidence and must not be copied into the paper. A failed synthetic H2 is a
successful demonstration that the gate stops an under-covered configuration,
not a conclusion about ProbeKV.

Likewise, the 60-case local E1/E2 run uses generated latent safe ratios. Its
88.9% selection coverage, 11.1% abstention rate and ranking metrics only prove
that the end-to-end software and audit paths execute; they are not H1/H2
evidence and are deliberately labelled `paper_evidence: false`.
