# Identity first, then depth and actual Source quality

Scope: existing Mistral single-Segment development cohorts, unchanged requests,
fixed15 normalized K/V repair, common first reuse layer 9. No threshold fitting,
new selector, production admission claim, or locked-test access.

1. Recompute Source identity from the original complete prefix, absolute positions,
   occurrence, model and content. Check target/source exact token rows; compare
   pre-RoPE K by content ordinal, not by equal historical/current absolute index.
2. Replay each original historical input through the live dense path. Compare
   independent SelectionState at completed depths 1/2/4/5/8. Record exact hashes
   and global relative-L2; use the existing numerical sentinel limit 1e-4.
   Fail before target ranking if self-state alignment fails. No full-KV fallback.
3. Capture all five depths on the same dense target trajectory. Rerun dense and
   all four fixed15 Source actions with unchanged QA contract. Deep residual
   ranking is a proxy, never the ground-truth quality oracle.
4. Report actual F1, safe coverage, exact generated-token equality, residual
   rankings and tie-aware best-source agreement. Compare earliest/latest fixed
   policies, post-hoc best fixed (upper bound only), and per-request QA oracle.
   Earlier outcomes were already seen: these cohorts remain diagnostics, not
   new untouched confirmatory validation.
5. If deeper scores improve actual ranking, shallow information is implicated.
   If all depths fail after alignment passes, proxy adequacy remains doubtful.
   Neither conclusion proves a universal cause from four small content groups.
6. Only broader unseen causal cohorts under equal byte budgets and full online
   accounting can establish net multi-Source necessity. No-positive-gain is valid.

## Cache-Craft primary-source check

Source: https://arxiv.org/html/2502.15734v1 (sections 3.3, 5.1, 5.4, 6).
It explicitly stores multiple prefix-conditioned variants per chunk, selects
minimum CFO, and manages a global N*M population. Its stated setup starts with
100 chunks and 5 variants. Figure 25 shows the final Sys-X pool with 186 unique
chunks and up to 11 variants. Its beta/CCI/focus ablations evaluate repair and
quality; this is not an isolated equal-byte K=1/2/4 experiment proving a
current-state selector beats a fixed Source at fixed repair. Figure 8 also
concerns chunks drawn from different historical requests, a different notion
from choosing among variants of one identical chunk. We must establish our
incremental contribution independently, not infer it from their system speedup.
