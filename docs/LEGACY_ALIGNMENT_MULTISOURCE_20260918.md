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

## Executed evidence: 2Wiki

Runtime SHA `80ad0278658da543cb15cb8536fdfde93c202cc5`, server51405,
GPU `GPU-f8226e6e-646e-6bad-ad43-dea7cf5860cb`.
Output: `/root/autodl-tmp/probekv_stage2/artifacts/qa-legacy-2wiki-80ad027-server51405-v1`.
Audit: `dominance-legacy-2wiki-80ad027-v1.json` in the same artifacts parent.

- Eight historical inputs x five checkpoints: all 40 current/canonical K hashes
  exactly equal, global relative-L2 = 0. Target identity/position checks pass.
- Eleven target requests x (dense + four Sources) completed, fixed15 K/V.
- Depths 1/2/4/5/8 each score 8/11 maximum-F1 tie hits and 9/11 dense-relative
  safe hits. Group00: every depth mean F1 .2200, safe 3/5; group01: every depth
  mean F1 .537698, safe 6/6. Deeper ranking did not resolve the known failures.
- Group00 earliest Source safe 3/5, latest 4/5, QA oracle 5/5; a post-hoc fixed
  Source also reaches 5/5. Group01 earliest 6/6, latest 5/6, oracle 6/6, also
  attainable by a fixed Source. Current-state ranking adds no demonstrated
  coverage improvement over earliest on this sample.

Interpretation: this numerical self-control rules out the tested canonical/live
state mismatch; it does not prove every runtime placement configuration correct.
The failure cannot be attributed solely to depth 1/2 information. Residual score
adequacy for repaired answer quality is now a leading concern, not a certified
causal explanation. An oracle advantage over an arbitrary fixed Source is not
proof that request-adaptive multi-Source selection is necessary.

## Executed evidence: MuSiQue and cross-run stability

Same runtime SHA/GPU. Output `qa-legacy-musique-80ad027-server51405-v1` and
audit `dominance-legacy-musique-80ad027-v1.json` under the artifacts parent above.
Eight more self inputs x five checkpoints are bit-identical (relative-L2 zero).
Twelve targets x five actions completed. Combined: 16 Sources, 80 exact self
checks, 23 targets, 115 dense/Source QA actions. Both datasets' generated token
IDs for every action match their respective previous fixed-boundary QA run.

| Dataset | Targets | d1 max-F1 tie hits | d2 | d4 | d5 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 11 | 8 | 8 | 8 | 8 | 8 |
| MuSiQue | 12 | 11 | 10 | 10 | 10 | 10 |

Ties dominate: only four 2Wiki targets and two MuSiQue targets have differing
Source F1. On these, d1 hits 1/4 and 1/2 respectively; later checkpoints hit
1/4 and 0/2. Source answer strings differ on 5/11 and 3/12 respectively.
Different outputs establish sensitivity to the Source, NOT necessity of dynamic
multi-Source selection. Every group still has a fixed Source that is safe for
all targets under the existing dense-relative tolerance. MuSiQue's modest
oracle-vs-fixed mean-F1 difference retains known answer-format confounding.

No thresholds were altered. Legacy did not rescue the tested counterexamples.
Do not promote early locking or claim multi-Source net benefit from this batch.
These are forced fixed-layer-9 diagnostics, not measured online d1/d2 commits
at layers 2/3, nor a proof of lack of benefit at all boundaries/workloads.

Next decision sequence: keep these negative controls; freeze an outcome-blind
expanded natural content-group cohort; compare earliest/latest and fit-selected
fixed policies against actual quality-constrained per-request oracle under
equal bytes. If meaningful oracle headroom exists, test proxy changes as separate
development ablations. Only then measure full online net benefit. No deeper
threshold sweep, CFO restoration, or added anchor mechanism as a substitute for
establishing this headroom. Existing QA cohorts cannot become fresh test data.

Local validation: 945 unittest cases, one historical skip; compileall and contract
validator pass. No formal Profile, qualification, or paper evidence produced.
