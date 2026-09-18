# Outcome-blind expansion within frozen HotPotQA development groups

Before new GPU outcomes: use the existing recovery/geometry/pilot pipeline and
unchanged runtime SHA `80ad0278658da543cb15cb8536fdfde93c202cc5` on server51405.
Official training file SHA256:
`26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316`.
Verified against configs/h1_official_datasets.json and prepared_stream audit.
Only parents in schema10-profile-25f0cc1/mistral/development_partition.jsonl
are eligible; no partition enlargement or locked-test reading.

There are 17 natural corpus-repeat parent groups, seven with additional targets.
Two cases meet the existing >=5-target criterion and all-input 4096-token
geometry contract, but identity checking found they are slices of ONE document
group. The ordinary fit/validation freezer correctly rejected this input.
Use the explicit validation-only diagnostic: choose the lexicographically first
case_id per group before outcomes, giving one group and seven targets, no fit
set or Profile qualification. Keep the ordinary >=2-group split unchanged. Historical
sources must have four distinct full prefixes; preserve all original documents.
Chronology remains deterministic corpus pseudo-time, not a production trace.

Freeze short_answer_v1, 32-token generation, next-question boundary, fixed15
normalized K/V repair, common first reuse layer 9. Run full legacy diagnostics
only after exact self-state checks. No new threshold, repair ratio, prompt tune,
or scoring metric selected after observing answers.

Baselines fixed before execution: earliest historical Source, latest historical
Source, d1/d2/d4/d5/d8 residual ranking. Report post-hoc best fixed Source and
per-request actual-QA oracle as upper bounds, never deployable policies.
F1 tolerance remains 0.02 below the matched dense answer; separately report
absolute answer F1 and exact answers because poor dense answers can make this
relative coverage misleading. Tie-aware maximum-F1 agreement is reported both
over all requests and over non-tied requests only.

Verify this pilot's exact Segment token identities do not overlap the prior
MuSiQue/2Wiki pilot content. Reject overlap rather than call it new evidence.
Source-dependent answers alone do not establish multi-Source necessity.
No online TTFT gain, equal-byte dynamic-pool experiment, or qualification claim
can be derived from this forced Source diagnostic. Negative results are retained.

## Execution and results

The new explicit validation-only entry point required runtime SHA
`9dd133fda45f55c016df6745b871fdb1c6d42977` (default fit/validation behavior unchanged).
Frozen pilot `multitarget-qa-pilot-hotpot-9dd133f-v1.json` has partition digest
`c23af71910434e212e02ad5d8e00af34d43d67ff0a4b936d0f8fb35125c058b9`.
Exact Segment token comparison found no overlap with either previous pilot.

On server51405, output `qa-legacy-hotpot-9dd133f-server51405-v1`:

- Four Sources x five self checkpoints: all exact, relative-L2 zero.
- Seven targets x (dense + four fixed15 Sources): 35 actions complete.
- Four Source outputs have identical generated token IDs on every target.
- Each fixed Source, each depth selector, and per-request oracle mean F1 are
  0.6530612245. Each has 100% dense-relative safe coverage. No selection gain.
- This is neither online latency evidence nor a formal qualification.

## Newly identified sampling limitation

Outcome-independent audit `scripts/server/audit_source_relevance.py` verifies
the pilot/raw/case digests and exact normalized original request digest before
reading original dataset supporting-document labels. It does NOT infer relevance
from model answers. Reports `relevance-{musique,2wiki,hotpot}-54dcceb-v1.json`
are under `/root/autodl-tmp/probekv_stage2/artifacts`.

| Pilot | Target requests | Shared document is supporting | Non-supporting |
| --- | ---: | ---: | ---: |
| MuSiQue | 12 | 5 | 7 |
| 2Wiki | 11 | 1 | 10 |
| HotPotQA | 7 | 0 | 7 |

Thus 24/30 requests reuse a document not marked as answer-supporting. Only six
supporting-document requests exist, and only two have differing Source F1.
Document support is not proof the specific canonical token slice includes the
supporting sentence; that field deliberately remains null. Current negatives
must not be generalized to all evidence-bearing segments. Also do not discard
these negatives: they characterize distractor reuse and potential unnecessary
selector overhead. The labels are oracle diagnostic strata, not an online
feature available to the system.

Next required design: a separately frozen development extension, content-group
isolated from these observed pilots, stratified by supporting-sentence inclusion
in the actual shared token span and non-supporting controls. Keep original full
contexts and natural historical occurrences; no synthetic repetitions, answer-
conditioned selection, or post-outcome prompt changes. Predeclare earliest/latest
baselines and a fit-only choice rule if desired; evaluate a held-out group set.
Report both the natural prevalence and stratum-specific performance so that
support-enriched sampling is not misrepresented as overall workload gain.
Existing frozen parents do not supply a broad evidence-bearing >=5-target cohort;
do not silently enlarge that partition or lower its minimum target count.

No runtime scoring/threshold changed. GPU process finished and memory returned
to zero. Formal profile, runtime qualification, paper evidence and locked-test
access remain false.
