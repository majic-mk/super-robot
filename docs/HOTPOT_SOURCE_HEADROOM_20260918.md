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
