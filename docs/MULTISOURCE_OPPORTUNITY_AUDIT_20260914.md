# Fast-path safety and multi-Source data audit

## Evidence correction: native capture is not a complementarity certificate

The server report `multisource-complementarity-79424e5.json` was inspected on
2026-09-14. It contains three captures, four Sources in each, and d1/d2 winner
disagreement in two of the three captures. These are dense-shadow observations:
native K is captured on GPU, but comparison/extraction is on CPU. They do not
measure online selector latency, QA, commit coverage, or net gain.

The v1 aggregator incorrectly set `multisource_complementarity_observed=true`
when global winner IDs differed across captures. Different candidate pools
necessarily have different Source IDs; this is not evidence of complementary
Sources for future requests. The old report must not drive Go/No-Go. Preserve
it as superseded evidence; do not rewrite its historical bytes.

The v2 aggregator requires raw `observation.json`, validates its digest and
recomputes `replay.json`. Duplicate observations and tampered replay are rejected.
CPU tests retain CPU provenance. Cross-request ranking changes are grouped only
within identical Source ID sets and model/tokenizer/code/patch/config/partition
bindings. Even such ranking changes leave complementarity **pending**: fixed-pool
multi-target QA and matched net-gain/coverage evidence are still necessary.

The previous conversational statement that these three captures prove Source
complementarity is withdrawn. Likewise, different d1/d2 winners do not tell us
which depth is correct; deep residual and QA oracles remain pending.

### Cost probe continuation status

The current remote checkout (`79424e5474bb65a4ee97dfeea7463b337de61e21`)
correctly rejected the old `native-9e296fef-server46068-v44-costprobe` inputs
because of code revision mismatch. A fresh cost run has **not been confirmed**.
The SSH session subsequently closed; two later TCP probes of port 46068 were
refused. Do not interpret this as proof that the instance is powered off or that
the submitted command ran. Inspect processes and output files after reconnecting
before launching another run. Never delete or reuse an existing output directory.

Resume with verified model/patch assets and the intended imported vLLM path.
Use the basic single-Segment correctness + `--cost-probe` path, not the separate
matched-executor-only path. A 512-token Segment exceeds the bounded eager CFO
reference's total-source limit; explicitly use `--skip-eager-cfo`, which must
not claim CFO qualification. Collect costs and run online closure at the same
exact checkout SHA. The three-cohort replay must be reaggregated into a new v2
output after deployment; no corrected server report is claimed yet.

## Safety revision (CPU-tested; GPU rerun pending)

Automatic preinstallation is removed from production preparation. An explicit
single-Segment, zero-Prefix diagnostic may preinstall only a complete supplied
resident replica, with all layer events present and no failed/cancelled/pending
transfer. Every destination copy is preceded by a stream wait. The loader also
carries the caller stream dependency into its ready events. Private working KV
remains separate from immutable Source tensors. CPU/SSD completion is not proof
of pre-existing residency. Allocation no longer iterates backing layers merely
to initialize an absent Prefix. Dedicated CPU tests cover wait-before-copy,
idempotence and scope rejection; they are not GPU correctness evidence.

## Read-only server data audit

Server port: 46068. Files inspected under
`/root/autodl-tmp/probekv_stage2/artifacts/schema10-profile-25f0cc1/mistral/`:

- `development_partition.jsonl`: SHA256
  `a46194573147e2df0f28e1201a4600eff1ede3ad4e7cf130cb488fc2f84146d88`.
- `development_cases.jsonl`: SHA256
  `43ac070db0b55bcfe5ece80c4b469bcaff55f7a38bdbcdbb27439a6cc9da6bc1a`.

The cases file contains 283 records. Exact membership in the frozen partition
selects 90: 47 corpus-repeat and 43 mixed-controlled. Every corpus-repeat record
has four distinct historical contexts and four distinct origin example IDs.
The 47 records comprise MuSiQue 17, 2WikiMultiHopQA 13, HotPotQA 17. Segment
lengths range from 27 to 512 tokens; short records must not be silently promoted
to the calibrated canonical length envelope.

Construction is explicitly `corpus_repeat_pseudotime`, not measured production
arrival time. The builder selects distinct contexts by deterministic hash order.
These are corpus-derived candidate opportunities, not deployment hit-rate
evidence. Raw-source membership and exact retokenization remain to be checked.
The 90 retained records have one target per exact token group. They cannot yet
establish winner switching over multiple future targets in a common Source pool.

## Next evidence boundary

1. Validate corpus-repeat provenance and tokenizer against the audited model.
2. Recover additional targets only within the already allowed content groups;
   do not open locked partitions or fabricate arrival history.
3. Freeze a small fixed15 Source-by-target matrix and d1/d2/full-depth shadow.
4. Measure QA and matched timing separately, then causal K=1/2/4 online replay.

No Source-complementarity, production-commit, positive-net-gain or architecture
freeze conclusion is available yet. Missing targets mean insufficient evidence,
not proof that multi-Source has no value. No new GPU job was started by this audit.

## Original-record verification

The CPU-only `audit_multisource_origins.py` checked the 17 frozen MuSiQue
corpus-repeat cases against `data/official/musique-train.jsonl` and the local
Mistral tokenizer revision `c170c708c41dac9275d15a8fff4eca08d52bab71`.
All 17 passed: four distinct historical origin IDs excluding the target,
original document identity, preceding-context rendering, complete parent-token
reconstruction and target question/answers. Only partition-member origins were
normalized/emitted. Output:
`/root/autodl-tmp/probekv_stage2/artifacts/multisource-origin-musique-418bb39-v1.json`.
The report binds raw/partition/case file digests. This is corpus provenance
evidence, not GPU execution, complementarity, chronological production frequency
or selector quality evidence. Other two datasets remain pending original-record
verification. The audit reported an environment OMP_NUM_THREADS warning; it did
not execute CUDA timing.

## Follow-up: remaining datasets and multi-target candidates

Original-record checks also passed for all 13 frozen 2Wiki and 17 frozen HotPotQA
corpus-repeat cases (same Mistral tokenizer). Reports on server:
`multisource-origin-2wiki-c8afd4e-v1.json` and
`multisource-origin-hotpot-c8afd4e-v1.json`. Thus all 47 corpus-repeat candidates
passed origin, context, question/answer and exact parent-token reconstruction.
This supersedes the pending two-dataset status above, but not the usefulness gate.

Within the 17 authorized MuSiQue document groups, recovery found additional
distinct-context targets for 12 groups. Extra-target count distribution is
0:5 groups, 1:4, 2:5, 3:1, 5:1, 7:1 (29 extra records total). Each is after all
four historical Sources under the frozen seed-20260726 pseudo-time ordering.
They exclude existing Source and target example IDs. Recovery does not assign
partition roles or grant execution: exact retokenization and group-isolation
checks must precede a derived development manifest. Report:
`/root/autodl-tmp/probekv_stage2/artifacts/multisource-target-recovery-musique-v2.json`.
The first recovery attempt failed on a string/Path output API mismatch; v2 fixes
the writer, with a fresh output name. No GPU job or qualification was started.
