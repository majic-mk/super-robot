# Fast-path safety and multi-Source data audit

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
