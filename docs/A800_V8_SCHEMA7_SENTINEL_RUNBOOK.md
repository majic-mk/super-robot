# A800 schema-v7 sentinel runbook (future GPU phase)

This runbook is generated during the no-GPU phase; it does not authorize an
experiment automatically.

1. Check a clean checkout of the frozen schema-v7 SHA and the exact CacheBlend
   base/patch tree.
2. Run native Prefix Cache, repair-check off-by-one, and `r=1` dense-equivalence
   correctness sentinels.
3. Run `qualification_full` integrity and prove canonical-before,
   destination-GPU, and canonical-after logical digests match. Record hashing
   and D2H time outside performance TTFT.
4. Validate CFO eager versus streaming attention, including repeated chunk
   occurrences.
5. Validate one-shot SelectionState comparison and deterministic microbatch
   fallback; no non-winner full KV may move.
6. Validate fixed15 first. Then collect d1/d2/legacy/deep-oracle and
   K/V/KV-gradual development traces without freezing a policy.
7. Validate GPU, pre-pinned CPU, staged SSD, and optional GDS transfer paths.
   GDS failure must choose staged SSD.
8. Validate A and C FinalCommitAdmission, partial ready-subset decisions,
   stale-snapshot rejection, and irreversible commits.
9. Stop after the bounded schema-v7 sentinel. Do not run 140-job qualification
   or H1 until all three new Profiles are frozen from real GPU measurements.
