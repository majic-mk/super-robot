# Full-context multi-Source QA pilot

## Frozen preparation evidence

- Implementation commit: `ae2a26d`.
- Server: 26111; isolated checkout `ProbeKV-ae2a26d-server26111`.
- Artifact: `/root/autodl-tmp/probekv_stage2/artifacts/multitarget-qa-pilot-ae2a26d-v1.json`.
- Canonical JSON partition digest: `c64e6763fd2dba66c8fa3d12514964a1f51e9cddb084892ebd9ce1b280fcdd18`.
- Two MuSiQue content groups, each with four distinct full historical prefixes;
  twelve additional target requests in total. One fit group, one validation group.
- Split fixed by seed 20260726 and content-group identity, before answer outcomes.
- Parent development/recovery/raw input hashes checked. Shared origins across groups,
  source/target overlap, duplicate content groups, changed canonical tokens, missing
  answers and overlength contexts are rejected.
- Full original document context and original answers retained. No truncation.
- Chronology is corpus-derived pseudotime, NOT observed production request time.
- Local regression: 933 tests, 932 passed, one historical skip; contract valid.

## Interpretation

This is a mechanism pilot, not a population estimate of multi-Source frequency,
not a 90-request profile and not a quality certificate. GPU was idle during this
CPU-only preparation. No Source QA matrix has been executed for this partition.

## Next execution requirements

1. Recapture all four sources per group using complete exact dense full prefill.
   Do not reuse older captures that omitted original preceding/following documents.
2. Bind an independently audited native runtime to the execution SHA. Check K/V
   repair metric, model/tokenizer, patch and source integrity.
3. Use fixed 15% normalized K/V repair and common first reuse layer 9 for the
   source-quality isolation matrix: dense plus each of four sources per target.
   Forced diagnostic source execution is not a production FinalCommit result.
4. Capture d1/d2 and legacy depth scores separately; existing d1/d2-only capture
   cannot be labeled a deep oracle. Actual quality oracle uses answer F1, with
   ties retained; fastest acceptable source must satisfy dense-relative QA.
5. Preserve per-target answers, token IDs, masks, digests, timings and failures.
   Report source-only diagnostic timing separately from complete online TTFT.
6. Do not tune on the validation group. Do not claim complementarity or net gain
   until measured. K=1/2/4 causal replay remains subsequent work.

All formal profile, qualification, H1-H5 and locked-test permissions remain false.

## Native matrix launched

- Runner commit: `77d3aafbdfd1878cbee52eb92c9b07ee96088e858`.
- Regression: 935 tests, 934 passed, one historical skip.
- Independent environment lock: server26111
  `/root/autodl-tmp/probekv_stage2/artifacts/qa-env-77d3aaf-v1/environment_lock.json`.
- Fresh runtime manifest generated without inventing measured costs. The matrix
  uses the explicitly diagnostic measurement backend, not production admission.
- Output: `/root/autodl-tmp/probekv_stage2/artifacts/qa-matrix-77d3aaf-v1`.
- Log: same path plus `.log`.
- Started September 17, 2026, approximately 19:58 Asia/Shanghai.
- First two complete source captures published. First capture: zero cached Prefix,
  no paged-cache writes, no decode, capture host time 82,360.092717 ms.
- Legacy canonical capture still collects CFO attention metadata despite CFO not
  being required by the online selector. This substantial construction cost must
  remain in total cost accounting. It is not source-comparison TTFT and is a
  separate cleanup candidate, not a reason to modify a running checkout.
- At this record time no target QA matrix had completed. Do not infer quality,
  complementarity or speedup from successful source construction.

## CFO-free recapture and completed first matrix

The previous job was interrupted at the user's request; its original outputs
and KeyboardInterrupt trace inside CFO aggregation are retained.

- Execution SHA: `c3e295352d9bfc44a46d537a2c841edf5288aa38`.
- Fresh environment: `qa-env-c3e2953-v1`; fresh outputs: `qa-matrix-c3e2953-v1`
  under server26111 `/root/autodl-tmp/probekv_stage2/artifacts`.
- Ordinary canonical construction and current-request exact-full-prefill export
  no longer collect CFO. Explicit bounded eager CFO diagnostics and historical
  metadata validation remain available. Missing CFO uses an explicit
  `cfo_collection=not_collected` marker, not fabricated statistics.
- First group's four complete Source captures: 279.061, 251.392, 237.685,
  271.840 ms. Prior first-source capture was 82,360.093 ms. These are construction
  observations, not paired online TTFT performance evidence.
- Both groups completed: 8 newly captured sources, 12 targets, each with dense
  and four fixed15 normalized-K/V actions (60 QA actions), common boundary 9.
- Whole diagnostic job: 77.848452 seconds. Integrity assertions passed or the
  runner would have stopped; this is not a production FinalCommit qualification.

### Answer-boundary problem: selector verdict remains inconclusive

11/12 targets have at least one action continuing into a new `Question:`.
Example dense output begins with the correct reference `Asia`, then generates
another question; scoring the complete 32-token continuation gives F1 0.105263.
Thus current raw F1 rankings mix answer correctness with unwanted continuation.
Do not retrospectively truncate answers and call this the preregistered result.

For debugging only, residual-trim-0.15 full-candidate d1 argmin matched the raw
maximum-source-F1 tie set on 5/12 targets; d2 argmin on 4/12. These counts include
all-zero ties, are NOT the d1/d2 early-exit policy accuracy, and cannot establish
selector reliability or failure under an adequate QA contract. Deep oracle was
not measured. No net-gain or multi-Source Go decision is authorized.

Next: freeze an answer-boundary/stop contract shared by dense and every Source,
test it independently, and rerun to a new directory. Preserve all current raw
answers and do not tune selector thresholds on this validation pilot.
