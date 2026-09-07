# Native backend closure: local implementation checkpoint

## Scope

Development branch: `codex/probekv-schema10-evidence-integration`.
This checkpoint continues `bd493b2caecc574f18ba00b73537c1d6dbeefbd2`.
No new algorithm/schema, no server access, no GPU run, no formal Profile or
main-branch merge is authorized by this checkpoint.

The generated handoff records the exact code SHA, validation logs and file
digests. **This is a partial implementation checkpoint, not a GPU-readiness
certificate.** CPU tests cannot certify vLLM/CUDA numerical correctness.

## Implemented code paths

- `v8_schema10_native_adapter.py`: native vLLM allocation, separate FAST/legacy
  adapters, complete legacy checkpoint sequence, resumable execution, actual
  winner readiness, V-only fixed15/r1 support, native decoding and cleanup.
  These use the schema10 dense-selection barrier, not a claim that historical
  A/C deployments have been requalified. Unsupported dispatches are rejected.
- `v8_schema10_inventory.py`: complete/partial Prefix execution ownership,
  canonical boundaries unchanged, and mandatory terminal suffix protection.
  Partial overlap is Prefix plus dense tail, never a freshly rechunked Source.
- `v8_schema10_prefix_shadow.py`: all-layer exact CPU shadows tied to token and
  native block ownership. Missing shadows give native Prefix+dense fallback;
  that fallback is not a passing combined-reuse sentinel. Retained A/B snapshots
  remain charged to the same host capacity.
- `v8_schema10_canonical.py`: independent original-token, no-paged-cache full
  prefill; no nonce or decode. Explicit preregistered
  `capture_original_full_prefill` tasks can instead export the current exact
  full-prefill capture. Prefix-derived and selective-derived captures are
  rejected, including r1 reuse. Canonical build/write is still budgeted.
- `v8_schema10_layer_storage.py` and `v8_schema10_staging.py`: layer-addressable
  BF16 file representation, bounded physical pinned slots, event-fenced reuse,
  actual layer H2D events. Historical torch serialization remains readable.
  SSD staging currently uses a host enqueue loop: no complete compute/I/O
  overlap or fully asynchronous SSD scheduler is claimed.
- Stable execution-shape measurement keys are separate from request/Source/
  scheduler identities. Exact masks, tier, bytes and boundaries remain in
  support keys. New Source IDs may reuse a measured shape; missing shapes are
  `UNSUPPORTED`, never zero or extrapolated costs.
- `v8_schema10_cost_collection.py`: actual CUDA/wall-clock collector, explicit
  reset/warm-up and sample policy, primitive/joint categories, no formal freeze.
- `v8_schema10_qa.py`: generated answers, token IDs, references and scorer
  provenance; QA is recomputed from raw evidence. Without a paired dense answer
  and declared quality contract, `quality_passed` remains null.
- `v8_schema10_native_oracle.py`: sequential fixed-Source diagnostic execution,
  fixed boundary/repair, dense baseline, real answer-F1 and first-token timing,
  three-way digest checking. Diagnostic forced actions are not online policy
  evidence and do not forge a passing FinalCommit. Its named timing scope is
  source-only/executor start, not client-arrival end-to-end performance.
- `prepare_manifest(..., measurement_pending=True)` creates an immutable
  premeasurement blueprint with a **null** cost digest. It cannot execute as an
  online trace. `bind_completed_measurements` derives a child only after actual
  primitive/joint records exist, preserving all jobs and the parent digest.
- Resume revalidates raw event hashes, original bindings and every request's
  start/completion/finalization. Failed/incomplete jobs cannot masquerade as
  an immutable successful prefix.

## Additional safety fixes from review

- GPU Prefix shadows and request working KV are separately accounted for.
- The 2 GiB pinned staging pool and CPU Prefix shadows reduce available backing
  capacity; they do not become free memory outside the common byte budget.
- Final snapshot changes produce dense fallback before commit. No selected
  Source or commit is invented to hide stale scheduling evidence.
- On a failed CUDA fence, HBM reservations are not returned to the allocator;
  a poisoned backend cannot reset over quarantined reservations.
- Actual installed vLLM source-file hashes, model audit, CUDA/library versions,
  single A800 geometry and physical GPU UUID must match the manifest.
- Prerequisite evidence is bound to code, patch, model, tokenizer, config and
  GPU, not merely a same-code `passed=true` from another model.
- No production full-KV cryptographic digest in `online_immutable`. Full source
  and destination verification is an explicit diagnostic/qualification cost.

## Required work still pending

1. Finish the concrete staged operation dispatcher: combined native Prefix+r1/
   mask observations, all required primitive and joint cost operations, support
   validation, then trace/A-B/Oracle and aggregate. The collector and native
   operators are not yet a complete one-command session runner. Existing
   `run_schema10_single_request_sentinel.py` remains the **post-measurement trace
   runner**, not a replacement for these preceding stages.
2. Extend the CPU native-context harness to cover complete model-context
   transitions and injected failures. Current tests primarily cover storage,
   pool orchestration, contracts and fail-closed inputs; they are not numerical
   evidence for the real patched engine.
3. Bind actual model/tokenizer audits and the current-SHA development partition
   to all five task lists. Historical handoffs are not new-SHA qualification.
4. Confirm the Mistral instance, price and budget only in a later authorized GPU
   turn. Do not connect to the running server to fill these gaps now.

The separate local CacheBlend audit replays the frozen patch series on the base
commit and parses the key source files. It confirms source syntax and file
identity only. No patch was changed and no server environment was inspected.

## Follow-up: premeasurement timing and evidence boundary

The continuation after `d3c996c` adds the following source paths. None has been
executed on a server/GPU by this local checkpoint:

- `v8_schema10_stage_journal.py` records ordered premeasurement work against the
  immutable measurement **plan**, with an explicitly null actual measurement
  digest. The online event log still rejects absent measurements. Successful
  prefix recovery checks every result-file digest; failed or unfinished jobs
  cannot resume or advance. File recovery does not reconstruct a live backend.
- Provisional cost provenance can explicitly name a preregistered plan with
  `runtime_profile=null`; it cannot claim a fabricated frozen Profile. Historical
  Profile-bound provenance remains readable.
- Dense-reference, Source-future and joint-future wall-clock cells now require
  an actual first-token endpoint. CUDA samples are explicitly labelled as whole
  operation completion samples; decode is not silently counted as joint TTFT.
- Teacher-forced logit diagnostics execute the declared token count even when
  EOS is predicted, validate teacher length before allocation, and produce no
  QA evidence or quality-pass flag. Production online execution and QA Oracle
  reject diagnostic request switches.
- `v8_schema10_native_preflight.py` adds real operator implementations for
  isolated native Prefix warm/hit/shadow checks and K-hook comparison against an
  independent monolithic full-prefill reference. They require actual CUDA,
  account diagnostic total time, restore retained Prefix state when safe, and
  do not create Source-pool entries or fake passing admission decisions.
  Prefix-only success explicitly does **not** pass the combined r1 sentinel.
- CPU tests exercise actual `NativeRequestContext.finish/close`, EOS and teacher
  semantics, sampling-index restoration, failed-fence quarantine and pre-cost
  journal recovery. This expands the harness but does not complete model-level
  integration or certify BF16 GPU numerical equivalence.

Validation at this follow-up: **656 tests, 655 passed, one skipped (`ijson`)**.
The generated handoff, not this prose, is authoritative for the exact validated
commit and the full command logs. The concrete all-stage dispatcher, combined
r1 operation, complete cost-operation grid and current-SHA data audits remain
pending; readiness flags below remain false.

## Handoff status

Keep `native_runtime_source_ready`, `artifact_preparation_ready`,
`ready_for_single_request_gpu_sentinel`, `online_trace_execution_allowed`,
`single_request_runtime_sentinel_passed`, `gpu_runtime_qualified`,
`h1_h2_execution_allowed`, `paper_evidence` and `locked_test_accessed` false
until their respective prerequisites are satisfied. Passing local tests is
reported separately. Never fill absent model/Profile hashes with placeholders.
