# E1 job and result contract

E1 is represented as immutable jobs over:

`case × source × reuse_layer × repair_ratio`.

The primary 15% reuse layer is generated for every included case. A
deterministic stratified 20% subset additionally receives the 10%, 15%, 22%,
30% and 40% layer anchors. Job IDs hash the full semantic identity, including
the model, a canonical digest of the complete case, Source context, split,
layer and repair ratio, so
shards can be rerun or moved between machines without changing identity while
changed manifests cannot accidentally reuse stale results.

## Build and exercise locally

```powershell
$env:PYTHONPATH = "src"
python scripts/build_e1_jobs.py `
  --manifest artifacts/data/cases.jsonl `
  --config configs/local_e1e2.json `
  --output artifacts/e1/jobs `
  --splits train,calibration

0..3 | ForEach-Object {
  python scripts/simulate_e1_shard.py `
    --jobs artifacts/e1/jobs/jobs.jsonl `
    --output artifacts/e1/shards `
    --shard-index $_ `
    --shard-count 4 `
    --resume
}

python scripts/merge_e1_results.py `
  --jobs artifacts/e1/jobs/jobs.jsonl `
  --result-dir artifacts/e1/shards `
  --output artifacts/e1/merged `
  --require-complete
```

The simulator is always non-paper evidence. The pinned CacheBlend worker writes
the same `E1Result` schema. It durably appends and fsyncs each completed
`case x Source` group, not an entire case at once, then releases model-global
KV references. CB gate timing uses two warmup and five measured repetitions;
H1 quality scans may use one deterministic run and never enter a paper
performance table.

## Failure policy

Every latest job result is retained with one of: completed, process crash, GPU
reset, OOM, data error or transient I/O. The merge audit lists missing,
unexpected, failed, duplicate-attempt and retryable job IDs. It also records
the chosen attempt and why older or conflicting attempts were ignored. OOM and
data errors are not silently retried because changing batch size or data
changes the experiment cell; process crash, GPU reset and transient I/O are
retryable with a strictly higher attempt number. Resume rejects rows belonging
to another shard and duplicate attempt numbers.

`result_set_complete` is true only when every expected job has exactly one
unambiguous final completed result, with no missing, unexpected, failed or
duplicate-conflict rows. `run_environment_valid` is evaluated independently
from completeness. `publication_ready` requires both conditions plus
`paper_evidence: true` on every resolved final row. The legacy aggregate field
is an alias:

`paper_evidence == publication_ready`.

A scientifically valid negative result such as "no safe repair ratio" does not
make the result set incomplete. It may cause H1 to fail, but it remains
publishable evidence when the execution and provenance contracts pass.

The generic E1 analysis reports `s0` (K=1) and `last-source` baselines. It does
not call the latter `Latest`: controlled construction has no real timestamp,
and corpus-repeat ordering is only explicitly documented pseudo-time. A true
Latest baseline requires timestamped production traces.

The analysis also reports the deterministic expected Random baseline (the
arithmetic mean cost across all Source choices) and stratifies results by
dataset and construction. Oracle ties are resolved only by stable Source ID;
measured GPU timing is not used to choose the offline Oracle.

## Locked test protection

Job generation defaults to `pilot,train`. The analyzer rejects any test job
unless `--allow-test` is explicitly supplied. A computed H1 diagnostic becomes
paper-claimable only when the complete expected-job audit is publication-ready;
checking only the rows that happened to arrive is insufficient.
