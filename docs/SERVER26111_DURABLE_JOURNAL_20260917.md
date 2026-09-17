# Initialization attribution and multi-target readiness

## Attribution (a57aed4)

The new contiguous ledger separates snapshot construction, snapshot hashing,
durable request-start evidence, block/shadow acquisition, inventory and native
input preparation. It changes no gate, source, mask or synchronization rule.

`p0-a57aed4-server26111-512-kv-online-v1` completed 22 replays.
After the prescribed two warm-ups: 11/20 commits, mean TTFT 68.2493474 ms.
Warm request-start durable logging has 68.318 ms and 154.516 ms stalls;
one warm-up has a 288.666 ms stall. Other warm durable-write intervals are
approximately 0.9–1.6 ms. These stalls are inside TTFT and FinalCommit sunk cost.
This identifies durable logging as one source of variance, not every cause
of fallback or an explanation of all historical slow runs.

## Controlled implementation change (29b1bc7)

`--event-journal-directory` is an optional fresh absolute directory. Default
behavior is unchanged. Both paths still fsync each event before continuing.
The external journal is preserved on success or failure; on success the
hash chain is validated and an fsynced, digest-verified copy is archived to
the usual output `events.jsonl`. The summary records source/archive paths,
digest, archive time and full replay-service time including restore,
finalization and archive. No time is subtracted from request TTFT.

Fresh GPU correctness and matched cost probes passed under 29b1bc7.
Two sequential, non-profiled 22-replay diagnostic arms are preregistered:

- `p0-29b1bc7-server26111-512-journal-data-v1`
- `p0-29b1bc7-server26111-512-journal-system-v1`

The latter journals to `/root/probekv-journals/29b1bc7-512-system-v1`.
Both use the same 29b1bc7 cost table and fixed joint-K/V repair configuration.
This sequential diagnostic is not an interleaved paired performance certificate.
Do not promote a default based solely on comparison with a different SHA.

Both arms completed, excluding exactly two warm-ups in each:

| Journal | Warm mean TTFT ms | Commits / 20 | Mean request-start log ms | Maximum request-start log ms |
|---|---:|---:|---:|---:|
| Data disk | 105.71659535 | 5 | 42.64256220 | 229.088338 |
| System disk | 78.58499250 | 5 | 14.54537085 | 111.980544 |

Same measured dense reference: 53.283621 ms. Each arm has zero unaccounted
TTFT ns and one committed sample with realized gamma overrun. Both archives
passed digest checks with 66 events. Full replay-service times (including
restore/finalize/archive) are 86214.922266 and 86846.300519 ms respectively;
these diagnostic reset runs are not production throughput measurements.

Conclusion: moving the journal is insufficient and is **not promoted**.
The observed interval includes hashing, JSON encoding and fsync; the current
ledger does not yet prove which internal operation causes its long stalls.
`p0-29b1bc7-server26111-512-log-profile-v1` is a separate 22-replay cProfile
diagnostic to attribute this, explicitly excluded from performance comparisons.

Function-level attribution subsequently caught replay 14: request-start log
92.659916 ms, with two profiled `posix.fsync` calls totalling 93.987699 ms
over execution (including completion logging). This supports fsync as the
large-stall cause in this observation; it does not make all log work free.

## Finalized-request durability experiment (f0dbb02)

An additional **nondefault** `--event-durability request_finalized` option
retains append/flush for every event but fsyncs at request finalization or
failure. The durability choice is bound in the manifest and raw event chain.
The default, and historical stage journals, remain per-event durable.
Do not describe the nondefault request-start append as power-loss durable:
the inherited `request_started_durable` ledger endpoint denotes log-call
return in this mode; manifest `request_event_fsync_enabled=false` is authoritative.
The evidence transaction is only durable after the finalization fence.
Incomplete or failed prefixes cannot resume as successful requests. All
finalization/archive cost remains in replay-service time; a crashed unfinalized
task is pending/failed, never passed. This is not a gamma or lease bypass.

Regression: 931 tests, 930 passed and one historical skip. A stage-journal
compatibility regression was caught and fixed before deployment. This option
requires new GPU evidence and is not yet a default performance recommendation.

GPU follow-up completed under f0dbb02: correctness/cost probe passed, and
`p0-f0dbb02-server26111-512-finalized-v1` completed 22 replays. Excluding exactly
the first two warm-ups leaves 20 requests, 15 commits, mean TTFT 48.58702505 ms,
median 40.890924 ms, range 39.674547–80.531318 ms. The matched dense cost sample
is 53.133501 ms. Request-start append mean/max is 0.4047461/0.441971 ms.
No committed request has a realized gamma overrun; all ledgers close exactly.
66 raw events archived with SHA256
`bec2b4be358f0ca17c0af862d5ee6bf21aafcb2a8269f31d6102345b4839c970`.
Full diagnostic replay-service time is 83847.353686 ms including finalization,
restore and archive; finalization I/O has not disappeared from service cost.

Remaining rejections are replays 4, 9, 11, 12 and 17. Selection/ready/admission
cost still crosses the frozen gamma boundary; admission intervals for replays
4 and 17 are 5.609/10.013 ms. Do not solve this by dropping those samples or
loosening gamma. This isolated run does not provide paired confidence intervals.
All `quality_passed` fields are null: no matched-quality gain claim is allowed.
The logging experiment is promising for TTFT, but remains explicitly selectable,
not silently substituted for the original per-event durability contract.

## Multi-target full-QA audit (29b1bc7)

`multitarget-qa-geometry-29b1bc7-musique-v1.json` verifies input digests
against the prior recovery proposal and reconstructs original documents and
canonical token slices without truncation. Two MuSiQue groups passed geometry:

| Parent example | Historical Sources | Additional target candidates | Maximum prompt + generation | Sources requiring full-context recapture |
|---|---:|---:|---:|---:|
| 3hop1__394115_99691_28721 | 4 | 7 | 3613 | 2 |
| 2hop__1345_785147 | 4 | 5 | 3152 | 1 |

These are corpus-derived pseudotime candidates, not real production arrival
timestamps. The parent 90-case partition contains calibration roles only;
fit/validation group assignment for the new experiment must be frozen before
quality/threshold measurement. The recovery proposal is not execution approval.

Next: freeze the audited group-disjoint cohort manifests, recapture canonical
Sources under full context, then evaluate every Source's fixed15 QA and cost.
No complementarity, d1/d2 reliability or multi-Source net gain is established
by geometry alone. Qwen, SparseX/QCFuse and formal qualification remain pending.

Local regression: 930 tests, 929 pass and one historical skip; compileall and
contract validator pass. No locked test accessed.
