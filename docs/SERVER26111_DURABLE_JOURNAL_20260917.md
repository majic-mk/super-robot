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
