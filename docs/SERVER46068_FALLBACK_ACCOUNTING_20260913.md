# Server 46068: final-rejection copy cancellation and exact TTFT accounting

Code: `e244145ad98c83b22d706a722461ab97dc730202`.
GPU: A800-SXM4-80GB, `GPU-936c826b-dd49-6d32-53f3-6640860c5e47`.
No operation was performed against the previous server in this run.

## Evidence and changes

- Independent clean-base patch reconstruction: 0001–0013, SHA256
  `72687e66e33f7b78984cfb6dfdd438f861cf36de3187eb6d3274a21b6672744c`.
- Correctness/cost: `/root/autodl-tmp/probekv_stage2/artifacts/native-e244145-server46068-v38`.
- Three independent same-pool-state replays: `/root/autodl-tmp/probekv_stage2/artifacts/online-e244145-server46068-v39`.
- Local copies of raw outcomes/events: `artifacts/server46068-e244145-fallback/`.
- Local regression: 853 tests run, 852 passed, 1 skipped; compileall and contract validator passed.

Final online rejection now marks uncommitted load tickets cancelled. Future
prefetch calls cannot enqueue more layers; submitted events/tensors, physical
leases and HBM reservations remain owned until the existing completion fence.
Committed Sources are unaffected. Diagnostic r=1 remains an independent path.

All three replays cancelled layers 3–32 (62,914,560 bytes not submitted). Layers
1–2 had already been submitted. No full-source ready or integrity certification
is claimed for the cancelled partial replica. Output overlap traces only contain
the initially submitted next-layer copy, not copies from the rejected tail.

## Actual results, including the slow sample

Matched native Prefix+dense reference: **53.464462 ms** on this GPU. Older-server
53.29 ms results are not a matched A/B control for this run.

| Replay | Complete TTFT (ms) | Context-open to selection-closed (ms) | Finish-entry to remaining-prefill-submitted (ms) | Commit |
|---|---:|---:|---:|---|
| 0, cold selector | 115.857086 | 38.481081 | 62.201140 | none |
| 1, after prior replay | 181.213201 | 39.782973 | 128.731249 | none |
| 2, after prior replay | 84.284996 | 10.589231 | 62.458321 | none |

Each outcome includes `request_wallclock`: chronological, contiguous intervals
covering arrival through the first-token callback. Sum of integer durations
equals TTFT exactly; `unaccounted_ns=0` for all three. These are host wall-clock
intervals, NOT additive GPU kernel service times. `remaining_prefill_submitted`
does not itself fence GPU completion; subsequent argmax host read supplies the
first-token completion endpoint. No additional timing synchronization was added.

Example complete replay-2 partition (ms):

| Interval | Duration |
|---|---:|
| Arrival to service start | 0.019044 |
| Service start to context open | 4.254630 |
| Context open to selection closed | 10.589231 |
| Selection closed to ready check | 0.000110 |
| Ready check | 1.109015 |
| Ready end to final admission end | 4.491363 |
| Cancellation and finish dispatch | 0.032819 |
| Finish-call entry | 0.001657 |
| Remaining resumable prefill submission | 62.458321 |
| Native prefill bookkeeping | 1.007443 |
| Logits submission | 0.189583 |
| Argmax host result | 0.130928 |
| First-token callback | 0.000852 |
| **Total** | **84.284996** |

## Conclusions and next bounded issue

Copy cancellation is demonstrated, not an end-to-end speedup. All replays remain
economically rejected; `online_closed_loop_passed=false`. The replay-1 spike is
preserved, not removed as an outlier. Its cause is not yet established.

Next inspect the ~62 ms remaining resumable prefill and the 128.73 ms outlier,
using separate host/kernel attribution. Do not assume SelectionState H2D caused
all cold comparison latency, and do not call the full-KV hot-cache diagnostic a
GPU-resident SelectionState cache. No new SparseX/QCFuse or multi-Segment arm is
introduced by these fixes. Formal profile, GPU qualification, paper evidence and
locked-test access remain false.

## Follow-up v40: native continuation candidate rejected

Code `0b247e26a34404e19692caeb706cad4d97fd9d8f` adds an opt-in
`--native-dense-continuation` diagnostic. It preserves already computed hidden
states and residuals and executes remaining dense layers using status=0 native
Prefix attention, guarded against any selective commit or shortened active rows.
It is NOT the production default. Local regression: 855 run, 854 pass, 1 skip.

The GPU numerical gate failed and stopped before cost/online execution:

- Original r=1 reuse-vs-no-cache-dense aggregate logit relative L2 remained 0.
- Native continuation-vs-no-cache-dense: aggregate **0.01047809**, maximum
  per-position **0.04492869**; both exceed 1e-4.
- Greedy tokens agreed for this synthetic request; that does not override the
  failed teacher-logit criterion.
- Existing native Prefix control (both v38 and v40) differs from no-cache dense:
  aggregate **0.00757023**, maximum per-position **0.02121325**. Earlier logs
  recorded this as a control rather than a passing equivalence certificate.
- Original resumable Prefix control in v38 had zero difference. Switching to
  native attention mid-prefill is therefore not a numerically equivalent
  optimization under the frozen tolerance, despite identical row ownership.

No threshold was relaxed and no performance result is accepted from v40. Raw
comparison and arm JSONs are copied to `artifacts/server46068-native-continuation-v40/`;
the server directory retains teacher tensors and their digests. The candidate
remains opt-in and failed, not enabled in standard requests.

The previous 128.73 ms remaining-prefill outlier is localized to layer 2's
68.115 ms event envelope (other layers about 1.9 ms). An event envelope includes
possible host launch gaps and is not proof of 68 ms kernel work. Further work
must profile host gaps and remove redundant preparation while preserving the
original attention numerical path; it must not silently replace that path with
native Prefix attention on the strength of greedy-token agreement alone.
