# ProbeKV v8 schema-v7 protocol

Schema-v7 aligns the winner-specific repair data plane with CacheBlend while
retaining ProbeKV's training-free multi-Source selection and request-level
scheduling. It is a new schema; schema-v3 through schema-v6 artifacts remain
readable but cannot unlock schema-v7 runs.

## Selection and repair are different decisions

Source selection compares exact BF16 pre-RoPE K `SourceSelectionState` objects.
For Source `s`, the largest `source_score_trim_ratio` K drifts are removed only
when computing residual score `J_s`. The resulting
`SourceScoreTrimIndices` object is not a repair mask and cannot be passed to the
runtime repair interface.

After one Source is frozen, a dense repair-check layer computes the current K
and V for that winner. The first selective layer is exactly the next layer:

```text
first_selective_reuse_layer = repair_check_completed_depth + 1
```

The winner repair metric is one of K-only, V-only, or normalized K/V. V-only is
the compatibility default. K-only and K/V remain development candidates.

## Depth policies

Four policies are explicit: `d1_only`, `d1_d2_rescue`,
`legacy_multicheckpoint`, and `deep_full_candidate_oracle`. In d1+d2 rescue, a
non-decisive or Gate-1-rejected depth-1 candidate continues to depth 2. It then
freezes a legal economic winner or abstains. Legacy checkpoints remain
Mistral `[1,2,4,5,8]` and Qwen `[1,2,4,5,7]`. The deep oracle runs only as a
development shadow and cannot replace an online frozen winner.

Cache-Craft CFO ranks which candidates receive the comparison budget. It never
selects the final Source. Its metadata and streaming attention implementation
remain the schema-v6 faithful implementation.

## Gradual repair

The initial winner-specific support is the top `ceil(0.15*N)` repair drifts.
Every later support is immutable, digest-linked to its parent, and must be a
subset of the parent. Ratios may decrease from 15% only to a floor certified by
the future RepairPolicyProfile; no token re-entry is allowed. Therefore
`fixed_15` is the only safe fallback before GPU profile freeze.
The Profile must preregister a minimum oracle-recall gate and cannot freeze a
gradual policy whose measured no-reentry recall is below it; no threshold is
silently invented by the no-GPU code.

For each layer and transfer path, the candidate critical path is:

```text
max(next-layer load time, current-layer repair time) + non-overlap time
```

The controller considers only floor-respecting ratios, keeps 15% whenever I/O
hides it, and breaks equal-time ties toward more repair. Its output is only a
candidate for the joint planner, never independent reuse admission.

## Admission and multi-Segment execution

The public chain is:

```text
Residual-K decision
→ Gate1
→ SOURCE_FROZEN + LogicalSourceLease
→ PreparationAdmission
→ winner-only layerwise preparation
→ FinalCommitAdmission
→ REUSE_COMMIT or DENSE
```

Gate1 is Source-local at boundary `d+1`. PreparationAdmission controls leases,
HBM, and bounded speculative waste; it is not another reuse-quality gate.
FinalCommitAdmission preserves the previous refined ready-subset planner but no
longer exposes the Gate3 name. It charges actual sunk work plus one request-level
joint future timeline and requires at most `0.8 * dense-reference`. Unresolved
Segments remain dense fallback. A policy waits for causal selection closure;
C policy can commit ready Segments incrementally. A committed reuse cannot be
rolled back for economic reasons.

## Transfer and integrity

Supported paths are GPU-resident, pinned CPU to GPU, staged SSD through the
global 2-GiB pinned pool, and capability-gated GDS. Failure of the GDS sentinel
falls back to staged SSD and is not a correctness failure.

`qualification_full` hashes canonical Source before transfer, waits for all
ready events, hashes the destination GPU Replica, and hashes canonical Source
after transfer. Hash/D2H time is reported separately. `online_immutable` uses
the creation-time Artifact digest plus generation, read-only interfaces,
leases, refcounts, placement epoch, geometry, and CUDA completion; it performs
no per-request full cryptographic hash. `online_sampled` is deterministic and
quarantines a mismatching Replica. Formal performance uses
`online_immutable`.

## Evidence boundary

No-GPU simulation is never paper evidence and never freezes Selector, Repair,
or Runtime profiles. Locked test data is not read. Schema-v7 GPU qualification
must use new profiles bound to schema 7, the final code SHA, CacheBlend patch
SHA, model revision, tokenizer hash, GPU identity, and real measurement digest.
