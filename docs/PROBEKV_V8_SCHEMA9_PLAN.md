# ProbeKV v8 schema9: absolute residual admission and online Variant growth

Schema9 keeps the schema8 execution and repair data plane, but closes one
missing loop: a relative Source winner is not automatically a compatible
historical context. The selected Source must also satisfy a model/depth-fixed
absolute Residual-K threshold before Gate1 may freeze it.

## Online decision

At completed depth `d`, rank the compared historical Sources by the existing
trimmed Residual-K score `J_s`. The best Source is compatible only when

```text
J_best(d) <= tau(model, d)
```

At d1, margin/stability, absolute compatibility and Gate1 must all pass. Any
failure continues to d2. At d2, the selector first removes Sources above the
absolute threshold, forms the residual band from the remaining Sources, and
chooses the lowest predicted-cost Gate1-passing Source. Gate1 remains an
economic test; the new threshold is not a replacement for cost admission.

If no d2 Source is compatible, the request executes dense. That dense result
may become a new historical Variant only when every correctness-eligible
candidate was actually compared. A CFO-truncated shortlist cannot prove that
all historical contexts are incompatible.

## Canonical materialization

Only an exact dense full prefill may create a Source Variant. Content misses,
complete-scope absolute mismatch and explicitly budgeted exploration are the
only admission reasons. Gate1 failure, final runtime rejection, slow storage,
comparison-budget truncation and selective repair never create a new Variant.
The initial schema9 protocol also rejects promotion of an `r=1` repair path;
its dense equivalence remains a correctness endpoint, not provenance.

Each exact content bucket stores at most 16 historical Source Variants. When
full, replacement uses the existing value-density policy with LRU tie-break,
while leased, copying, executing and probation Variants remain protected. A
failed safe-victim search rejects materialization. This replacement is not a
CPU/SSD migration: every surviving Variant still owns one canonical BF16
Artifact with one exclusive CPU-or-SSD backing and optional transient GPU
Replica.

## Profile and experiment order

`VariantAdmissionProfile` is independent from SelectionDepth, RepairPolicy
and RuntimeCost profiles. It freezes the Source trim ratio, d1/d2 absolute
thresholds, candidate-coverage rule, materialization budget and exact-dense
provenance contract from the pre-isolated development partition.

- H1 measures historical Source opportunity, online Variant warm-up and
  miss-to-reuse conversion.
- H2 freezes trim ratio, absolute thresholds, d1/d2 versus legacy dispatch,
  SelectionDepthProfile and VariantAdmissionProfile.
- H3 freezes winner-specific repair only among threshold-admitted Sources.
- H4 evaluates churn, write amplification, replacement, tier migration and
  concurrent requests.
- H5 alone may open locked test and compares single-Variant, Cache-Craft-style
  Variant management, multi-Variant without absolute admission, ProbeKV
  schema9 and Oracle admission.

Schema9 fast execution remains disabled until all new Profiles and runtime
qualification pass. Otherwise the independently qualified schema8 legacy
multi-checkpoint/three-gate path, or full dense, is used without mixing
protocols inside a request.
