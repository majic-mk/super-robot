# Protocol v5: one cost identity, one Source selection

Protocol v5 preserves the v4 execution order and adds a single component cost
identity shared by preliminary Source filtering and refined reuse admission.
Protocols v3 and v4 remain available for artifact reproduction.

## State machine

1. At every probe checkpoint, quality-safe Sources are filtered with predicted
   component-cost upper bounds.
2. Before `L_probe_max`, interval separation may immediately lock a Source.
   Protocol v5 does not force selection to wait for the final checkpoint.
3. At `L_probe_max`, the configured final selector either locks exactly one
   Source or abstains.
4. Abstention executes full recomputation without loading any historical
   Source.
5. A selected Source is immutable for the rest of the request. Loading,
   scheduling and refined measurement must all return that same Source ID.
6. Refined admission may only accept reuse or reject it and execute full
   recomputation. It cannot switch to another, latest or default Source.

## Shared accounting identity

Both stages use the same request-arrival to first-token-ready identity:

```text
reuse_total =
    probe
  + metadata
  + compare
  + visible_load
  + post_ready_blocking
  + interference_charge
  + repair_selection
  + repair
  + remaining_layers
```

The preliminary stage fills this structure with predicted lower/upper values.
The refined stage replaces them with scheduler observations and costs evaluated
at the actual reuse boundary. The origin, endpoint and interference accounting
mode must match, while component values and the boundary may differ.

`included_in_load` means measured load already includes contention and
`load_interference_ms` is diagnostic only. `explicit_penalty` means load is an
interference-free base and the interference penalty is added exactly once.

`full_total_ms` uses the same origin and endpoint. Admission is:

```text
reuse_total_ms <= gamma * full_total_ms
```

## Migration

`legacy_aggregate` accepts the old scalar `cost_lower_ms/cost_upper_ms` and
ambiguous aggregate `repair_ms`. `unified_components_v1` requires paired
predicted component bounds on every candidate and a two-stage refined
controller. `cost_breakdown_from_total` exists only to migrate old predictors;
hardware predictors must eventually populate the individual components.

The main local v5 configuration is `configs/local_system_v5.json`.
