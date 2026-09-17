# Joint K/V repair: primary native experiment contract

New native correctness/cost runs default explicitly to `normalized_kv_deviation`.
The locked launcher forwards the metric; manifests, measurement-plan hashes,
raw cost observations, source-local keys and prepared joint keys bind it.
An old V-only observation cannot generate K/V cost support. Missing metric in
historical manifests/results retains its historical `normalized_v_legacy`
meaning; new runs must not omit the identity.

Winner-only token score is sqrt(delta_K^2 + delta_V^2), with each delta the
FP32 L2 difference divided by the current token's corresponding K or V norm
(floor 1e-12). K is pre-RoPE; V is raw. Stable descending sort breaks ties by
ordered absolute position. Repair uses ceil(r*N) Segment rows, excluding Prefix
and mandatory suffix. Source selection remains K-only and its trim indices
never become the repair mask.

This is an explicit ProbeKV normalized joint metric, not a claim of exact
equivalence to the pinned CacheBlend operator. Existing raw-V and normalized-V
modes remain explicitly selectable historical/ablation controls. External-mask
executor comparisons are labelled external_fixed_mask, not joint-K/V evidence.

The prior 42.48 ms online result belongs to V-only; do not relabel it. Joint
repair requires new-SHA/new-directory r=1, fixed15 mask/numerical checks,
cost collection and online replays. QA on real development requests and
multi-Source quality-oracle evaluation remain pending. No GPU qualification
or paper evidence is granted by this local change.
