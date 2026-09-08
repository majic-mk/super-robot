# Mistral native r=1 GPU checkpoint — 2026-09-08

## Outcome and scope

The combined native Prefix + non-prefix Source r=1 primitive now passes the
unchanged contract on the rented A800: greedy IDs equal and **32 common-teacher
logit positions are bitwise equal (relative L2 = 0)**. This was reproduced.

This is a synthetic, single-Segment executor diagnostic, **not** production
Source selection, a QA experiment, a speedup result, or runtime qualification.
The diagnostic explicitly prepares/commits the chosen Source at r=1; it never
uses that forced commit as evidence of passing production economics.

Verified evidence:

| Run (server artifacts directory) | Execution SHA | Backing / first reuse layer | Result |
| --- | --- | --- | --- |
| native-mistral-304bcb2-norm1 | 304bcb244d51e65c8e4304473126f519cfbd313c | pinned CPU / 2 | r1 L2 = 0; all 32 resumable prefill K/V and Prefix shadow layers exact |
| native-mistral-304bcb2-repeat2 | 304bcb244d51e65c8e4304473126f519cfbd313c | pinned CPU / 2 | repeated r1 L2 = 0 without extra layer instrumentation |
| native-mistral-1be6430-cfo1 | 1be643031849cfd6161d16b8ae7d4cea1cfe35ba | pinned CPU / 2 | r1 L2 = 0; actual CFO eager/streaming comparison 32/32 |
| native-mistral-ad77942-ssd-depth8 | ad7794239e5d3c4c4ba5668a78fef0450fbf06f5 | SSD staged / 9 | r1 L2 = 0; transfer and resource checks pass |
| native-mistral-d923c0e-cpu-depth1 | d923c0e7d1e5e9252e434d25a51788053e05fb39 | pinned CPU / 2 | r1 L2 = 0; transfer and resource checks pass |

Every combined request has 448 tokens, 256 actual cached Prefix tokens
(16 native blocks of size 16), a 128-token canonical Segment and a mandatory
dense suffix. Source before / destination / Source after logical digests agree.
The K-hook sentinel covers completed depths {1,2,4,5,8}; the Source commit
boundaries above do not by themselves qualify all production dispatches.

The actual CFO hook compares post-RoPE Q/K with causal masking and GQA mapping
at all 32 layers of the 224-token canonical full-prefill source construction.
The observed max absolute attention-mass error was
1.6278445400530472e-6, below the unchanged 2e-5 contract.

SSD execution used actual layer files, pinned staging and H2D copies; it did
not load a permanently maintained CPU backing. Each transfer carried 16 MiB
of full KV. Observed staging peak was 524288 bytes under the 2 GiB cap (one
completed reusable slot sufficed in this run). This does not demonstrate
maximum double-buffer throughput or behavior under contention. After each
completed diagnostic, active HBM reservations were zero.

## Root causes and changes

1. Native Prefix ownership: composite full-request KV was previously written
   against suffix-only slot mapping, and the partial-query path could enter a
   contiguous-suffix Triton kernel. Independent patch 0009 corrects these.
2. CUDA RoPE ABI: the pinned kernel expects flattened [tokens, heads*dim] K,
   not [tokens, heads, dim]. Patch 0010 fixes both model adapters.
3. BF16 GEMM shape dependence: disabling reduced-precision reductions made
   the first-layer equal-input QKV shape control exact. The flag is explicit.
4. Attention arithmetic: dense default and tensor-bias partial query selected
   different kernels. Patch 0011 permits a shared existing Cutlass/MHA path.
5. Fused RMSNorm: the pinned CUDA kernel changes reduction width at 256 rows.
   Dense (448 rows) and Prefix tail (192 rows) could round differently. Patch
   0012 bounds each existing fused-norm invocation to <=128 rows, preserving
   both in-place outputs. No CUDA extension or new kernel was introduced.
6. CFO observation overhead: bounded block copies replace per-element CUDA
   scalar reads; comparison semantics/order are unchanged.

The first MLP discrepancy was initially localized at gate/up projection.
Further input capture showed the discrepancy already existed at its input:
it was not evidence that changing the MLP GEMM batching alone would fix it.
The subsequent normalization fix removed the full-model discrepancy.

These numerical settings are opt-in, process-local, checked before requests,
and bound to runtime/cost provenance. Historical patch modes are unchanged.
They change the execution stack and may add overhead; **all future matched
baselines and cost measurements must declare this policy**. Old timing tables
must not be reused. No performance claim is supported by these diagnostics.

The original native Prefix-only Triton control still differs numerically
from uncached dense under this policy. It remains separately recorded and is
not substituted for the original dense reference. Passing the composite path
does not establish bitwise equality for every native attention path.

## Evidence and environment

GPU: A800-SXM4-80GB, UUID
GPU-4607665c-9c96-23f6-9c6d-284bf26e552a.
Server: PyTorch 2.2.1+cu121, vLLM 0.4.1, xformers 0.0.25,
Transformers 4.40.2. Local CPU validation uses a different Torch/runtime.

Mistral revision: c170c708c41dac9275d15a8fff4eca08d52bab71.
Current independent patch identity:

- base: b72d7945e6d6306f12be66520196e0f081fa2b0c
- patch SHA256: f63a3d93a37ab593c91141811e9644d9194ddda7ed61722f90f7a2e624d028b6
- tree: 91ec6f46f1cb2cdee29a4c90643ccb1e98c391ba
- actual import: /root/autodl-tmp/probekv_stage2/src/CacheBlend-native-304bcb2/vllm_blend/vllm

All failed runs retain their logs, token/logit observations and failure files.
Source backing stores remain on the server; the separate evidence archive
excludes bulky stores, not raw logit files.

The offline verifier at d923c0e recomputed the two final tier observations
from signed arm JSON and saved tensors. It checks declared inputs, true
commit/Prefix/mask records, file hashes, CFO layer errors and transfer records;
it does not independently rehash GPU memory after the process has exited.

- SSD report digest: 2e0c4bf3fa4b822c698b4057706cc497e66457c5fd24e7654c62cdcc7b42fb71
- CPU report digest: fbf1073453a344af601754b0affa2b4674f4b56675aad0817ed112f51b72b116

Local validation: 685 tests, 684 passed and one ijson-dependent skip;
compileall, contract validator, 22 CLI configurations and diff whitespace
checks passed. These CPU results are not GPU qualification evidence.

## Pending work (do not collapse these into the primitive PASS)

1. Complete preregistered concrete cost operations and exact-support queries,
   using online integrity mode and separate diagnostic hashing accounting.
2. Validate Prefix full/partial coverage, missing-shadow fallback and injected
   transfer failure cleanup on the native execution entry.
3. Run actual dynamic-pool production selection/preparation/FinalCommit and
   observe real admitted reuse, not the diagnostic forced-r1 commit.
4. Freeze trustworthy development partitions/manifests, then execute causal
   K traces, matched QA/TTFT, paired Gate1 A/B and Source Oracle.
5. Validate Qwen independently after the Mistral closed loop; it has not run
   on the GPU in this checkpoint.

Remaining flags are false: online_trace_execution_allowed,
single_request_runtime_sentinel_passed, formal_profile_bundle_frozen,
gpu_runtime_qualified, integrated_concurrency_qualified,
h1_h2_execution_allowed, paper_evidence, locked_test_accessed.

No total GPU budget cap was reintroduced. Per-operation/logged durations are
not cloud billable instance duration. Hourly price and total bill remain
unknown (null, not zero); the instance has not been released or powered off.
