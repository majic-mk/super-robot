"""Concrete CUDA backend construction. No diagnostic executor is imported.

Input is an immutable sentinel manifest with a `native_runtime` attachment.
That attachment must itself be covered by the outer manifest hash. This factory
does not rent instances, create Profiles, or approve missing GPU evidence.
"""
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import time

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest, TensorFileSourceStore
from .v8_schema10_cost_provider import EXECUTION_SHAPE_KEY
from .v8_schema10_measured_costs import MeasuredRequestCostProvider
from .v8_schema10_native_adapter import FastNativeOnlineAdapter, LegacyNativeOnlineAdapter, dispatch_depths
from .v8_schema10_online_backend import Schema10OnlineExperimentBackend
from .v8_schema10_staging import PhysicalPinnedStagingPool, PhysicalLayerwiseSourceLoader
from .v8_schema10_prefix_shadow import PrefixShadowStore
from .v8_schema10_pool import Schema10SourcePool
from .v8_schema10_profile import VariantAdmissionProfileV10, PreparationPolicyProfile
from .v8_schema10_selector import Schema10CheckpointSelector
from .v8_schema9_contracts import AbsoluteResidualThreshold
from .model_adapters import SCHEMA6_MODEL_SPECS
from .v8_schema6_hbm import UnifiedHBMReservationManager


def validate_native_attachment(manifest, *, allow_unmeasured=False):
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest_json(unsigned) != manifest.get("manifest_sha256"):
        raise ValueError("native runtime attachment is not covered by manifest digest")
    runtime = manifest.get("native_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("native runtime configuration is missing")
    required = {"model_path", "model_key", "model_audit_path", "model_audit_sha256",
        "cost_provenance", "source_provenance",
        "allocator_capacity_bytes", "prefix_shadow_capacity_bytes", "max_model_len",
        "gpu_memory_utilization", "storage_root", "cpu_backing_bytes", "selector_parameters",
        "sentinel_evidence_paths", "installed_runtime_source_files_sha256"}
    if not required <= runtime.keys():
        raise ValueError("incomplete native runtime configuration: " + str(sorted(required - runtime.keys())))
    if not allow_unmeasured and (not runtime.get("cost_table_path") or not runtime.get("cost_table_sha256")):
        raise ValueError("online factory requires completed actual measurements")
    if runtime["model_key"] not in SCHEMA6_MODEL_SPECS:
        raise ValueError("native model is outside the frozen Mistral/Qwen adapters")
    if runtime.get("repair_policy", "fixed_15") != "fixed_15":
        raise ValueError("this native integration dispatch is fixed15; no silent gradual fallback")
    audit_path = Path(runtime["model_audit_path"])
    if file_digest(audit_path) != runtime["model_audit_sha256"]:
        raise ValueError("native model audit SHA mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = runtime["source_provenance"]
    if source.get("model_id") != runtime["model_key"]:
        raise ValueError("runtime adapter and model namespace differ")
    if (audit.get("model_id") != source["model_id"] or audit.get("revision") != source["model_revision"]
            or audit.get("tokenizer_assets_sha256") != source["tokenizer_hash"]):
        raise ValueError("native model/revision/tokenizer audit differs")
    for relative, sha in audit.get("files", {}).items():
        path = (Path(runtime["model_path"]) / relative).resolve()
        if not path.is_relative_to(Path(runtime["model_path"]).resolve()) or file_digest(path) != sha:
            raise ValueError("model asset changed since audit")
    if not audit.get("files"):
        raise ValueError("model audit contains no actual asset hashes")
    if source["code_commit"] != manifest["binding"]["code_commit"]:
        raise ValueError("native source namespace belongs to another code revision")
    for cost_field, binding_field in (("model", "model_signature"), ("code", "code_commit"),
                                       ("patch", "patch_sha256"), ("config", "config_sha256")):
        if runtime["cost_provenance"].get(cost_field) != manifest["binding"].get(binding_field):
            raise ValueError("cost/manifest provenance differs: " + cost_field)
    return runtime


def verify_installed_runtime_sources(runtime, vllm_root):
    """Verify installed patched code, not a different local checkout/tree."""
    files = runtime["installed_runtime_source_files_sha256"]
    required = {"model_executor/models/llama.py", "model_executor/models/qwen2.py",
                "attention/backends/xformers.py", "worker/model_runner.py",
                "core/block_manager_v1.py", "sequence.py"}
    if not required <= files.keys():
        raise ValueError("installed runtime source audit is incomplete")
    root = Path(vllm_root).resolve()
    for relative, sha in files.items():
        path = (root / relative).resolve()
        if (not path.is_relative_to(root) or not re.fullmatch("[0-9a-f]{64}", sha)
                or file_digest(path) != sha):
            raise ValueError("installed patched runtime source differs: " + relative)


class NativeExperimentBackend(Schema10OnlineExperimentBackend):
    def execute(self, request, *args, **kwargs):
        if not self.costs.sha:
            raise RuntimeError("measurement-only backend cannot start online trace")
        if "teacher_token_ids" in request or request.get("capture_logits") or "correctness_repair_ratio" in request:
            raise ValueError("correctness diagnostics cannot enter the measured online QA path")
        return super().execute(request, *args, **kwargs)

    def install_measurements(self, path, *, expected_sha256, provenance):
        if self.pending or any(a.active for a in self.adapters.values()):
            raise RuntimeError("cannot replace costs during active execution")
        costs = MeasuredRequestCostProvider(path, expected_sha256=expected_sha256, provenance=provenance)
        if costs.key_contract != EXECUTION_SHAPE_KEY:
            raise ValueError("native backend requires execution-shape cost cells")
        self.costs = costs
        for adapter in self.adapters.values():
            adapter.costs = costs

    def set_session_deadline(self, deadline):
        for adapter in self.adapters.values():
            adapter.deadline = deadline

    def verify_prerequisite_evidence(self, manifest):
        from .v8_schema10_event_log import read_events
        paths = manifest["native_runtime"]["sentinel_evidence_paths"]
        required = {"native_prefix", "k_hook", "r1", "source_digest", "absolute_mask"}
        if set(paths) != required:
            raise RuntimeError("native correctness prerequisites are incomplete")
        for name, descriptor in paths.items():
            path = Path(descriptor["path"])
            if file_digest(path) != descriptor["sha256"]:
                raise ValueError("native prerequisite file changed")
            events = read_events(path, binding=descriptor["binding"])
            for field in ("code_commit", "patch_sha256", "model_signature", "model_revision", "tokenizer_hash", "config_sha256"):
                if (not manifest["binding"].get(field)
                        or descriptor["binding"].get(field) != manifest["binding"][field]):
                    raise ValueError("correctness evidence binding differs: " + field)
            if descriptor["binding"].get("gpu_uuid") != manifest["native_runtime"]["cost_provenance"]["gpu"]:
                raise ValueError("sentinel evidence belongs to another physical GPU")
            observations = [e["payload"] for e in events if e["kind"] == "native_correctness_observation" and e["payload"].get("category") == name]
            if not observations or any(e["kind"].endswith("failed") for e in events):
                raise RuntimeError("missing/failed native correctness evidence")
            from .v8_schema10_native_validation import validate_correctness_observation
            for observation in observations:
                validate_correctness_observation(name, observation)
        return True


class UnmeasuredCosts:
    sha = ""
    key_contract = EXECUTION_SHAPE_KEY
    def _lookup(self, *args):
        return None


def create_native_backend(manifest, *, _measurement_only=False):
    runtime = validate_native_attachment(manifest, allow_unmeasured=_measurement_only)
    cost = UnmeasuredCosts() if _measurement_only else MeasuredRequestCostProvider(runtime["cost_table_path"],
        expected_sha256=runtime["cost_table_sha256"], provenance=runtime["cost_provenance"])
    if cost.key_contract != EXECUTION_SHAPE_KEY:
        raise RuntimeError("historical identity cost tables cannot drive the new native backend")
    import torch
    import vllm
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError("native sentinel requires confirmed CUDA compute capability 8.0")
    if not str(torch.__version__).startswith("2.2.1") or str(vllm.__version__) != "0.4.1":
        raise RuntimeError("local testing stack is not the frozen server stack")
    import xformers
    import transformers
    if (torch.version.cuda != "12.1" or xformers.__version__ != "0.0.25"
            or transformers.__version__ != "4.40.2"):
        raise RuntimeError("CUDA/xformers/tokenizer stack differs from server lock")
    verify_installed_runtime_sources(runtime, Path(vllm.__file__).parent)
    properties = torch.cuda.get_device_properties(0)
    if (torch.cuda.device_count() != 1 or not re.fullmatch(r"NVIDIA A800.*80GB", properties.name)
            or properties.total_memory < 80_000 * 1024**2):
        raise RuntimeError("single A800 80GB hardware contract differs")
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                    text=True, timeout=10).splitlines()
    if len(uuids) != 1 or uuids[0].strip() != runtime["cost_provenance"]["gpu"]:
        raise RuntimeError("cost collection provenance does not name the actual single GPU")
    from vllm import LLM
    spec = SCHEMA6_MODEL_SPECS[runtime["model_key"]]
    llm = LLM(model=runtime["model_path"], tokenizer=runtime["model_path"], dtype="bfloat16",
        max_model_len=runtime["max_model_len"], gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=False)
    # Explicit capacity is reserved by the caller after native runtime
    # allocation. cudaMemGetInfo is only a cross-check, not the capacity source.
    free, _ = torch.cuda.mem_get_info()
    if runtime["allocator_capacity_bytes"] > free:
        raise MemoryError("declared ProbeKV reservation exceeds actual unallocated HBM")
    hbm = UnifiedHBMReservationManager(allocator_capacity_bytes=runtime["allocator_capacity_bytes"])
    staging = PhysicalPinnedStagingPool()
    source = runtime["source_provenance"]
    cfg = runtime["selector_parameters"]
    thresholds = tuple(AbsoluteResidualThreshold(int(d), float(v)) for d, v in cfg["thresholds"])
    template = VariantAdmissionProfileV10(code_commit=source["code_commit"],
        cacheblend_patch_sha256=manifest["binding"]["patch_sha256"], model_id=source["model_id"],
        model_revision=source["model_revision"], tokenizer_hash=source["tokenizer_hash"],
        source_residual_trim_ratio=cfg["source_residual_trim_ratio"], thresholds=thresholds)
    if not set(spec.checkpoints) <= {t.completed_depth for t in thresholds}:
        raise ValueError("native legacy path lacks preregistered absolute thresholds")
    owner = {}
    def store_factory(capacity, budget):
        # Prefix shadows are charged to the same global host budget for all K.
        cpu = runtime["cpu_backing_bytes"]
        shadow = runtime["prefix_shadow_capacity_bytes"]
        auxiliary = shadow + staging.capacity_bytes
        if not 0 < auxiliary < cpu < budget:
            raise ValueError("invalid explicit CPU/SSD/shadow partition")
        pool = Schema10SourcePool(profile=replace(template, max_variants_per_content=capacity))
        pool.activate_namespace(source["model_signature"])
        store = TensorFileSourceStore(pool, Path(runtime["storage_root"]) / str(time.perf_counter_ns()),
            cpu_bytes=cpu - auxiliary, ssd_bytes=budget - cpu, pin_cpu=True)
        store.auxiliary_host_bytes = auxiliary
        return store
    def authorize(**kwargs):
        backend = owner["backend"]
        sid, source_id = kwargs["segment_id"], kwargs["source_id"]
        obj = backend.store.objects[source_id]
        pool = backend.store.pool
        row = next(v for v in pool._variants.values() if v.source_variant_id == source_id)
        if (not pool.logical_lease_counts.get(source_id) or not any(p.busy for p in row.healthy_backing_replicas)
                or not any(not r.released and r.segment_id == sid and r.bytes >= kwargs["bytes_required"]
                           for r in hbm.reservations.values())):
            raise RuntimeError("full-KV transfer lacks live Source/Replica/HBM ownership")
    loader = PhysicalLayerwiseSourceLoader(staging, authorize=authorize,
        integrity_mode=runtime.get("integrity_mode", "online_immutable"))
    layer = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].self_attn
    shadows = PrefixShadowStore(model_signature=source["model_signature"], num_layers=spec.num_layers,
        kv_heads=layer.num_kv_heads, head_dim=layer.head_dim, capacity_bytes=runtime["prefix_shadow_capacity_bytes"])
    shared = {"active": None, "warm_history": [], "generation": 1}
    adapters = {path: (LegacyNativeOnlineAdapter if path == "legacy_multicheckpoint" else FastNativeOnlineAdapter)(llm=llm, model_spec=spec, selection_path=path, loader=loader,
        hbm=hbm, shadow_store=shadows, store_provider=lambda: owner["backend"].store, provenance=source,
        cost_provider=cost, shared_runtime_state=shared) for path in ("d1_only", "d1_d2_rescue", "legacy_multicheckpoint")}
    def selector_factory(dispatch, capacity):
        return Schema10CheckpointSelector(variant_profile=replace(template, max_variants_per_content=capacity),
            preparation_profile=PreparationPolicyProfile(code_commit=source["code_commit"], model_id=source["model_id"],
                runtime_policy="dense_selection_barrier", gate1_mode=dispatch["gate1_mode"]),
            strong_margin=cfg["strong_margin"], stable_margin=cfg["stable_margin"],
            residual_band_relative_tolerance=cfg["residual_band_relative_tolerance"],
            checkpoint_depths=dispatch_depths(dispatch["selection_path"], spec))
    backend = NativeExperimentBackend(store_factory=store_factory, adapters=adapters, selector_factory=selector_factory,
        cost_provider=cost, hbm_manager=hbm, provenance=source)
    owner["backend"] = backend
    backend.capabilities = {**backend.capabilities, "real_cuda_native_online_backend": True}
    return backend


def create_native_measurement_backend(manifest):
    """Preflight/capture can start without inventing a runtime Profile SHA."""
    return create_native_backend(manifest, _measurement_only=True)
