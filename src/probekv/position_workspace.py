"""Request-owned immutable position tensors for the resumable native bridge.

Only indices are cached, never hidden states or KV. CPU tuple identity determines
reuse; no device-to-host equality check is needed. At most the current active,
target and full-prompt indices are retained.
"""

POSITION_WORKSPACE_KEY = "probekv_position_workspace"


def request_position_tensors(metadata, active_positions, target_positions, device, torch):
    active_key, target_key = tuple(active_positions), tuple(target_positions)
    n = int(metadata["org_seq_len"])
    host_validation = bool(metadata.get("probekv_host_position_validation", False))
    shape = (str(device), n, host_validation)
    old = metadata.get(POSITION_WORKSPACE_KEY)
    if old is None or old["shape"] != shape:
        old = {"shape": shape, "full": torch.arange(n, dtype=torch.long, device=device), "rows": {}}
    previous = old["rows"]
    current = {}
    for key in (active_key, target_key):
        if key not in current:
            current[key] = previous.get(key)
            if current[key] is None:
                if host_validation:
                    # Version-tracked indices even inside model inference_mode.
                    with torch.inference_mode(False):
                        current[key] = torch.as_tensor(key, dtype=torch.long, device=device)
                else:
                    current[key] = torch.as_tensor(key, dtype=torch.long, device=device)
    proofs = {}
    if host_validation:
        for key, tensor in current.items():
            prior = old.get("proofs", {}).get(id(tensor))
            if prior is not None:
                if tensor._version != prior[1]:
                    raise RuntimeError("request position tensor was mutated")
                proofs[id(tensor)] = prior
            else:
                proofs[id(tensor)] = (tensor, tensor._version,
                    all(isinstance(p, int) and not isinstance(p, bool) and 0 <= p < n for p in key)
                    and all(a < b for a, b in zip(key, key[1:])))
    metadata[POSITION_WORKSPACE_KEY] = {"shape": shape, "full": old["full"], "rows": current,
                                        "proofs": proofs}
    return current[active_key], current[target_key], old["full"]


def active_positions_strictly_increasing(metadata, active, torch):
    """Avoid a GPU scalar fence only for owned, unchanged position tensors."""
    if metadata.get("probekv_host_position_validation", False):
        proof = metadata.get(POSITION_WORKSPACE_KEY, {}).get("proofs", {}).get(id(active))
        if proof is not None and proof[0] is active:
            if active._version != proof[1]:
                raise RuntimeError("request position tensor was mutated")
            audit = metadata.setdefault("probekv_position_validation_audit", {})
            audit["owned_host_checks"] = audit.get("owned_host_checks", 0) + 1
            return proof[2]
    audit = metadata.setdefault("probekv_position_validation_audit", {})
    audit["device_checks"] = audit.get("device_checks", 0) + 1
    return bool(torch.all(active[1:] > active[:-1]))
