"""Request-owned immutable position tensors for the resumable native bridge.

Only indices are cached, never hidden states or KV. CPU tuple identity determines
reuse; no device-to-host equality check is needed. At most the current active,
target and full-prompt indices are retained.
"""

POSITION_WORKSPACE_KEY = "probekv_position_workspace"


def request_position_tensors(metadata, active_positions, target_positions, device, torch):
    active_key, target_key = tuple(active_positions), tuple(target_positions)
    n = int(metadata["org_seq_len"])
    shape = (str(device), n)
    old = metadata.get(POSITION_WORKSPACE_KEY)
    if old is None or old["shape"] != shape:
        old = {"shape": shape, "full": torch.arange(n, dtype=torch.long, device=device), "rows": {}}
    previous = old["rows"]
    current = {}
    for key in (active_key, target_key):
        if key not in current:
            current[key] = previous.get(key)
            if current[key] is None:
                current[key] = torch.as_tensor(key, dtype=torch.long, device=device)
    metadata[POSITION_WORKSPACE_KEY] = {"shape": shape, "full": old["full"], "rows": current}
    return current[active_key], current[target_key], old["full"]
