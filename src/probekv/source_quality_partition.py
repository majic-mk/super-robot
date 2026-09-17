"""Deterministic, pre-outcome content-group split for a small QA pilot."""
from .v8_schema10_execution import digest_json


def freeze_group_roles(groups, *, seed=20260726):
    """Reject cross-role shared origins/content, rather than leaking examples."""
    if len(groups) < 2:
        raise ValueError("at least two independent content groups required")
    ids = [g['group_id'] for g in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate content group")
    origins, contents = set(), set()
    for group in groups:
        members = group['source_origin_ids'] + group['target_origin_ids']
        if len(group['source_origin_ids']) != 4 or len(group['target_origin_ids']) < 5:
            raise ValueError("four historical sources and at least five targets required")
        if len(set(members)) != len(members) or origins.intersection(members):
            raise ValueError("shared origin would leak across groups or history/targets")
        if group['content_key'] in contents:
            raise ValueError("shared content must belong to one group")
        origins.update(members)
        contents.add(group['content_key'])
    ordered = sorted(ids, key=lambda g: (digest_json([seed, g]), g))
    # Pilot only: a held-out group is not a statistically qualified profile.
    n_validation = max(1, len(ordered) // 3)
    validation = set(ordered[:n_validation])
    return {g: 'validation' if g in validation else 'fit' for g in sorted(ids)}
