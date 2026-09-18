"""CPU-only proposal from unused calibration groups, never execution permission."""
import argparse
from collections import Counter
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.manifest import manifest_case_from_row
from probekv.rag_data import iter_raw_records, normalize_example, segment_text
from probekv.source_support_span import classify_support
from probekv.v8_profile_manifest import canonicalize_v8_profile_cases
from recover_multisource_targets import recover_targets


def unused_calibration(rows, partition, dataset):
    if any(r.get('partition_role') != 'development_profile_freeze' or
           r.get('locked_test_accessed') is not False for r in partition):
        raise ValueError('original development partition required')
    excluded = {r['group_id'] for r in partition}
    return [r for r in rows if r.get('split') == 'calibration'
            and r.get('regime') == 'corpus-repeat' and r['dataset'] == dataset
            and r['group_id'] not in excluded]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'partition', 'raw', 'raw-sha256', 'dataset', 'tokenizer', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError('new census output required')
    if sha256_file(Path(a.raw)) != a.raw_sha256:
        raise ValueError('official training source hash mismatch')
    read = lambda path: [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    parents = unused_calibration(read(a.cases), read(a.partition), a.dataset)
    if not parents:
        raise ValueError('no unused calibration groups')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, use_fast=True)
    tokenizer_sha = sha256_file(Path(a.tokenizer)/'tokenizer.json')
    canonical = canonicalize_v8_profile_cases(
        [manifest_case_from_row(r) for r in parents], tokenizer_signature=tokenizer_sha,
        document_revision=a.raw_sha256, decode=lambda ids: tok.decode(list(ids)))
    cases = [c.to_row() for c in canonical]
    wanted_docs = {c['target_document_id'] for c in cases}
    relevant = []
    for raw in iter_raw_records(Path(a.raw)):
        ex = normalize_example(a.dataset, raw)
        if any(d.document_id in wanted_docs for d in ex.documents):
            relevant.append(raw)
    targets = recover_targets(cases, relevant, a.dataset)
    originals = {normalize_example(a.dataset, r).example_id:r for r in relevant}
    rows = []
    for case in cases:
        labels = []
        left = case['canonical_parent_left_token_ids']
        parent = left + case['segment_token_ids'] + case['canonical_parent_right_token_ids']
        for target in targets[case['case_id']]:
            raw = originals[target['origin_example_id']]
            ex = normalize_example(a.dataset, raw)
            docs = [d for d in ex.documents if d.document_id == case['target_document_id']]
            if len(docs) != 1:
                raise ValueError('ambiguous repeated document')
            encoded = tok(segment_text(docs[0]), add_special_tokens=False, return_offsets_mapping=True)
            error = None
            try:
                label = classify_support(raw, docs[0], parent_token_ids=parent, encoded=encoded,
                    token_start=len(left), token_end=len(left)+len(case['segment_token_ids']))
            except ValueError as exc:
                label, error = 'audit_rejected', str(exc)
            labels.append(dict(origin_example_id=ex.example_id, support_stratum=label,
                               rejection_reason=error, pseudo_time_rank=target['pseudo_time_rank']))
        rows.append(dict(case_id=case['case_id'], group_id=case['group_id'],
            source_origin_ids=[s['origin_example_id'] for s in case['sources']],
            token_count=len(case['segment_token_ids']), targets=labels,
            counts=dict(Counter(r['support_stratum'] for r in labels))))
    report = dict(kind='unused_calibration_support_census_v1', dataset=a.dataset,
        input_sha256={k:sha256_file(Path(getattr(a,k))) for k in ('cases','partition','raw')},
        tokenizer_json_sha256=tokenizer_sha, parent_groups=len({r['group_id'] for r in parents}),
        canonical_segments=len(cases), rows=rows,
        counts_are_case_target_pairs_not_independent_requests=True,
        source_geometry_verified=False, cross_group_origin_disjointness_verified=False,
        chronology='corpus_derived_pseudotime', outcome_based_sampling=False,
        gpu_execution_allowed=False, paper_evidence=False, locked_test_accessed=False)
    atomic_write_json(Path(a.output), report)
    print(json.dumps(dict(dataset=a.dataset, groups=report['parent_groups'], segments=len(cases),
        segments_with_five_full_support_targets=sum(r['counts'].get('full_support_sentence',0)>=5 for r in rows),
        groups_with_full_support_target=len({r['group_id'] for r in rows if r['counts'].get('full_support_sentence',0)}))))


if __name__ == '__main__':
    main()
