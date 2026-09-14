"""Recover candidate targets within frozen content groups; never grant execution."""
import argparse
import hashlib
import json
from pathlib import Path
from probekv.io import atomic_write_json
from probekv.rag_data import iter_raw_records, normalize_example, render_preceding_context


def rank_event(event_id):
    return hashlib.sha256(('20260726:' + event_id).encode()).hexdigest()


def recover_targets(cases, raw_rows, dataset):
    by_doc = {}
    for case in cases:
        by_doc.setdefault(case['target_document_id'], []).append(case)
    results = {r['case_id']: [] for r in cases}
    for raw in raw_rows:
        example = normalize_example(dataset, raw)
        for position, doc in enumerate(example.documents):
            for case in by_doc.get(doc.document_id, ()):
                sources = case['sources']
                used_ids = {s['origin_example_id'] for s in sources} | {case['case_id'].split(':')[0]}
                if example.example_id in used_ids:
                    continue
                event_id = example.example_id + ':' + doc.document_id
                if rank_event(event_id) <= max(rank_event(s['context_id']) for s in sources):
                    continue
                context = render_preceding_context(example.documents[max(0, position - 5):position])
                if context in [s['historical_context'] for s in sources] + [case['current_context']]:
                    continue
                results[case['case_id']].append({
                    'origin_example_id': example.example_id, 'event_id': event_id,
                    'pseudo_time_rank': rank_event(event_id), 'parent_case_id': case['case_id'],
                    'content_group': case['group_id'], 'target_document_id': doc.document_id,
                    'current_context': context,
                    'current_suffix_context': render_preceding_context(example.documents[position + 1:]),
                    'question': example.question, 'answers': list(example.answers),
                    'requires_token_and_partition_audit': True,
                    'online_execution_allowed': False})
    for values in results.values():
        values.sort(key=lambda r: (r['pseudo_time_rank'], r['event_id']))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'partition', 'raw', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('fresh output required')
    partition = [json.loads(x) for x in Path(args.partition).read_text().splitlines() if x.strip()]
    if any(r.get('locked_test_accessed') is not False or r.get('partition_role') != 'development_profile_freeze' for r in partition):
        raise ValueError('development-only partition required')
    allowed = {r['case_id'] for r in partition}
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    cases = [r for r in cases if r['case_id'] in allowed and r['dataset'] == args.dataset and r['regime'] == 'corpus-repeat']
    if not cases:
        raise ValueError('no frozen content groups')
    candidates = recover_targets(cases, iter_raw_records(Path(args.raw)), args.dataset)
    hashes = {}
    for name in ('cases', 'partition', 'raw'):
        h = hashlib.sha256()
        with open(getattr(args, name), 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
        hashes[name] = h.hexdigest()
    atomic_write_json(Path(args.output), {'kind': 'multisource_target_recovery_proposal_v1',
        'input_sha256': hashes, 'dataset': args.dataset, 'groups': candidates,
        'groups_with_additional_targets': sum(bool(v) for v in candidates.values()),
        'chronology': 'corpus_derived_pseudotime', 'online_execution_allowed': False,
        'partition_expansion_approved': False, 'paper_evidence': False, 'locked_test_accessed': False})
    print(json.dumps({'groups': len(cases), 'groups_with_additional_targets': sum(bool(v) for v in candidates.values())}))


if __name__ == '__main__':
    main()
