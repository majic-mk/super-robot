"""CPU-only audit of frozen corpus-repeat members against original QA records.

Does not assign new partitions, load a model, or certify Source usefulness.
"""
import argparse
import hashlib
import json
from pathlib import Path

from probekv.rag_data import iter_raw_records, normalize_example, render_preceding_context, segment_text
from probekv.io import atomic_write_json


def audit_case(case, examples, encode):
    failures = []
    target_id = case['case_id'].split(':')[0]
    sources = case['sources']
    origin_ids = [s['origin_example_id'] for s in sources]
    if len(set(origin_ids)) != 4 or target_id in origin_ids:
        failures.append('origins_not_four_distinct_prior_examples')
    contexts = []
    for identifier, source in [(s['origin_example_id'], s) for s in sources] + [(target_id, None)]:
        example = examples.get(identifier)
        if example is None:
            failures.append('missing_origin:' + identifier)
            continue
        matches = [(i, d) for i, d in enumerate(example.documents)
                   if d.document_id == case['target_document_id']]
        if len(matches) != 1:
            failures.append('missing_exact_document:' + identifier)
            continue
        position, doc = matches[0]
        expected = render_preceding_context(example.documents[max(0, position - 5):position])
        observed = source['historical_context'] if source else case['current_context']
        if expected != observed:
            failures.append('prefix_mismatch:' + identifier)
        contexts.append(expected)
        parent = list(encode(segment_text(doc)))
        reconstructed = (case.get('canonical_parent_left_token_ids', [])
                         + case['segment_token_ids']
                         + case.get('canonical_parent_right_token_ids', []))
        if parent != reconstructed:
            failures.append('token_mismatch:' + identifier)
        if source is None and (example.question != case['question'] or list(example.answers) != case['answers']):
            failures.append('qa_mismatch:' + identifier)
    return {'case_id': case['case_id'], 'origin_ids': origin_ids,
            'target_origin_id': target_id, 'distinct_contexts': len(set(contexts)),
            'segment_tokens': len(case['segment_token_ids']),
            'origin_token_verification_passed': not failures, 'failures': failures,
            'chronology': 'corpus_derived_pseudotime', 'production_frequency_verified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('partition', 'cases', 'raw', 'dataset', 'tokenizer', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('fresh output required')
    partition = [json.loads(x) for x in Path(args.partition).read_text().splitlines() if x.strip()]
    if any(r.get('locked_test_accessed') is not False or r.get('partition_role') != 'development_profile_freeze' for r in partition):
        raise ValueError('frozen development partition required')
    allowed = {r['case_id'] for r in partition}
    rows = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r['case_id'] in allowed and r['dataset'] == args.dataset
            and r['regime'] == 'corpus-repeat']
    if not rows:
        raise ValueError('no allowed corpus-repeat cases')
    required = {r['case_id'].split(':')[0] for r in rows}
    required.update(s['origin_example_id'] for r in rows for s in r['sources'])
    examples = {}
    # Read original train records only to verify already authorized origins.
    # Never relabel other records as development or emit their contents.
    for raw in iter_raw_records(Path(args.raw)):
        identifier = str(raw.get('id', raw.get('_id', '')))
        if identifier in required:
            if identifier in examples:
                raise ValueError('duplicate raw origin')
            examples[identifier] = normalize_example(args.dataset, raw)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    results = [audit_case(r, examples, lambda text: tokenizer.encode(text, add_special_tokens=False)) for r in rows]
    def digest(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
        return h.hexdigest()
    atomic_write_json(output, {'kind': 'frozen_multisource_origin_audit_v1',
        'inputs': {k: {'path': getattr(args, k), 'sha256': digest(getattr(args, k))} for k in ('partition', 'cases', 'raw')},
        'dataset': args.dataset, 'cases': results, 'case_count': len(results),
        'verified_cases': sum(r['origin_token_verification_passed'] for r in results),
        'source_complementarity_verified': False, 'production_frequency_verified': False,
        'gpu_runtime_qualified': False, 'paper_evidence': False, 'locked_test_accessed': False})
    print(json.dumps({'case_count': len(results), 'verified_cases': sum(r['origin_token_verification_passed'] for r in results)}))


if __name__ == '__main__':
    main()
