"""Outcome-independent document-support stratification for frozen QA pilots."""
import argparse
import hashlib
import json
from pathlib import Path
from probekv.io import atomic_write_json
from probekv.rag_data import iter_raw_records, normalize_example
from probekv.v8_schema10_execution import digest_json


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'cases', 'raw', 'output'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('fresh relevance audit required')
    pilot = json.loads(Path(args.pilot).read_text())
    if (digest_json({k:v for k,v in pilot.items() if k != 'partition_sha256'}) != pilot['partition_sha256'] or
            pilot['locked_test_accessed'] is not False):
        raise ValueError('invalid pilot identity')
    for key in ('cases', 'raw'):
        if file_sha(getattr(args, key)) != pilot['input_sha256'][key]:
            raise ValueError('original input SHA mismatch: '+key)
    cases = {c['case_id']:c for c in (json.loads(x) for x in Path(args.cases).read_text().splitlines())}
    wanted = {q['origin_example_id']:(g,q) for g in pilot['groups'] for q in g['target_requests']}
    rows = []
    for raw in iter_raw_records(Path(args.raw)):
        oid = str(raw.get('id', raw.get('_id','')))
        if oid not in wanted:
            continue
        group, request = wanted[oid]
        example = normalize_example(pilot['dataset_scope'], raw)
        if digest_json(example.to_row()) != request['origin_example_digest']:
            raise ValueError('normalized original request changed')
        shared = [d for d in example.documents if d.document_id == cases[group['case_id']]['target_document_id']]
        if len(shared) != 1:
            raise ValueError('ambiguous shared document')
        rows.append(dict(request_id=request['request_id'], origin_id=oid, group_id=group['group_id'],
            shared_document_supporting=shared[0].supporting,
            shared_segment_contains_supporting_sentence=None,
            label_scope='original_dataset_document_support_not_token_slice'))
    if len(rows) != len(wanted) or len({r['origin_id'] for r in rows}) != len(wanted):
        raise ValueError('missing/duplicate target origin')
    report = dict(kind='source_document_relevance_audit_v1', rows=rows,
        targets=len(rows), supporting_targets=sum(r['shared_document_supporting'] for r in rows),
        pilot_file_sha256=file_sha(args.pilot), input_sha256=pilot['input_sha256'],
        answers_or_model_outcomes_used=False, paper_evidence=False, locked_test_accessed=False)
    report['audit_sha256'] = digest_json(report)
    atomic_write_json(output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','input_sha256')}))


if __name__ == '__main__':
    main()
