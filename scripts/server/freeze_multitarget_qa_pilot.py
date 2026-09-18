"""Freeze full-context QA pilot inputs on CPU, before any answer comparison."""
import argparse
import json
from pathlib import Path

from audit_multitarget_qa_geometry import file_sha
from probekv.io import atomic_write_json
from probekv.rag_data import iter_raw_records, normalize_example
from probekv.source_quality_partition import freeze_group_roles
from probekv.source_quality_requests import build_quality_request
from probekv.v8_schema10_execution import digest_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cases', 'partition', 'recovery', 'raw', 'geometry', 'tokenizer', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--prompt-protocol', choices=('legacy_context_qa', 'short_answer_v1'),
                        default='legacy_context_qa')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('fresh output required')
    geometry = json.loads(Path(args.geometry).read_text())
    recovery = json.loads(Path(args.recovery).read_text())
    hashes = {k: file_sha(getattr(args, k)) for k in ('cases', 'partition', 'recovery', 'raw')}
    if hashes != geometry['input_sha256'] or geometry.get('locked_test_accessed') is not False:
        raise ValueError('geometry inputs or partition provenance mismatch')
    for k in ('cases', 'partition', 'raw'):
        if hashes[k] != recovery['input_sha256'][k]:
            raise ValueError('recovery input mismatch')
    parents = [json.loads(x) for x in Path(args.partition).read_text().splitlines() if x.strip()]
    if any(p.get('partition_role') != 'development_profile_freeze' or
           p.get('locked_test_accessed') is not False for p in parents):
        raise ValueError('only frozen development parents permitted')
    allowed = {p['case_id']: p['group_id'] for p in parents}
    eligible = {g['case_id'] for g in geometry['groups'] if not g['failures'] and g['all_fit_4096']}
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    selected, descriptors = [], []
    for case in cases:
        if case['case_id'] not in eligible:
            continue
        if allowed.get(case['case_id']) != case['group_id'] or case['dataset'] != recovery['dataset']:
            raise ValueError('case outside audited development group')
        sources = [s['origin_example_id'] for s in case['sources'][:4]]
        targets = sorted(t['origin_example_id'] for t in recovery['groups'][case['case_id']])
        descriptors.append(dict(group_id=case['group_id'], content_key=case['reuse_content_key'],
                                source_origin_ids=sources, target_origin_ids=targets))
        selected.append(case)
    roles = freeze_group_roles(descriptors)
    required = {o for d in descriptors for o in d['source_origin_ids'] + d['target_origin_ids']}
    examples = {}
    for raw in iter_raw_records(Path(args.raw)):
        oid = str(raw.get('id', raw.get('_id', '')))
        if oid in required:
            if oid in examples:
                raise ValueError('duplicate raw origin')
            examples[oid] = normalize_example(recovery['dataset'], raw)
    if set(examples) != required:
        raise ValueError('missing original examples')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=False)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    assets = {p.name: file_sha(p) for p in Path(args.tokenizer).iterdir()
              if p.is_file() and (p.name.startswith('tokenizer') or p.name in
                                 ('special_tokens_map.json', 'config.json'))}
    if not assets:
        raise ValueError('tokenizer identity unavailable')
    groups = []
    for case, descriptor in zip(selected, descriptors):
        requests = []
        for epoch, oid in enumerate(descriptor['source_origin_ids'] + descriptor['target_origin_ids'], 1):
            requests.append(build_quality_request(examples[oid], case, encode,
                request_id=digest_json([case['group_id'], oid]), request_epoch=epoch,
                partition_role=roles[case['group_id']], partition_digest=hashes['partition'],
                max_model_len=4096, prompt_protocol=args.prompt_protocol))
        prefixes = {digest_json(q['token_ids'][:q['segments'][0]['positions'][0]]) for q in requests[:4]}
        if len(prefixes) != 4:
            raise ValueError('four distinct historical prefix states required')
        groups.append(dict(case_id=case['case_id'], group_id=case['group_id'],
                           role=roles[case['group_id']], source_requests=requests[:4],
                           target_requests=requests[4:]))
    report = dict(kind='multitarget_qa_pilot_partition_v1', groups=groups,
                  input_sha256=hashes, geometry_sha256=file_sha(args.geometry),
                  tokenizer_asset_sha256=assets, split_seed=20260726,
                  chronology='corpus_derived_pseudotime_not_production_chronology',
                  selection_outcomes_used_for_split=False, repair_ratio=0.15,
                  answer_boundary_contract='qa_next_question_boundary_v1',
                  prompt_protocol=args.prompt_protocol,
                  exploratory_revision_after_answer_format_diagnostic=True,
                  repair_metric='normalized_kv_deviation', common_first_reuse_layer=9,
                  dataset_scope=recovery['dataset'], mechanism_pilot_only=True,
                  online_execution_allowed=False, gpu_runtime_qualified=False,
                  paper_evidence=False, locked_test_accessed=False)
    report['partition_sha256'] = digest_json(report)
    atomic_write_json(output, report)
    print(json.dumps({'groups': len(groups), 'roles': roles,
                      'targets': sum(len(g['target_requests']) for g in groups),
                      'partition_sha256': report['partition_sha256']}))


if __name__ == '__main__':
    main()
