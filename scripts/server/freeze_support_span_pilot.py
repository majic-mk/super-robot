"""Freeze a no-fit supporting-span diagnostic; preserve the original partition."""
import argparse
import json
from pathlib import Path

from census_source_support_extension import unused_calibration
from probekv.io import atomic_write_json, sha256_file
from probekv.manifest import manifest_case_from_row
from probekv.rag_data import iter_raw_records, normalize_example, segment_text
from probekv.source_quality_partition import validation_only_roles
from probekv.source_quality_requests import build_quality_request
from probekv.source_support_span import classify_support
from probekv.v8_profile_manifest import canonicalize_v8_profile_cases
from probekv.v8_schema10_execution import digest_json
from run_multitarget_qa_pilot import validate_pilot


def select_census_groups(census):
    # Pre-outcome choice: all qualifying groups, one deterministic slice/group.
    selected = {}
    for row in sorted(census['rows'], key=lambda r:r['case_id']):
        good = [t for t in row['targets'] if t['support_stratum']=='full_support_sentence']
        if len(good) >= 5:
            selected.setdefault(row['group_id'], dict(row, targets=good))
    if not selected:
        raise ValueError('no group with five full-support targets')
    return list(selected.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('census','cases','partition','raw','tokenizer','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--prior-pilot', action='append', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('new output directory required')
    census = json.loads(Path(a.census).read_text())
    hashes = {k:sha256_file(Path(getattr(a,k))) for k in ('cases','partition','raw')}
    if hashes != census['input_sha256'] or census.get('locked_test_accessed') is not False:
        raise ValueError('census input provenance mismatch')
    excluded_origins, excluded_groups, excluded_tokens = set(),set(),set()
    for path in a.prior_pilot:
        prior = json.loads(Path(path).read_text())
        validate_pilot(prior)
        for g in prior['groups']:
            excluded_groups.add(g['group_id'])
            for q in g['source_requests']+g['target_requests']:
                excluded_origins.add(q['origin_example_id'])
                excluded_tokens.add(digest_json(q['segments'][0]['token_ids']))
    chosen = select_census_groups(census)
    read = lambda p:[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    parents = unused_calibration(read(a.cases),read(a.partition),census['dataset'])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,use_fast=False)
    fast = AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,use_fast=True)
    if sha256_file(Path(a.tokenizer)/'tokenizer.json') != census['tokenizer_json_sha256']:
        raise ValueError('tokenizer changed since census')
    encode = lambda s:tok.encode(s,add_special_tokens=False)
    canonical = canonicalize_v8_profile_cases([manifest_case_from_row(r) for r in parents],
        tokenizer_signature=census['tokenizer_json_sha256'],document_revision=hashes['raw'],
        decode=lambda ids:tok.decode(list(ids)))
    cases = {c.case_id:c.to_row() for c in canonical}
    descriptors = []
    for row in chosen:
        case=cases[row['case_id']]
        ids=[s['origin_example_id'] for s in case['sources']]
        targets=[t['origin_example_id'] for t in sorted(row['targets'],key=lambda t:(t['pseudo_time_rank'],t['origin_example_id']))]
        if (set(ids+targets)&excluded_origins or case['group_id'] in excluded_groups or
                digest_json(case['segment_token_ids']) in excluded_tokens):
            raise ValueError('new cohort overlaps previously observed pilot')
        descriptors.append(dict(group_id=case['group_id'],content_key=case['reuse_content_key'],
            source_origin_ids=ids,target_origin_ids=targets))
    roles=validation_only_roles(descriptors)
    wanted={i for d in descriptors for i in d['source_origin_ids']+d['target_origin_ids']}
    originals={}
    for raw in iter_raw_records(Path(a.raw)):
        oid=str(raw.get('id',raw.get('_id','')))
        if oid in wanted:
            if oid in originals:
                raise ValueError('duplicate raw origin')
            originals[oid]=raw
    if set(originals)!=wanted:
        raise ValueError('missing original requests')
    groups,failures,geometry=[],[],[]
    for row,d in zip(chosen,descriptors):
        case=cases[row['case_id']]; requests=[]
        for epoch,oid in enumerate(d['source_origin_ids']+d['target_origin_ids'],1):
            raw=originals[oid]; example=normalize_example(census['dataset'],raw)
            try:
                q=build_quality_request(example,case,encode,request_id=digest_json([case['group_id'],oid]),
                    request_epoch=epoch,partition_role=roles[case['group_id']],
                    partition_digest=sha256_file(Path(a.census)),max_model_len=4096,prompt_protocol='short_answer_v1')
                if epoch>4:
                    doc=next(x for x in example.documents if x.document_id==case['target_document_id'])
                    left=case['canonical_parent_left_token_ids']
                    label=classify_support(raw,doc,
                        parent_token_ids=left+case['segment_token_ids']+case['canonical_parent_right_token_ids'],
                        encoded=fast(segment_text(doc),add_special_tokens=False,return_offsets_mapping=True),
                        token_start=len(left),token_end=len(left)+len(case['segment_token_ids']))
                    if label!='full_support_sentence':
                        raise ValueError('target support label changed')
                requests.append(q)
                geometry.append(dict(origin_id=oid,prompt_tokens=len(q['token_ids']),role='source' if epoch<=4 else 'target'))
            except ValueError as exc:
                failures.append(dict(origin_id=oid,reason=str(exc)))
        if len(requests)!=len(d['source_origin_ids']+d['target_origin_ids']):
            continue
        prefixes={digest_json(q['token_ids'][:q['segments'][0]['positions'][0]]) for q in requests[:4]}
        if len(prefixes)!=4:
            failures.append(dict(group_id=case['group_id'],reason='historical prefixes not distinct'))
        groups.append(dict(case_id=case['case_id'],group_id=case['group_id'],role='validation',
            source_requests=requests[:4],target_requests=requests[4:]))
    out.mkdir(parents=True)
    audit=dict(kind='support_extension_geometry_v1',failures=failures,members=geometry,
        input_sha256=hashes,census_sha256=sha256_file(Path(a.census)),
        prior_pilot_sha256={str(p):sha256_file(Path(p)) for p in a.prior_pilot},
        outcomes_used=False,locked_test_accessed=False)
    atomic_write_json(out/'geometry.json',audit)
    if failures:
        raise ValueError('geometry failed; evidence retained, no pilot frozen')
    assets={f.name:sha256_file(f) for f in Path(a.tokenizer).iterdir() if f.is_file()
        and (f.name.startswith('tokenizer') or f.name in ('config.json','special_tokens_map.json'))}
    report=dict(kind='multitarget_qa_pilot_partition_v1',groups=groups,
        input_sha256=dict(hashes,partition=sha256_file(Path(a.census))),
        original_partition_sha256=hashes['partition'],geometry_sha256=sha256_file(out/'geometry.json'),
        tokenizer_asset_sha256=assets,split_seed=20260726,validation_only_no_profile_fit=True,
        chronology='corpus_derived_pseudotime_not_production_chronology',
        selection_outcomes_used_for_split=False,repair_ratio=.15,repair_metric='normalized_kv_deviation',
        common_first_reuse_layer=9,answer_boundary_contract='qa_next_question_boundary_v1',
        prompt_protocol='short_answer_v1',dataset_scope=census['dataset'],mechanism_pilot_only=True,
        sampling_scope='all_unused_calibration_groups_with_at_least_five_full_support_targets',
        online_execution_allowed=False,gpu_runtime_qualified=False,paper_evidence=False,locked_test_accessed=False)
    report['partition_sha256']=digest_json(report)
    validate_pilot(report)
    atomic_write_json(out/'pilot.json',report)
    print(json.dumps(dict(groups=len(groups),targets=sum(len(g['target_requests']) for g in groups),
        file_sha256=sha256_file(out/'pilot.json'),max_prompt_tokens=max(r['prompt_tokens'] for r in geometry))))


if __name__=='__main__':
    main()
