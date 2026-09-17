"""Sequential fixed15 K/V Source QA matrix; never production admission evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_source_policy_capture import read_verified, validate_plan
from probekv.io import atomic_write_json
from probekv.v8_schema10_execution import digest_json


def validate_pilot(pilot):
    if (pilot.get('kind') != 'multitarget_qa_pilot_partition_v1' or
        digest_json({k: v for k, v in pilot.items() if k != 'partition_sha256'}) != pilot.get('partition_sha256') or
        pilot.get('repair_metric') != 'normalized_kv_deviation' or pilot.get('repair_ratio') != .15 or
        pilot.get('common_first_reuse_layer') != 9 or pilot.get('locked_test_accessed') is not False):
        raise ValueError('invalid frozen QA pilot')
    entries = [dict(request_id=q['request_id'], request_sha256=digest_json(q),
                    role=q['partition_role'], content_group=q['content_group'])
               for g in pilot['groups'] for q in g['source_requests'] + g['target_requests']]
    partition = dict(kind='source_policy_capture_partition_v1', entries=entries,
                     parent_development_partition_sha256=pilot['input_sha256']['partition'],
                     locked_test_accessed=False)
    for group in pilot['groups']:
        if pilot.get('answer_boundary_contract') != 'qa_next_question_boundary_v1' or any(
            q.get('answer_boundary_contract') != pilot['answer_boundary_contract']
            for q in group['source_requests'] + group['target_requests']):
            raise ValueError('uniform preregistered answer boundary required')
        if len(group['source_requests']) != 4 or len(group['target_requests']) < 5:
            raise ValueError('incomplete group')
        for target in group['target_requests']:
            plan = dict(kind='source_policy_native_capture_plan_v1',
                        selection_path='legacy_multicheckpoint', host_budget_bytes=2**30,
                        global_byte_budget=8 * 2**30, source_requests=group['source_requests'],
                        target_request=target, paper_evidence=False, locked_test_accessed=False)
            plan['plan_sha256'] = digest_json(plan)
            validate_plan(plan, partition)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pilot', 'pilot-sha256', 'runtime', 'runtime-sha256', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    pilot = read_verified(args.pilot, args.pilot_sha256)
    runtime = read_verified(args.runtime, args.runtime_sha256)
    validate_pilot(pilot)
    from probekv.v8_schema10_native_factory import validate_native_attachment
    native = validate_native_attachment(runtime, allow_unmeasured=True)
    if native.get('repair_metric') != 'normalized_kv_deviation':
        raise ValueError('joint K/V repair runtime required')
    if not args.execute:
        print(json.dumps({'valid': True, 'gpu_started': False}))
        return
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if sha != runtime['binding']['code_commit'] or subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
        raise ValueError('clean matching execution SHA required')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / 'inputs.json', dict(pilot=pilot, runtime=runtime))
    started = time.perf_counter()
    try:
        from probekv.v8_schema10_native_factory import create_native_measurement_backend
        from probekv.v8_schema10_native_oracle import run_native_source_oracle
        from probekv.source_policy_native_capture import capture_native_source_observation
        from probekv.v7_contracts import SourceVariantIdentity
        backend = create_native_measurement_backend(runtime)
        path = 'legacy_multicheckpoint'
        adapter = backend.adapters[path]
        for gi, group in enumerate(pilot['groups']):
            folder = root / ('group-%02d' % gi)
            folder.mkdir()
            backend.reset(capacity=16, global_byte_budget=8 * 2**30)
            source_ids = []
            for si, q in enumerate(group['source_requests']):
                s = q['segments'][0]
                capture = adapter.build_exact_dense_source(q, s['segment_id'])
                identity = SourceVariantIdentity(s['content_key'], digest_json(q['token_ids'][:s['positions'][0]]),
                    digest_json(s['positions']), q['request_id'] + ':' + s['segment_id'], backend.provenance['model_signature'])
                source = backend.store.publish_exact_dense(identity, layers=capture['layers'],
                    selection_states=capture['selection_states'], metadata=capture['source_metadata'],
                    request_epoch=q['request_epoch'], whole_request_origin='exact_dense_full_prefill',
                    materialization_reason='content_miss')
                source_ids.append(source.source_variant_id)
                atomic_write_json(folder / ('source-%02d.json' % si), dict(source_id=source.source_variant_id,
                    request_sha256=digest_json(q), capture_audit=capture['capture_audit']))
                del capture
            for ti, request in enumerate(group['target_requests']):
                adapter.reset()
                observation = capture_native_source_observation(backend, request=request,
                    selection_path=path, source_ids=source_ids, binding=runtime['binding'], host_budget_bytes=2**30)
                atomic_write_json(folder / ('target-%02d-selection.json' % ti), observation)
                adapter.reset()
                report = run_native_source_oracle(backend, request=request,
                    dispatch={'selection_path': path}, segment_id='C', source_ids=source_ids,
                    first_reuse_layer=9, repair_ratio=.15, chosen_source_id=None, max_answer_f1_drop=.02)
                atomic_write_json(folder / ('target-%02d-qa.json' % ti), report)
                print(json.dumps({'group': gi, 'target': ti, 'qa_matrix_completed': True}), flush=True)
        atomic_write_json(root / 'completed.json', dict(diagnostic_matrix_complete=True,
            wall_seconds=time.perf_counter()-started, production_admission_applicable=False,
            deep_selection_oracle_measured=False, paper_evidence=False, locked_test_accessed=False))
    except Exception as exc:
        atomic_write_json(root / 'failed.json', dict(type=type(exc).__name__, error=str(exc),
            wall_seconds=time.perf_counter()-started, paper_evidence=False))
        raise


if __name__ == '__main__':
    main()
