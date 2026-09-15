"""Freeze a no-GPU run schedule using implemented entry points."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from probekv.io import atomic_write_json


def build_plan(sha, config_sha, python, cacheblend, model_audit, remote_root):
    if len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('full code SHA required')
    root = remote_root.rstrip('/') + '/recovery-' + sha
    jobs = []
    previous = []
    for length in (512, 128, 640):
        base_id = '%d-baseline' % length
        def native(name, flags, requires, scope):
            job_id = '%d-%s' % (length, name)
            output = root + '/' + job_id
            argv = [python, 'scripts/server/run_locked_single_segment.py',
                    '--cacheblend', cacheblend, '--model-audit', model_audit,
                    '--config', 'configs/a800_server_lock_v8_schema10.json',
                    '--output', output, '--segment-tokens', str(length), '--execute'] + flags
            jobs.append(dict(job_id=job_id, argv=argv, requires=requires,
                             timing_scope=scope, output=output, status='pending_gpu'))
            return job_id
        native('baseline', [], previous, 'correctness_and_matched_prefix_cost')
        native('deferred', ['--defer-layer-timing'], [base_id], 'correctness_and_matched_prefix_cost')
        host = native('owned-host', ['--defer-layer-timing', '--host-position-validation',
                      '--position-validation-ab-repeats', '20'], [str(length)+'-deferred'],
                      'paired_dense_continuation_same_prefix')
        continuation = native('native-continuation', ['--defer-layer-timing',
                              '--host-position-validation', '--native-dense-continuation'],
                              [host], 'correctness_and_matched_prefix_cost')
        native('executor', ['--matched-executor-only', '--backend-repeats', '20'],
               [base_id], 'zero_prefix_fixed_source_common_mask_executor_only')
        for variant in ('baseline', 'deferred', 'owned-host', 'native-continuation'):
            source = '%d-%s' % (length, variant)
            jobs.append(dict(job_id=source+'-online', requires=[source],
                argv=[python, 'scripts/server/run_schema10_native_online_closure.py',
                      '--correctness-root', root+'/'+source+'/native',
                      '--output', root+'/'+source+'-online', '--replays', '22',
                      '--selection-cache-mode', 'off'],
                timing_scope='arrival_to_first_token', status='pending_gpu',
                prerequisite='fresh correctness and all cost cells verified; same explicit PYTHONPATH',
                sample_interpretation='first two warmups; 20 warm requests, not paired cross-process A/B'))
        for cold in range(3):
            native('cold-%d' % cold, [], [base_id], 'independent_process_setup_and_correctness_not_cold_online_TTFT')
        previous = [continuation, '%d-native-continuation-online' % length]
    payload = dict(code_sha=sha, config_sha256=config_sha, jobs=jobs,
        pythonpath_entries=['<exact checkout>/src', cacheblend+'/vllm_blend'],
        pair_schedule=[dict(pair=i, warmup=i<2, order=['A','B'] if i%2==0 else ['B','A']) for i in range(22)],
        pair_contract=['same request, Prefix, Source, mask, boundary and first-token endpoint',
                       'restore same initial state; one changed factor',
                       'cross-process online sweeps do not establish paired speedup',
                       'preinstall/setup/materialization costs reported separately'],
        required_report_fields=['token_identity', 'teacher_logits_32_relative_l2', 'mask_digest',
            'source_destination_integrity', 'resource_cleanup', 'raw_host_intervals',
            'mean_p50_p95', 'paired_difference_95pct_interval'],
        model_audit_sha256=None, environment_lock_sha256=None, cost_measurement_sha256=None,
        prerequisites='bind and verify actual assets/environment on A800 before inference; stop on failure',
        native_continuation_default=False, online_trace_execution_allowed=False,
        formal_profile_bundle_frozen=False, gpu_runtime_qualified=False,
        paper_evidence=False, locked_test_accessed=False)
    payload['plan_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--cacheblend', required=True)
    parser.add_argument('--model-audit', required=True)
    parser.add_argument('--remote-output-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo).strip():
        raise ValueError('commit tracked changes before freezing plan')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    config_sha = hashlib.sha256((repo/'configs/a800_server_lock_v8_schema10.json').read_bytes()).hexdigest()
    if Path(args.output).exists():
        raise FileExistsError('fresh manifest required')
    atomic_write_json(args.output, build_plan(sha, config_sha, args.python, args.cacheblend,
                                            args.model_audit, args.remote_output_root))


if __name__ == '__main__':
    main()
