"""Independently audit the selected environment, then optionally run one sentinel.

No global editable-install changes, tree-only attestations or old audit input.
Run from a clean exact ProbeKV checkout. All output is a fresh directory.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


EXTRAS = (
    '0013-probekv-deferred-layer-timing.patch',
    '0014-probekv-request-position-workspace.patch',
    '0015-probekv-owned-position-validation.patch',
    '0016-probekv-matched-boundary-source-kv.patch',
)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require_origin(origin, root):
    actual, expected = Path(origin).resolve(), Path(root).resolve()
    if expected not in actual.parents:
        raise RuntimeError('unexpected imported module path: ' + str(actual))
    return str(actual)


def verify_assets(audit):
    if not audit.get('complete') or not audit.get('files') or not audit.get('tokenizer_assets_sha256'):
        raise ValueError('complete model asset audit required')
    root = Path(audit['snapshot_path'])
    for name, expected in audit['files'].items():
        # HF files can legitimately be symlinks to the cache blob store.
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid model audit relative path')
        if file_sha(root / relative) != expected:
            raise ValueError('model asset digest mismatch: ' + name)


def matched_executor_args(repeats):
    """Keep the fixed-resident diagnostic's required flags together."""
    if not 1 <= repeats <= 20:
        raise ValueError('backend repeats must be in 1..20')
    return ['--matched-executor-only', '--matched-repair-backends',
            '--cacheblend-loop-control', '--gpu-hot-cache', '--cost-probe',
            '--backend-repeats', str(repeats)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cacheblend', required=True)
    p.add_argument('--model-audit', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--segment-tokens', type=int, choices=(128, 512, 640), default=512)
    p.add_argument('--defer-layer-timing', action='store_true')
    p.add_argument('--host-position-validation', action='store_true')
    p.add_argument('--native-dense-continuation', action='store_true')
    p.add_argument('--contiguous-source-rows', action='store_true')
    p.add_argument('--kv-layout-mode', choices=('legacy', 'packed_slice'), default='legacy')
    p.add_argument('--prefix-shadow-transfer', choices=('layerwise', 'batched'), default='layerwise')
    p.add_argument('--position-validation-ab-repeats', type=int, default=0, choices=range(21))
    p.add_argument('--matched-executor-only', action='store_true')
    p.add_argument('--backend-repeats', type=int, default=20, choices=range(1, 21))
    p.add_argument('--execute', action='store_true', help='run correctness + cost-probe after verification')
    args = p.parse_args()
    if args.position_validation_ab_repeats and not args.host_position_validation:
        p.error('position-validation A/B requires --host-position-validation')
    if args.matched_executor_only and args.position_validation_ab_repeats:
        p.error('executor-only and continuation A/B require separate runs')
    repo = Path(__file__).resolve().parents[2]
    cb, output = Path(args.cacheblend).resolve(), Path(args.output).resolve()
    config, audit_path = Path(args.config).resolve(), Path(args.model_audit).resolve()
    if output.exists():
        raise ValueError('fresh environment/experiment directory required')
    output.mkdir(parents=True)
    sys.path[:0] = [str(repo / 'src'), str(cb / 'vllm_blend')]
    from probekv.io import atomic_write_json
    from probekv.cacheblend_patch import validate_native_patch_audit
    started = time.perf_counter()
    def command(*argv, cwd=repo):
        return subprocess.check_output(argv, cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    try:
        sha = command('git', 'rev-parse', 'HEAD')
        if command('git', 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('clean tracked ProbeKV checkout required')
        busy = command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits')
        if busy:
            raise RuntimeError('GPU has active compute processes; do not interrupt them: ' + busy)
        gpu = command('nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv')
        patch_audit = output / 'patch_audit.json'
        argv = [sys.executable, str(repo / 'scripts/server/verify_cacheblend_patch.py'),
                '--cacheblend', str(cb), '--mode', 'probekv_v8_variant_growth_counterfactual',
                '--manifest', str(repo / 'patches/cacheblend/manifest.json'), '--output', str(patch_audit)]
        for patch in EXTRAS:
            argv.extend(['--extra-patch', patch])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo / 'src'), str(cb / 'vllm_blend')]))
        with (output / 'independent_patch_rebuild.log').open('w') as log:
            subprocess.run(argv, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        patch = json.loads(patch_audit.read_text())
        validate_native_patch_audit(patch, repo / 'patches/cacheblend/manifest.json',
                                    deferred_timing=args.defer_layer_timing)
        # The existing consumer inspects the real index as well as the worktree.
        # Do not silently stage or alter an old checkout on its behalf.
        if command('git', 'write-tree', cwd=cb) != patch['cacheblend_tree']:
            raise RuntimeError('independent worktree passed, but actual CacheBlend index is stale')
        command('git', 'diff', '--quiet', cwd=cb)
        audit = json.loads(audit_path.read_text())
        verify_assets(audit)
        origins = {}
        for name, root in [('probekv', repo / 'src'), ('vllm', cb / 'vllm_blend')]:
            origins[name] = require_origin(importlib.util.find_spec(name).origin, root)
        for name in ('torch', 'transformers', 'vllm._C'):
            module = importlib.import_module(name)
            origins[name] = str(Path(module.__file__).resolve())
        require_origin(origins['vllm._C'], cb / 'vllm_blend')
        extensions = {name: {'path': path, 'sha256': file_sha(path)}
                      for name, path in origins.items() if name == 'vllm._C'}
        launch = [str(repo / 'scripts/server/run_schema10_native_correctness.py'),
                  '--model-audit', str(audit_path), '--patch-audit', str(patch_audit),
                  '--config', str(config), '--output', str(output / 'native'),
                  '--segment-tokens', str(args.segment_tokens), '--execute', '--cost-probe', '--skip-eager-cfo']
        launch.extend(['--kv-layout-mode', args.kv_layout_mode])
        launch.extend(['--prefix-shadow-transfer', args.prefix_shadow_transfer])
        for key in ('defer_layer_timing', 'host_position_validation', 'native_dense_continuation',
                    'contiguous_source_rows'):
            if getattr(args, key):
                launch.append('--' + key.replace('_', '-'))
        if args.position_validation_ab_repeats:
            launch.extend(['--position-validation-ab-repeats', str(args.position_validation_ab_repeats)])
        if args.matched_executor_only:
            launch.extend(matched_executor_args(args.backend_repeats))
        lock = dict(code_sha=sha, patch_audit_sha256=file_sha(patch_audit),
                    cacheblend_tree=patch['cacheblend_tree'], model_audit_sha256=file_sha(audit_path),
                    model_id=audit['model_id'], model_revision=audit['revision'],
                    tokenizer_sha256=audit['tokenizer_assets_sha256'], config_sha256=file_sha(config),
                    imports=origins, extensions=extensions, gpu=gpu, python=sys.executable,
                    versions={n: getattr(sys.modules[n], '__version__', None) for n in ('torch', 'transformers')},
                    setup_wall_seconds=time.perf_counter() - started, launch_argv=launch,
                    invalid_prior_audits=['patch-audit-418b0628-schema10.json'],
                    environment_verified=True, gpu_runtime_qualified=False, paper_evidence=False,
                    locked_test_accessed=False)
        atomic_write_json(output / 'environment_lock.json', lock)
        print(json.dumps({'environment_verified': True, 'code_sha': sha, 'execute': args.execute}), flush=True)
        if args.execute:
            # Same interpreter and explicitly pinned paths; no global installation change.
            sys.argv = launch
            runpy.run_path(launch[0], run_name='__main__')
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        atomic_write_json(output / 'failed.json', {'error': str(error), 'type': type(error).__name__,
                          'paper_evidence': False, 'gpu_runtime_qualified': False})
        raise


if __name__ == '__main__':
    main()
