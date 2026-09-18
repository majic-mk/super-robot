"""Read-only post-outcome dominance audit; never select a deployable policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.source_policy_replay import validate_observation
from probekv.v8_schema10_execution import digest_json


def summarize_group(rows, tolerance=.02):
    if not rows:
        raise ValueError('empty group')
    sources = set(rows[0]['f1_by_source'])
    if not sources or any(set(r['f1_by_source']) != sources for r in rows):
        raise ValueError('fixed cohort required')
    safe_sets = [{s for s in sources if r['f1_by_source'][s] >= r['dense_f1']-tolerance-1e-12} for r in rows]
    best_sets = [{s for s in sources if r['f1_by_source'][s] >= max(r['f1_by_source'].values())-1e-12} for r in rows]
    return dict(target_count=len(rows),
        fixed_sources_safe_on_every_target=sorted(set.intersection(*safe_sets)),
        fixed_sources_max_f1_on_every_target=sorted(set.intersection(*best_sets)),
        oracle_safe_coverage=sum(bool(s) for s in safe_sets)/len(rows),
        fixed_safe_coverage={s: sum(s in x for x in safe_sets)/len(rows) for s in sorted(sources)},
        oracle_mean_f1=sum(max(r['f1_by_source'].values()) for r in rows)/len(rows),
        fixed_mean_f1={s: sum(r['f1_by_source'][s] for r in rows)/len(rows) for s in sorted(sources)},
        post_hoc_fixed_source_not_deployable=True, targets=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError('fresh audit output required')
    if not json.loads((root/'completed.json').read_text()).get('diagnostic_matrix_complete'):
        raise ValueError('complete matrix required')
    hashes, groups = {}, {}
    for folder in sorted(root.glob('group-*')):
        rows = []
        source_order = [json.loads(p.read_text())['source_id'] for p in sorted(folder.glob('source-*.json'))]
        for path in sorted(folder.glob('target-*-qa.json')):
            select = path.with_name(path.name.replace('-qa.json', '-selection.json'))
            for f in (path, select):
                hashes[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
            qa = json.loads(path.read_text())
            observation = json.loads(select.read_text())['observation']
            validate_observation(observation)
            raw = qa['raw_observations']
            for r in raw:
                if digest_json({k:v for k,v in r.items() if k != 'raw_observation_sha256'}) != r['raw_observation_sha256']:
                    raise ValueError('raw QA digest mismatch')
            dense = [r for r in raw if r['source_id'] is None]
            sources = [r for r in raw if r['source_id'] is not None]
            if len(dense) != 1 or len({r['source_id'] for r in sources}) != len(sources):
                raise ValueError('invalid action inventory')
            scores = {r['source_id']: r['qa_evidence']['answer_f1'] for r in sources}
            if any(not isinstance(v, (int,float)) or not math.isfinite(v) for v in scores.values()):
                raise ValueError('missing finite F1')
            ranks = []
            for depth in observation['depth_observations']:
                residual = {}
                for s, state in depth['sources'].items():
                    v = sorted(state['normalized_k_drifts'])
                    count = len(v)-min(len(v)-1, math.ceil(.15*len(v)))
                    residual[s] = math.fsum(v[:count])/count
                if set(residual) != set(scores):
                    raise ValueError('QA/selection cohort mismatch')
                order = sorted(residual, key=lambda s:(residual[s],s))
                ranks.append(dict(depth=depth['completed_depth'], source_order=order,
                    residual_by_source=residual, f1_in_rank_order=[scores[s] for s in order],
                    margin=(residual[order[1]]-residual[order[0]])/max(residual[order[1]],1e-12)))
            rows.append(dict(target=path.stem, dense_f1=dense[0]['qa_evidence']['answer_f1'],
                f1_by_source=scores, rankings=ranks,
                token_ids_by_source={r['source_id']:r['qa_evidence']['token_ids'] for r in sources},
                source_answers={r['source_id']:r['answer'] for r in sources}))
        groups[folder.name] = summarize_group(rows)
        if source_order:
            groups[folder.name]['outcome_independent_fixed_policies'] = {
                policy: dict(source_id=s, mean_f1=groups[folder.name]['fixed_mean_f1'][s],
                    safe_coverage=groups[folder.name]['fixed_safe_coverage'][s])
                for policy, s in [('earliest', source_order[0]), ('latest', source_order[-1])]}
        groups[folder.name]['per_depth'] = {
            str(d): dict(mean_selected_f1=sum(r['f1_by_source'][next(x for x in r['rankings'] if x['depth']==d)['source_order'][0]] for r in rows)/len(rows),
                max_f1_tie_hits=sum(r['f1_by_source'][next(x for x in r['rankings'] if x['depth']==d)['source_order'][0]] >= max(r['f1_by_source'].values())-1e-12 for r in rows),
                safe_hits=sum(r['f1_by_source'][next(x for x in r['rankings'] if x['depth']==d)['source_order'][0]] >= r['dense_f1']-.02-1e-12 for r in rows))
            for d in [x['depth'] for x in rows[0]['rankings']]}
    report = dict(kind='post_hoc_source_dominance_audit_v1', groups=groups, input_sha256=hashes,
        fixed_repair_ratio=.15, residual_trim_ratio=.15, quality_drop_tolerance=.02,
        threshold_fitted=False, production_admission_applicable=False, paper_evidence=False)
    report['audit_sha256'] = digest_json(report)
    atomic_write_json(output, report)
    print(json.dumps({g:{k:v for k,v in r.items() if k not in ('targets',)} for g,r in groups.items()}))


if __name__ == '__main__':
    main()
