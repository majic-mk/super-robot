"""Audit recovered multi-target full QA geometry; does not authorize execution."""
import argparse
import hashlib
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.rag_data import iter_raw_records, normalize_example, render_preceding_context, segment_text


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_member(example, case, encode, *, old_prefix=None, generation_tokens=32):
    matches = [(i, d) for i, d in enumerate(example.documents)
               if d.document_id == case['target_document_id']]
    if len(matches) != 1:
        raise ValueError('shared document missing or ambiguous')
    index, document = matches[0]
    parent = list(encode(segment_text(document)))
    left = list(case.get('canonical_parent_left_token_ids', []))
    shared = list(case['segment_token_ids'])
    right = list(case.get('canonical_parent_right_token_ids', []))
    if parent != left + shared + right or not shared:
        raise ValueError('canonical token reconstruction differs')
    if not example.answers:
        raise ValueError('original QA answers missing')
    prefix = render_preceding_context(example.documents[:index])
    suffix = render_preceding_context(example.documents[index + 1:])
    count = len(list(encode(prefix))) + len(parent) + len(list(encode(suffix)))
    count += len(list(encode('\nQuestion: ' + example.question + '\nAnswer:')))
    return dict(origin_id=example.example_id, prompt_tokens=count,
                required_model_len=count + generation_tokens,
                source_recapture_required=(old_prefix != prefix or bool(left)) if old_prefix is not None else None,
                original_document_count=len(example.documents),
                supporting_document_count=sum(d.supporting for d in example.documents),
                full_context_preserved=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'partition', 'recovery', 'raw', 'dataset', 'tokenizer', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('fresh audit output required')
    recovery = json.loads(Path(args.recovery).read_text())
    for key in ('cases', 'partition', 'raw'):
        if file_sha(getattr(args, key)) != recovery['input_sha256'][key]:
            raise ValueError('recovery input digest mismatch: ' + key)
    if recovery['dataset'] != args.dataset:
        raise ValueError('recovery dataset mismatch')
    partitions = [json.loads(x) for x in Path(args.partition).read_text().splitlines() if x.strip()]
    if any(r.get('locked_test_accessed') is not False or r.get('partition_role') != 'development_profile_freeze'
           for r in partitions):
        raise ValueError('frozen development-only parents required')
    allowed = {r['case_id']: r['group_id'] for r in partitions}
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    cases = [c for c in cases if c['dataset'] == args.dataset and c['regime'] == 'corpus-repeat'
             and allowed.get(c['case_id']) == c['group_id']]
    selected = [(c, recovery['groups'].get(c['case_id'], [])) for c in cases]
    selected = [(c, targets) for c, targets in selected if len(targets) >= 5]
    required = {s['origin_example_id'] for c, targets in selected for s in c['sources'][:4]}
    required.update(t['origin_example_id'] for c, targets in selected for t in targets)
    examples = {}
    for raw in iter_raw_records(Path(args.raw)):
        identifier = str(raw.get('id', raw.get('_id', '')))
        if identifier in required:
            if identifier in examples:
                raise ValueError('duplicate original example')
            examples[identifier] = normalize_example(args.dataset, raw)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=False)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    groups = []
    for case, targets in selected:
        rows, failures = [], []
        sources = case['sources'][:4]
        source_ids = {s['origin_example_id'] for s in sources}
        if len(source_ids) != 4 or source_ids & {t['origin_example_id'] for t in targets}:
            failures.append('origins_not_disjoint')
        for member, role in [(s, 'source') for s in sources] + [(t, 'target') for t in targets]:
            identifier = member['origin_example_id']
            try:
                row = audit_member(examples[identifier], case, encode,
                                   old_prefix=member.get('historical_context'))
                if role == 'target' and (member['question'] != examples[identifier].question
                                        or member['answers'] != list(examples[identifier].answers)):
                    raise ValueError('recovered QA differs from original')
                rows.append(dict(row, role=role))
            except (KeyError, ValueError) as exc:
                failures.append(identifier + ': ' + str(exc))
        groups.append(dict(case_id=case['case_id'], group_id=case['group_id'], members=rows,
                           failures=failures, all_fit_4096=not failures and all(r['required_model_len'] <= 4096 for r in rows)))
    report = dict(kind='multitarget_full_qa_geometry_audit_v1', groups=groups,
                  input_sha256={k: file_sha(getattr(args, k)) for k in ('cases', 'partition', 'raw', 'recovery')},
                  tokenizer_path=args.tokenizer, chronology='corpus_derived_pseudotime',
                  fit_validation_partition_frozen=False, online_execution_allowed=False,
                  source_complementarity_verified=False, paper_evidence=False, locked_test_accessed=False)
    atomic_write_json(Path(args.output), report)
    print(json.dumps({'groups': len(groups), 'fit_4096_groups': sum(g['all_fit_4096'] for g in groups)}))


if __name__ == '__main__':
    main()
