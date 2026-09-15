"""Read-only raw-outcome analysis; writes a new report, never promotes GPU gates."""
import argparse
import json
from pathlib import Path
from probekv.request_wallclock import refine_request_wallclock
from probekv.v8_schema10_storage import file_digest
from probekv.io import atomic_write_json


def analyze(path):
    path = Path(path)
    row = json.loads(path.read_text(encoding='utf-8'))
    partition = row['request_wallclock']
    expected = row['first_token_ns'] - row['arrival_ns']
    if (partition['intervals'][0]['start_ns'] != row['arrival_ns']
            or partition['intervals'][-1]['end_ns'] != row['first_token_ns']):
        raise ValueError('outcome absolute endpoints disagree')
    if partition['ttft_ns'] != expected or abs(row['request_ttft_ms'] - expected/1e6) > 1e-9:
        raise ValueError('outcome timing endpoints disagree')
    events = []
    for event in row.get('runtime_events', []):
        if event['kind'] != 'online_timing_landmarks':
            continue
        detailed = event.get('selection_interval_events')
        if detailed is not None:
            events.extend(detailed)
        else:
            events.extend(dict(event_id='historical_unlabeled_selection_%d' % i, start_ns=a, end_ns=b)
                          for i, (a, b) in enumerate(event.get('selection_intervals') or []))
        for sid, span in event.get('preparation_intervals', {}).items():
            events.append(dict(event_id='winner_preparation:' + sid, start_ns=span['start_ns'], end_ns=span['end_ns']))
    return dict(input_path=str(path.resolve()), input_sha256=file_digest(path),
                code_commit=row.get('code_commit'), evidence_origin=row.get('evidence_origin'),
                producer_authentication_verified=False,
                selected_sources=row.get('selected_source_variant_ids'),
                committed_sources=row.get('committed_source_variant_ids'),
                dense_reference_ms=row.get('matched_dense_ttft_ms'),
                timing=refine_request_wallclock(partition, events), paper_evidence=False,
                performance_improvement_proven=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('fresh timing report required')
    atomic_write_json(output, analyze(args.input))
