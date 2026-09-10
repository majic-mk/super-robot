"""Emit an offline policy-spec manifest; never load a model or admit GPU work."""
import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.source_policy_development import development_experiment_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('retain previous evidence; choose a new output path')
    value = development_experiment_spec()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)
    print(json.dumps({'output': str(output), 'spec_sha256': value['spec_sha256'],
                      'gpu_execution_allowed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
