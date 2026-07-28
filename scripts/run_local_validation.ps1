$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
Push-Location $workspace
try {
    $env:PYTHONPATH = "src"

    python -m compileall -q src scripts tests
    if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }

    python -m probekv validate-config --config configs/local_smoke.json
    if ($LASTEXITCODE -ne 0) { throw "local config validation failed" }

    python -m probekv validate-config --config configs/a100_e0.json
    if ($LASTEXITCODE -ne 0) { throw "formal server config validation failed" }

    python -m probekv validate-config --config configs/local_e1e2.json
    if ($LASTEXITCODE -ne 0) { throw "local E1/E2 config validation failed" }

    python scripts/validate_contract.py `
        --contract configs/experiment_contract.yaml `
        --output artifacts/local_validation/contract.json
    if ($LASTEXITCODE -ne 0) { throw "contract validation failed" }

    python -m probekv simulate `
        --config configs/local_smoke.json `
        --output artifacts/local_validation/simulation
    if ($LASTEXITCODE -ne 0) { throw "simulation failed" }

    python -m probekv local-e1e2 `
        --config configs/local_e1e2.json `
        --output artifacts/local_validation/local_e1e2 `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "local E1/E2 pipeline failed" }

    python scripts/build_e1_jobs.py `
        --manifest artifacts/local_validation/local_e1e2/case_manifest.jsonl `
        --config configs/local_e1e2.json `
        --output artifacts/local_validation/e1/jobs `
        --splits train,calibration
    if ($LASTEXITCODE -ne 0) { throw "E1 job construction failed" }

    0..3 | ForEach-Object {
        python scripts/simulate_e1_shard.py `
            --jobs artifacts/local_validation/e1/jobs/jobs.jsonl `
            --output artifacts/local_validation/e1/shards `
            --shard-index $_ `
            --shard-count 4 `
            --total-layers 32
        if ($LASTEXITCODE -ne 0) { throw "E1 shard $_ failed" }
    }

    python scripts/merge_e1_results.py `
        --jobs artifacts/local_validation/e1/jobs/jobs.jsonl `
        --result-dir artifacts/local_validation/e1/shards `
        --output artifacts/local_validation/e1/merged `
        --total-layers 32 `
        --require-complete
    if ($LASTEXITCODE -ne 0) { throw "E1 merge failed" }

    python scripts/audit_environment.py `
        --output artifacts/local_validation/environment.json
    if ($LASTEXITCODE -ne 0) { throw "environment audit failed" }

    $modelRoot = Join-Path $env:USERPROFILE `
        ".cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots"
    if (Test-Path $modelRoot) {
        $snapshot = Get-ChildItem -LiteralPath $modelRoot -Directory |
            Select-Object -First 1 -ExpandProperty FullName
        if ($snapshot) {
            python scripts/local_model_h0.py `
                --model $snapshot `
                --output artifacts/local_validation/model_h0.json `
                --threads 8
            if ($LASTEXITCODE -ne 0) { throw "local model H0 failed" }

            python scripts/local_reference_probe.py `
                --model $snapshot `
                --output artifacts/local_validation/reference_probe.json `
                --threads 8
            if ($LASTEXITCODE -ne 0) { throw "local reference probe failed" }

            python scripts/prepare_rag_data.py `
                --dataset hotpotqa `
                --input examples/hotpot_fixture.json `
                --output artifacts/local_validation/hotpot_fixture `
                --tokenizer $snapshot `
                --model-signature "TinyLlama-local-fixture" `
                --construction controlled
            if ($LASTEXITCODE -ne 0) { throw "RAG fixture construction failed" }

            python scripts/probe_rag_manifest.py `
                --manifest artifacts/local_validation/hotpot_fixture/cases.jsonl `
                --model $snapshot `
                --output artifacts/local_validation/hotpot_fixture/reference_probe `
                --limit-cases 1 `
                --threads 8
            if ($LASTEXITCODE -ne 0) { throw "RAG manifest probe failed" }
        }
    }

    python setup.py bdist_wheel --dist-dir artifacts/dist
    if ($LASTEXITCODE -ne 0) { throw "wheel build failed" }

    Write-Host "ProbeKV local validation completed."
}
finally {
    Pop-Location
}
