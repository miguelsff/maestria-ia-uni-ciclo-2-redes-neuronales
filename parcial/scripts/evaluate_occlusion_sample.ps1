param(
    [string]$ParentOutput = 'E:/Kaggle/exam-output-wildlife-v6',
    [string]$SampleOutput = 'E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskAction = if ($CheckOnly) { 'check' } else { 'run' }
if (-not (Test-Path -LiteralPath (Join-Path $SampleOutput 'protocol.json'))) {
    throw 'Prepare the separate 600-case review before evaluation.'
}
& docker run --rm --network none --cpus 2 --memory 4g --entrypoint python `
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src `
    --mount "type=bind,source=$taskProject,target=/workspace,readonly" `
    --mount "type=bind,source=$ParentOutput,target=/parent,readonly" `
    --mount "type=bind,source=$SampleOutput,target=/sample" `
    neural-exam-wildlife:0.2 /workspace/scripts/evaluate_occlusion_sample.py $taskAction `
    --parent /parent --output /sample
if ($LASTEXITCODE -ne 0) { throw 'Sample evaluation did not complete; inspect the diagnostics above.' }
