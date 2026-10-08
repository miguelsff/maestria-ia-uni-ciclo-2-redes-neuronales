param([string]$EnvironmentFile = '')

$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskEnvironment = if ($EnvironmentFile) { [IO.Path]::GetFullPath($EnvironmentFile) } else {
    Join-Path $taskProject '.env.wildlife-v6'
}
$taskValues = @{}
foreach ($taskLine in Get-Content -LiteralPath $taskEnvironment) {
    if ($taskLine -match '^([A-Z_]+)=(.*)$') { $taskValues[$Matches[1]] = $Matches[2].Trim() }
}
$taskOutput = $taskValues['OUTPUT_HOST_DIR']
if (-not $taskOutput -or -not (Test-Path -LiteralPath (Join-Path $taskOutput 'wildlife/yolov8s/training_complete.json'))) {
    throw 'Recovery requires the preserved completed scientific training.'
}
$taskReceipts = Join-Path $taskOutput 'receipts'
$taskAttempt = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
$taskContainer = "neural-exam-wildlife-v6-recovery-$taskAttempt"
$taskCompose = @('compose', '--env-file', $taskEnvironment)
foreach ($taskLayer in @('compose.wildlife.yaml', 'compose.wildlife.v2.yaml', 'compose.wildlife.v3.yaml',
        'compose.wildlife.v4.yaml', 'compose.wildlife.v5.yaml', 'compose.wildlife.v6.yaml')) {
    $taskCompose += @('-f', $taskLayer)
}

Push-Location $taskProject
try {
    $taskOther = @(& docker ps --filter 'label=com.docker.compose.service=runner' --format '{{.Names}}')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect active experiment containers.' }
    if ($taskOther.Count -gt 0) { throw 'Another experiment is active; preserve it and retry later.' }
    $taskLog = Join-Path $taskReceipts "recovery-evaluation-$taskAttempt.log"
    & docker @taskCompose run --no-deps --name $taskContainer --entrypoint python runner `
        scripts/recover_wildlife_evaluation.py --finish *> $taskLog
    $taskExitCode = $LASTEXITCODE
    $taskStateText = & docker inspect --format '{{json .State}}' $taskContainer
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the retained recovery container.' }
    $taskReceipt = [pscustomobject]@{
        container = $taskContainer
        exit_code = $taskExitCode
        state = ($taskStateText | ConvertFrom-Json)
        log = $taskLog
        training_repeated = $false
        frozen_sources_changed = $false
        bootstrap_protocol_changed = $false
    }
    [IO.File]::WriteAllText((Join-Path $taskReceipts "recovery-run-state-$taskAttempt.json"),
        ($taskReceipt | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
    if ($taskExitCode -ne 0) { throw "Recovery failed; inspect $taskLog and its retained container state." }
    & docker image inspect neural-exam-wildlife-v6-reports *> $null
    if ($LASTEXITCODE -ne 0) {
        & docker @taskCompose build reports *> (Join-Path $taskReceipts "recovery-reports-build-$taskAttempt.log")
        if ($LASTEXITCODE -ne 0) { throw 'Cannot build the draft report compiler.' }
    }
    & docker @taskCompose run --rm --no-deps reports --task wildlife --draft *> `
        (Join-Path $taskReceipts "recovery-report-$taskAttempt.log")
    if ($LASTEXITCODE -ne 0) { throw 'Draft compilation failed; inspect the recovery report log.' }
    Write-Host 'Recovered evaluation, notebook and draft PDF finished. Human review remains explicit.'
} finally {
    Pop-Location
}
