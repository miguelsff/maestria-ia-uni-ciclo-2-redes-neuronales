param(
    [switch]$Resume,
    [switch]$PrepareOnly,
    [ValidateSet('v2', 'v3', 'v4', 'v5', 'v6')][string]$Variant = 'v6',
    [string]$EnvironmentFile = ''
)

$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskEnvironment = if ($EnvironmentFile) { [IO.Path]::GetFullPath($EnvironmentFile) } else {
    Join-Path $taskProject ".env.wildlife-$Variant"
}
if (-not (Test-Path -LiteralPath $taskEnvironment)) {
    throw 'Create the selected wildlife environment from its example and set a private Jupyter token first.'
}
$taskValues = @{}
foreach ($taskLine in Get-Content -LiteralPath $taskEnvironment) {
    if ($taskLine -match '^([A-Z_]+)=(.*)$') { $taskValues[$Matches[1]] = $Matches[2].Trim() }
}
$taskOutput = $taskValues['OUTPUT_HOST_DIR']
$taskBackgrounds = $taskValues['CCT_BACKGROUND_HOST_DIR']
if (-not $taskOutput -or -not $taskBackgrounds) { throw 'Output and background directories must be explicit.' }
$taskReceipts = Join-Path $taskOutput 'receipts'
$taskCompose = @('compose', '--env-file', $taskEnvironment, '-f', 'compose.wildlife.yaml', '-f', 'compose.wildlife.v2.yaml')
if ($Variant -in @('v3', 'v4', 'v5', 'v6')) { $taskCompose += @('-f', 'compose.wildlife.v3.yaml') }
if ($Variant -in @('v4', 'v5', 'v6')) { $taskCompose += @('-f', 'compose.wildlife.v4.yaml') }
if ($Variant -in @('v5', 'v6')) { $taskCompose += @('-f', 'compose.wildlife.v5.yaml') }
if ($Variant -eq 'v6') { $taskCompose += @('-f', 'compose.wildlife.v6.yaml') }

# Check before any preparation can overwrite files in a previously frozen output.
$taskFreeze = Join-Path $taskReceipts 'source-freeze.json'
if (Test-Path -LiteralPath $taskFreeze) {
    $taskFrozen = Get-Content -LiteralPath $taskFreeze -Raw | ConvertFrom-Json
    foreach ($taskEntry in $taskFrozen.sha256.PSObject.Properties) {
        $taskSource = Join-Path $taskProject $taskEntry.Name
        if (-not (Test-Path -LiteralPath $taskSource) -or
            (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskEntry.Value) {
            throw 'Experimental sources differ from this frozen output. Preserve it and select a fresh output.'
        }
    }
}

function Invoke-WildlifeDocker {
    param([string[]]$Arguments, [string]$Log)
    & docker @taskCompose @Arguments *> (Join-Path $taskReceipts $Log)
    if ($LASTEXITCODE -ne 0) { throw "Wildlife operation failed; inspect receipts/$Log in the output directory." }
}

function Test-WildlifeResources {
    $taskOther = @(& docker ps --filter 'label=com.docker.compose.service=runner' --format '{{.Names}} {{.Command}}' |
        Where-Object { $_ -match 'run|smoke|train' })
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect active training containers.' }
    $taskGpu = & nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or @($taskGpu).Count -ne 1) { throw 'Expected one available NVIDIA GPU.' }
    $taskGpuValues = $taskGpu.Split(',').Trim()
    if ($taskOther.Count -gt 0 -or [int]$taskGpuValues[0] -ge 20 -or [int]$taskGpuValues[1] -lt 8192) {
        throw 'Another GPU job is active or insufficient GPU memory is available; preserve it and retry later.'
    }
}

Push-Location $taskProject
try {
    New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
    & docker image inspect neural-exam-wildlife:0.2 *> $null
    New-Item -ItemType Directory -Force -Path $taskReceipts | Out-Null
    if ($LASTEXITCODE -ne 0) { Invoke-WildlifeDocker -Arguments @('build', 'runner') -Log 'build.log' }
    if ($Variant -in @('v3', 'v4', 'v5', 'v6') -and -not (Test-Path -LiteralPath (Join-Path $taskReceipts 'bootstrap.json'))) {
        $taskParent = $taskValues['PARENT_OUTPUT_HOST_DIR']
        if (-not $taskParent -or -not (Test-Path -LiteralPath (Join-Path $taskParent 'wildlife/manifest.json'))) {
            throw 'Set PARENT_OUTPUT_HOST_DIR to the preserved preceding output.'
        }
        $taskParentVersion = switch ($Variant) { 'v6' { 'v5' } 'v5' { 'v4' } 'v4' { 'v3' } default { 'v2' } }
        & docker run --rm --network none --cpus 2 --memory 2g --entrypoint python `
            --env PYTHONDONTWRITEBYTECODE=1 --volume "${taskProject}:/workspace:ro" `
            --volume "${taskParent}:/parent:ro" --volume "${taskOutput}:/outputs" `
            neural-exam-wildlife:0.2 /workspace/scripts/bootstrap_wildlife_v3.py `
            --parent /parent --parent-version $taskParentVersion --output /outputs *> (Join-Path $taskReceipts 'bootstrap.log')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot stage the preserved parent inputs; inspect receipts/bootstrap.log.' }
    }
    Invoke-WildlifeDocker -Arguments @('config', '--quiet') -Log 'compose-check.log'
    if (-not (Test-Path -LiteralPath (Join-Path $taskBackgrounds 'cohort.json'))) {
        New-Item -ItemType Directory -Force -Path $taskBackgrounds | Out-Null
        Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', 'backgrounds') -Log 'backgrounds.log'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $taskOutput 'cache/weights/yolov8s.pt')) -or
        -not (Test-Path -LiteralPath (Join-Path $taskOutput 'cache/weights/yolo26n.pt'))) {
        Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', 'downloads', 'download', '--dataset', 'wildlife-weights') -Log 'weights.log'
    }
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', 'runner', '--device', 'cpu', 'prepare', '--task', 'wildlife') -Log 'prepare.log'
    if ($Variant -in @('v3', 'v4', 'v5', 'v6')) {
        Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'python', 'runner', 'scripts/migrate_wildlife_reviews.py') -Log 'review-migration.log'
    }
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', 'runner', '--device', 'cpu', 'review-occlusion', '--full') -Log 'occlusion-gallery.log'
    Invoke-WildlifeDocker -Arguments @('up', '-d', '--no-deps', 'review') -Log 'review-server.log'
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'python', 'runner', 'scripts/audit_wildlife_preflight.py') -Log 'preflight.log'
    if ($Variant -eq 'v6') {
        Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'python', 'runner', 'scripts/verify_wildlife_loader.py') -Log 'yolo-loader-verification.log'
    }
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'python', 'runner', 'scripts/freeze_wildlife_sources.py') -Log 'source-freeze.log'
    if ($PrepareOnly) {
        Write-Host 'Wildlife data and review galleries prepared. Scientific training has not started.'
        return
    }
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'python', 'runner', 'scripts/check_wildlife_gate.py') -Log 'scientific-gate.log'
    Test-WildlifeResources
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', 'runner', 'doctor', '--gpu-check', '--amp-check') -Log 'doctor.log'
    $taskRun = @('run', '--rm', '--no-deps', 'runner', 'run', '--task', 'wildlife')
    if ($Resume) { $taskRun += '--resume' }
    Invoke-WildlifeDocker -Arguments $taskRun -Log 'scientific-run.log'
    Invoke-WildlifeDocker -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'jupyter', 'runner', 'nbconvert', '--execute', '--to', 'notebook', 'notebooks/03_wildlife.ipynb', '--output', '03_wildlife', '--output-dir', '/outputs/notebooks', '--ExecutePreprocessor.timeout=180') -Log 'notebook.log'
    Invoke-WildlifeDocker -Arguments @('build', 'reports') -Log 'reports-build.log'
    $taskReport = @('run', '--rm', '--no-deps', 'reports', '--task', 'wildlife')
    $taskApprovalPath = Join-Path $taskProject 'reports/wildlife/finalized.json'
    $taskFinalized = (Test-Path -LiteralPath $taskApprovalPath) -and
        (Get-Content -LiteralPath $taskApprovalPath -Raw | ConvertFrom-Json).reviewed
    if (-not $taskFinalized) { $taskReport += '--draft' }
    Invoke-WildlifeDocker -Arguments $taskReport -Log 'report.log'
    Write-Host 'Scientific execution, notebook and PDF finished. Any pending occlusion or editorial review remains explicit.'
} finally {
    Pop-Location
}
