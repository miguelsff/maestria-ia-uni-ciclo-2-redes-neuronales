param([switch]$Resume)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$outputDirectory = if ($env:MRI_OUTPUT_DIR) { $env:MRI_OUTPUT_DIR } else { 'E:/Kaggle/exam-output-mri/stratified-v2' }
$auditDirectory = Join-Path $outputDirectory 'audit'
$composeArguments = @('compose', '-f', 'compose.mri.yaml')
$trainingContainer = 'neural-exam-mri-training'

# Reject a reused scientific output before writing logs or auditing its files.
if (-not $Resume -and (Test-Path -LiteralPath (Join-Path $outputDirectory 'mri'))) {
    $recordedRuns = @(Get-ChildItem -LiteralPath (Join-Path $outputDirectory 'mri') -Directory |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName 'complete.json')) -or
            (Test-Path -LiteralPath (Join-Path $_.FullName 'last.pt'))
        })
    if ($recordedRuns.Count -gt 0) {
        throw 'MRI output already contains training runs. Set MRI_OUTPUT_DIR to a new directory, or use -Resume for the same code/configuration/data.'
    }
}

function Invoke-MRIDocker {
    param([string[]]$Arguments, [string]$Log)
    & docker @composeArguments @Arguments *> $Log
    if ($LASTEXITCODE -ne 0) { throw "MRI operation failed; inspect $Log" }
}

function Wait-MRIResources {
    while ($true) {
        $other = @(& docker ps --filter 'label=com.docker.compose.service=runner' --format '{{.Names}} {{.Command}}' |
            Where-Object { $_ -notmatch '^mri-' -and $_ -notmatch '^neural-exam-mri-' -and $_ -match 'run|smoke|train' })
        $memoryLine = & wsl -d docker-desktop -e sh -c 'cat /proc/meminfo' | Select-String '^MemAvailable:'
        if (-not $memoryLine) { throw 'Cannot inspect available WSL memory.' }
        $availableKiB = [long]([regex]::Match($memoryLine.ToString(), '\d+').Value)
        $gpuLine = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
        if ($other.Count -eq 0 -and $availableKiB -ge 4 * 1024 * 1024 -and [int]$gpuLine -lt 20) { return }
        Write-Host 'Waiting for other GPU work and sufficient WSL memory.'
        Start-Sleep -Seconds 30
    }
}

Push-Location $projectDirectory
try {
    New-Item -ItemType Directory -Force -Path $auditDirectory | Out-Null
    & docker image inspect neural-exam-mri:0.1 *> $null
    if ($LASTEXITCODE -ne 0) { & docker build --target runtime -t neural-exam-mri:0.1 .; if ($LASTEXITCODE -ne 0) { throw 'MRI image build failed.' } }
    & docker image inspect neural-exam-mri-reports:0.1 *> $null
    if ($LASTEXITCODE -ne 0) { & docker build --target reports -t neural-exam-mri-reports:0.1 .; if ($LASTEXITCODE -ne 0) { throw 'MRI reports image build failed.' } }
    & docker volume create neural-exam-mri_dicom_cache | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $auditDirectory 'cache_source_hashes.json'))) {
        Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', 'scripts/select_mri_series.py') -Log (Join-Path $auditDirectory 'selection.log')
        if (& docker ps -a --filter 'name=^/mri-cache-writer$' --format '{{.ID}}') { throw 'Inspect the existing MRI cache writer before creating another.' }
        & docker run -d --name mri-cache-writer --network none --cpus 2 --memory 2g --mount 'type=volume,source=neural-exam-mri_dicom_cache,target=/cache' --entrypoint sleep neural-exam-mri:0.1 infinity | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the MRI cache writer.' }
        $pythonExecutable = if ($env:MRI_PYTHON) { $env:MRI_PYTHON } else { 'python' }
        & $pythonExecutable scripts/cache_mri_data.py *> (Join-Path $auditDirectory 'cache-copy.log')
        if ($LASTEXITCODE -ne 0) { throw 'MRI cache copy failed; preserve files and inspect cache-copy.log.' }
        & docker stop mri-cache-writer | Out-Null
        & docker rm mri-cache-writer | Out-Null
    }
    Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', 'scripts/audit_mri.py') -Log (Join-Path $auditDirectory 'data-integrity.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', 'runner', '--config', 'configs/mri.toml', 'prepare', '--task', 'mri') -Log (Join-Path $auditDirectory 'prepare.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', 'downloads') -Log (Join-Path $auditDirectory 'assets.log')
    Wait-MRIResources
    Invoke-MRIDocker -Arguments @('run', '--rm', 'runner', '--config', 'configs/mri.toml', 'doctor', '--gpu-check') -Log (Join-Path $auditDirectory 'doctor.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', 'scripts/verify_mri_runtime.py') -Log (Join-Path $auditDirectory 'runtime.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', '-m', 'pytest', 'tests/test_mri.py', 'tests/test_metrics.py', 'tests/test_utilities.py', '-q', '-p', 'no:cacheprovider') -Log (Join-Path $auditDirectory 'tests.log')
    $smokeArguments = @('run', '--rm', 'runner', '--config', 'configs/mri.toml', 'smoke', '--task', 'mri')
    if (Test-Path -LiteralPath (Join-Path $outputDirectory 'smoke/mri/scratch_bce_weighted/last.pt')) { $smokeArguments += '--resume' }
    Invoke-MRIDocker -Arguments $smokeArguments -Log (Join-Path $auditDirectory 'smoke.log')
    Copy-Item -LiteralPath 'configs/mri.toml' -Destination (Join-Path $auditDirectory 'mri-frozen.toml')
    & docker image inspect neural-exam-mri:0.1 --format '{{.Id}}' | Set-Content -LiteralPath (Join-Path $auditDirectory 'runtime_image_id.txt')
    $snapshot = Join-Path $auditDirectory 'source_snapshot'
    New-Item -ItemType Directory -Force -Path $snapshot | Out-Null
    Copy-Item -LiteralPath 'src', 'configs', 'scripts', 'tests' -Destination $snapshot -Recurse -Force
    Wait-MRIResources
    $existing = & docker ps -a --filter "name=^/$trainingContainer$" --format '{{.ID}}'
    if ($existing) { throw "Inspect existing MRI container $existing before creating another run." }
    $arguments = @('run', '-d', '--name', $trainingContainer, 'runner', '--config', 'configs/mri.toml', 'run', '--task', 'mri')
    if ($Resume) { $arguments += '--resume' }
    $containerId = (& docker @composeArguments @arguments | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or $containerId -notmatch '^[0-9a-f]{64}$') { throw 'Could not start MRI training.' }
    while ((& docker inspect --format '{{.State.Running}}' $containerId) -eq 'true') {
        $stats = & docker stats --no-stream --format '{{json .}}' $containerId
        $gpu = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
        [pscustomobject]@{ utc=(Get-Date).ToUniversalTime().ToString('o'); container=($stats | ConvertFrom-Json); gpu=$gpu } |
            ConvertTo-Json -Compress -Depth 4 | Add-Content -LiteralPath (Join-Path $auditDirectory 'training-resources.jsonl') -Encoding utf8
        & docker logs $containerId *> (Join-Path $auditDirectory 'training.log')
        $other = @(& docker ps --filter 'label=com.docker.compose.service=runner' --format '{{.Names}} {{.Command}}' |
            Where-Object { $_ -notmatch '^mri-' -and $_ -notmatch '^neural-exam-mri-' -and $_ -match 'run|smoke|train' })
        if ($other.Count -gt 0) {
            & docker stop --time 10 $containerId | Out-Null
            throw 'Another training runner appeared. MRI stopped with its last complete checkpoint preserved; resume when the GPU is free.'
        }
        Start-Sleep -Seconds 15
    }
    & docker logs $containerId *> (Join-Path $auditDirectory 'training.log')
    $exitCode = [int](& docker inspect --format '{{.State.ExitCode}}' $containerId)
    if ($exitCode -ne 0) { throw "MRI exited with $exitCode; preserve container and checkpoints, inspect training.log." }
    Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', 'scripts/audit_mri_results.py') -Log (Join-Path $auditDirectory 'result-integrity.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', 'scripts/prepare_mri_report.py') -Log (Join-Path $auditDirectory 'report-assets.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', '--entrypoint', 'jupyter', 'runner', 'nbconvert', '--execute', '--to', 'notebook', 'notebooks/01_mri.ipynb', '--output', '01_mri', '--output-dir', '/outputs/notebooks', '--ExecutePreprocessor.timeout=180') -Log (Join-Path $auditDirectory 'notebook.log')
    Invoke-MRIDocker -Arguments @('run', '--rm', 'reports', '--task', 'mri', '--draft') -Log (Join-Path $auditDirectory 'report-build.log')
    Write-Host "MRI experiments and draft artifacts completed in $outputDirectory"
} finally {
    Pop-Location
}
