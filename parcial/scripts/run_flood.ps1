param([switch]$Resume)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$outputDirectory = if ($env:FLOOD_OUTPUT_DIR) { $env:FLOOD_OUTPUT_DIR } else { 'E:/Kaggle/exam-output-flood' }
$auditDirectory = Join-Path $outputDirectory 'audit'
$composeArguments = @('compose', '-f', 'compose.flood.yaml')
$trainingContainer = 'neural-exam-flood-training'

function Invoke-FloodDocker {
    param([string[]]$Arguments, [string]$Log)
    & docker @composeArguments @Arguments *> $Log
    if ($LASTEXITCODE -ne 0) { throw "Flood operation failed; inspect $Log" }
}

function Wait-FloodResources {
    while ($true) {
        $wildlife = @(& docker ps --filter 'label=com.docker.compose.project=neural-exam-wildlife' --filter 'label=com.docker.compose.service=runner' --format '{{.Names}}')
        $memoryLine = & wsl -d docker-desktop -e sh -c 'cat /proc/meminfo' | Select-String '^MemAvailable:'
        if (-not $memoryLine) { throw 'Cannot inspect available WSL memory.' }
        $availableKiB = [long]([regex]::Match($memoryLine.ToString(), '\d+').Value)
        if ($wildlife.Count -eq 0 -and $availableKiB -ge 4 * 1024 * 1024) { return }
        Write-Host 'Waiting for the wildlife runner and sufficient WSL memory.'
        Start-Sleep -Seconds 30
    }
}

Push-Location $projectDirectory
try {
    New-Item -ItemType Directory -Force -Path $auditDirectory | Out-Null
    Invoke-FloodDocker -Arguments @('build', 'runner', 'reports') -Log (Join-Path $auditDirectory 'build.log')
    Invoke-FloodDocker -Arguments @('run', '--rm', 'data-cache') -Log (Join-Path $auditDirectory 'data-cache.log')
    Wait-FloodResources
    Invoke-FloodDocker -Arguments @('run', '--rm', 'runner', '--config', 'configs/flood.toml', 'doctor', '--gpu-check') -Log (Join-Path $auditDirectory 'doctor.log')
    Invoke-FloodDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', '-m', 'pytest', '-q', 'tests/test_flood.py', 'tests/test_metrics.py::test_nodata_does_not_enter_confusion_counts', 'tests/test_utilities.py::test_resume_restores_optimizer_rng_and_loader_state', '-p', 'no:cacheprovider') -Log (Join-Path $auditDirectory 'tests.log')
    $smokeArguments = @('run', '--rm', 'runner', '--config', 'configs/flood.toml', 'smoke', '--task', 'flood')
    if (Test-Path -LiteralPath (Join-Path $outputDirectory 'smoke/flood/gold_bce/last.pt')) { $smokeArguments += '--resume' }
    Invoke-FloodDocker -Arguments $smokeArguments -Log (Join-Path $auditDirectory 'smoke.log')
    $attempt = 0
    $existingContainer = & docker ps -a --filter "name=^/$trainingContainer$" --format '{{.ID}}'
    if ($existingContainer) { throw "The flood training container already exists ($existingContainer); inspect it before starting another run." }
    while ($true) {
        Wait-FloodResources
        $attempt += 1
        $trainingArguments = @('run', '-d', '--name', $trainingContainer, 'runner', '--config', 'configs/flood.toml', 'run', '--task', 'flood')
        if ($Resume -or $attempt -gt 1) { $trainingArguments += '--resume' }
        $containerId = (& docker @composeArguments @trainingArguments | Select-Object -Last 1).Trim()
        if ($LASTEXITCODE -ne 0 -or $containerId -notmatch '^[0-9a-f]{64}$') { throw 'Could not start the flood training container.' }
        $trainingLog = Join-Path $auditDirectory "training-attempt-$attempt.log"
        $preempted = $false
        while ((& docker inspect --format '{{.State.Running}}' $containerId) -eq 'true') {
            $stats = & docker stats --no-stream --format '{{json .}}' $containerId
            $gpu = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
            $containerStats = if ($stats) { $stats | ConvertFrom-Json } else { $null }
            [pscustomobject]@{ utc=(Get-Date).ToUniversalTime().ToString('o'); container=$containerStats; gpu=$gpu } | ConvertTo-Json -Compress -Depth 4 | Add-Content -LiteralPath (Join-Path $auditDirectory 'training-resources.jsonl') -Encoding utf8
            & docker logs $containerId *> $trainingLog
            $wildlife = @(& docker ps --filter 'label=com.docker.compose.project=neural-exam-wildlife' --filter 'label=com.docker.compose.service=runner' --format '{{.Names}}')
            if ($wildlife.Count -gt 0) {
                & docker stop --time 10 $containerId | Out-Null
                $preempted = $true
                break
            }
            Start-Sleep -Seconds 15
        }
        & docker logs $containerId *> $trainingLog
        $exitCode = [int](& docker inspect --format '{{.State.ExitCode}}' $containerId)
        $oomKilled = (& docker inspect --format '{{.State.OOMKilled}}' $containerId) -eq 'true'
        $recordedName = & docker inspect --format '{{.Name}}' $containerId
        if ($recordedName -ne "/$trainingContainer") { throw 'Unexpected container identity; preserving it.' }
        if ($oomKilled -and -not $preempted) {
            throw "Flood exceeded available memory; container and checkpoints preserved. Inspect $trainingLog before resuming."
        }
        if ($exitCode -eq 0 -or $preempted) {
            & docker rm $containerId | Out-Null
        } else {
            throw "Flood training exited with $exitCode; container and checkpoints preserved. Inspect $trainingLog"
        }
        if ($exitCode -eq 0) { break }
        Write-Host 'Flood checkpoint preserved; waiting before resuming.'
        Wait-FloodResources
    }
    Invoke-FloodDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', 'scripts/audit_flood_results.py') -Log (Join-Path $auditDirectory 'result-integrity.log')
    Invoke-FloodDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', '-m', 'neural_exam.flood.qualitative', '--config', 'configs/flood.toml') -Log (Join-Path $auditDirectory 'qualitative.log')
    Invoke-FloodDocker -Arguments @('run', '--rm', '--entrypoint', 'python', 'runner', '-m', 'neural_exam.flood.reporting', '--config', 'configs/flood.toml') -Log (Join-Path $auditDirectory 'report-assets.log')
    Invoke-FloodDocker -Arguments @('run', '--rm', '--entrypoint', 'jupyter', 'runner', 'nbconvert', '--execute', '--to', 'notebook', 'notebooks/02_flood.ipynb', '--output', '02_flood', '--output-dir', '/outputs/notebooks', '--ExecutePreprocessor.timeout=180') -Log (Join-Path $auditDirectory 'notebook.log')
    if (-not (Test-Path -LiteralPath (Join-Path $outputDirectory 'report-assets/neurips_2026.sty'))) {
        Invoke-FloodDocker -Arguments @('run', '--rm', 'style') -Log (Join-Path $auditDirectory 'style.log')
    }
    Invoke-FloodDocker -Arguments @('run', '--rm', 'reports', '--task', 'flood', '--draft') -Log (Join-Path $auditDirectory 'report-build.log')
    Write-Host "Flood experiments and draft artifacts completed in $outputDirectory"
} finally {
    Pop-Location
}
