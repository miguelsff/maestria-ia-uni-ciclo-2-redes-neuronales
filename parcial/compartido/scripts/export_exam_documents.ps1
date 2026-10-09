[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$RendererImage = 'exam-document-renderer:b1'
)

$ErrorActionPreference = 'Stop'
$inputRoot = (Resolve-Path -LiteralPath $InputDir).Path
$null = New-Item -ItemType Directory -Path $OutputDir -Force
$outputRoot = (Resolve-Path -LiteralPath $OutputDir).Path
$documents = @(Get-ChildItem -LiteralPath $inputRoot -Recurse -File -Filter '*.odt')
if ($documents.Count -eq 0) { throw 'No se encontraron maestros ODT en InputDir.' }

$imageId = & docker image inspect $RendererImage --format '{{.Id}}' 2>$null
if ($LASTEXITCODE -ne 0) {
    $rendererSource = Join-Path $PSScriptRoot '../plantillas/document_renderer'
    & docker build --tag $RendererImage $rendererSource
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo preparar el renderer administrado.' }
    $imageId = & docker image inspect $RendererImage --format '{{.Id}}'
}

$receipts = @()
foreach ($document in $documents) {
    $relative = [IO.Path]::GetRelativePath($inputRoot, $document.FullName)
    $relativePosix = $relative.Replace('\', '/')
    $relativeDirectory = [IO.Path]::GetDirectoryName($relative)
    $targetDirectory = Join-Path $outputRoot $relativeDirectory
    $null = New-Item -ItemType Directory -Path $targetDirectory -Force
    $stagingDirectory = Join-Path $targetDirectory ('.export-pdf-' + [Guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory -Path $stagingDirectory
    $before = (Get-FileHash -LiteralPath $document.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $containerInput = '/input/' + $relativePosix
    & docker run --rm --network none --read-only --tmpfs /tmp:rw,size=512m `
        --env XDG_CACHE_HOME=/tmp/cache --env SAL_USE_VCLPLUGIN=svp `
        --mount "type=bind,source=$inputRoot,target=/input,readonly" `
        --mount "type=bind,source=$stagingDirectory,target=/output" `
        $imageId soffice '-env:UserInstallation=file:///tmp/writer-profile' `
        --headless --norestore --convert-to 'pdf:writer_pdf_Export' --outdir /output $containerInput
    if ($LASTEXITCODE -ne 0) { throw "Falló la exportación de $relative." }
    $generatedPdf = Join-Path $stagingDirectory ($document.BaseName + '.pdf')
    if (-not (Test-Path -LiteralPath $generatedPdf -PathType Leaf)) { throw "No se produjo un PDF nuevo para $relative." }
    $after = (Get-FileHash -LiteralPath $document.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($before -ne $after) { throw "El maestro ODT cambió: $relative." }
    $targetPdf = Join-Path $targetDirectory ($document.BaseName + '.pdf')
    Move-Item -LiteralPath $generatedPdf -Destination $targetPdf -Force
    Remove-Item -LiteralPath $stagingDirectory
    $receipts += [ordered]@{
        odt = $relativePosix
        odt_sha256_before = $before
        odt_sha256_after = $after
        odt_preserved = $true
        pdf = [IO.Path]::GetRelativePath($outputRoot, $targetPdf).Replace('\', '/')
        pdf_sha256 = (Get-FileHash -LiteralPath $targetPdf -Algorithm SHA256).Hash.ToLowerInvariant()
        bytes_pdf = (Get-Item -LiteralPath $targetPdf).Length
    }
}

$version = & docker run --rm --network none $imageId soffice --version
$packages = & docker run --rm --network none $imageId cat /opt/document-renderer-packages.tsv
$fonts = & docker run --rm --network none $imageId cat /opt/document-renderer-fonts.txt
$evidence = [ordered]@{
    format_version = 1
    renderer_image = $RendererImage
    renderer_image_id = $imageId
    libreoffice_version = ($version -join "`n").Trim()
    source_of_pdf = 'Maestro ODT leído por LibreOffice Writer; sin TEX ni DOCX intermedio.'
    input_directory = $inputRoot
    output_directory = $outputRoot
    documents = $receipts
}
$evidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputRoot 'exportacion_documentos.json') -Encoding utf8
$packages | Set-Content -LiteralPath (Join-Path $outputRoot 'renderer_packages.tsv') -Encoding utf8
$fonts | Set-Content -LiteralPath (Join-Path $outputRoot 'renderer_fonts.txt') -Encoding utf8
Write-Output "Exportados $($documents.Count) PDF. Todos los maestros ODT conservan sus bytes."
