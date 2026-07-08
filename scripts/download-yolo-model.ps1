$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $ProjectRoot ".venv/bin/python"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
}

if (-not (Test-Path $VenvPython)) {
    throw "Expected project virtual environment at $ProjectRoot/.venv. Run: pwsh -File scripts/setup.ps1 -ObjectDetection"
}

Set-Location $ProjectRoot
& $VenvPython scripts/download_yolo_model.py @args
