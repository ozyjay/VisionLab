param(
    [switch]$ObjectDetection
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = if ($env:VISION_VENV_DIR) { $env:VISION_VENV_DIR } else { Join-Path $ProjectRoot ".venv" }
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$InstallObjectDetection = $ObjectDetection -or ($env:VISION_INSTALL_OBJECT_DETECTION -match "^(1|true|yes|on)$")

Set-Location $ProjectRoot

Write-Host "Local AI Vision Assistant setup"
Write-Host "Project: $ProjectRoot"
Write-Host "Virtual environment: $VenvDir"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating project-local virtual environment..."
    & $PythonBin -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "bin/python"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = Join-Path $VenvDir "Scripts/python.exe"
}

if (-not (Test-Path $VenvPython)) {
    throw "Expected Python executable was not found in $VenvDir"
}

$ActualPrefix = & $VenvPython -c "import sys; print(sys.prefix)"
$ExpectedPrefix = (Resolve-Path $VenvDir).Path

if ($ActualPrefix -ne $ExpectedPrefix) {
    throw "Refusing to install outside the project virtual environment. Expected prefix: $ExpectedPrefix; actual prefix: $ActualPrefix"
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing dependencies..."
& $VenvPython -m pip install -r requirements.txt

if ($InstallObjectDetection) {
    Write-Host "Installing optional object-detection dependencies..."
    & $VenvPython -m pip install -r requirements-object-detection.txt
} else {
    Write-Host "Skipping optional object-detection dependencies."
    Write-Host "Install them later with:"
    Write-Host "  ./.venv/bin/python -m pip install -r requirements-object-detection.txt"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with:"
Write-Host "  ./.venv/bin/Activate.ps1"
Write-Host ""
Write-Host "Run health check:"
Write-Host "  ./.venv/bin/python -m src.main health"
