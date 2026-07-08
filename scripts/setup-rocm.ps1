param(
    [string]$VenvDir = ".venv-rocm312",
    [string]$PythonBin = "python3.12",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = if ([System.IO.Path]::IsPathRooted($VenvDir)) { $VenvDir } else { Join-Path $ProjectRoot $VenvDir }

$TorchWheel = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
$TorchvisionWheel = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
$TorchaudioWheel = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
$TritonWheel = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

Set-Location $ProjectRoot

function Resolve-Python312 {
    param([string]$RequestedPython)

    try {
        $Version = & $RequestedPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $Version -eq "3.12") {
            return $RequestedPython
        }
    } catch {
        # Continue to pyenv fallback below.
    }

    $PyenvVersions = "/home/jase/.pyenv/versions"
    if (Test-Path $PyenvVersions) {
        $Candidate = Get-ChildItem -Directory $PyenvVersions |
            Where-Object { $_.Name -like "3.12.*" } |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($Candidate) {
            $CandidatePython = Join-Path $Candidate.FullName "bin/python"
            if (Test-Path $CandidatePython) {
                return $CandidatePython
            }
        }
    }

    return $RequestedPython
}

$ResolvedPythonBin = Resolve-Python312 -RequestedPython $PythonBin

Write-Host "VisionLab ROCm setup"
Write-Host "Project: $ProjectRoot"
Write-Host "Virtual environment: $VenvPath"
Write-Host "Python: $ResolvedPythonBin"

$VenvPython = Join-Path $VenvPath "bin/python"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = Join-Path $VenvPath "Scripts/python.exe"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python 3.12 virtual environment..."
    & $ResolvedPythonBin -m venv $VenvPath
    $VenvPython = Join-Path $VenvPath "bin/python"
    if (-not (Test-Path $VenvPython)) {
        $VenvPython = Join-Path $VenvPath "Scripts/python.exe"
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "Expected Python executable was not found in $VenvPath"
}

$ActualPrefix = & $VenvPython -c "import sys; print(sys.prefix)"
$ExpectedPrefix = (Resolve-Path $VenvPath).Path
if ($ActualPrefix -ne $ExpectedPrefix) {
    throw "Refusing to install outside the project virtual environment. Expected prefix: $ExpectedPrefix; actual prefix: $ActualPrefix"
}

$PythonVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PythonVersion -ne "3.12") {
    throw "ROCm PyTorch wheels for this setup require Python 3.12. Found Python $PythonVersion at $VenvPython"
}

Write-Host "Upgrading pip and wheel..."
& $VenvPython -m pip install --upgrade pip wheel

if (-not $SkipTorch) {
    Write-Host "Removing any existing Torch packages from this ROCm environment..."
    & $VenvPython -m pip uninstall -y torch torchvision torchaudio triton

    Write-Host "Installing AMD ROCm PyTorch wheels..."
    & $VenvPython -m pip install $TorchWheel $TorchvisionWheel $TorchaudioWheel $TritonWheel
} else {
    Write-Host "Skipping Torch wheel installation."
}

Write-Host "Installing base app dependencies..."
& $VenvPython -m pip install -r requirements.txt

Write-Host "Installing object-detection dependencies..."
& $VenvPython -m pip install -r requirements-object-detection.txt

Write-Host "Verifying Torch accelerator access..."
& $VenvPython -c "import torch; print('torch', torch.__version__); print('hip', torch.version.hip); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

Write-Host ""
Write-Host "ROCm setup complete."
Write-Host "Run GPU viewer:"
Write-Host "  pwsh -File scripts/run-gpu.ps1"
