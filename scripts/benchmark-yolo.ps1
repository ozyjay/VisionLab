param(
    [string]$VenvDir = ".venv-rocm312",
    [string]$ModelPath = $(if ($env:VISION_OBJECT_MODEL_PATH) { $env:VISION_OBJECT_MODEL_PATH } else { "models/yolo11s.pt" }),
    [string]$Backend = $(if ($env:VISION_OBJECT_DETECTOR_BACKEND) { $env:VISION_OBJECT_DETECTOR_BACKEND } else { "auto" }),
    [string]$Prompts = $(if ($env:VISION_OBJECT_PROMPTS) { $env:VISION_OBJECT_PROMPTS } else { "" }),
    [string]$Device = $(if ($env:VISION_OBJECT_DEVICE) { $env:VISION_OBJECT_DEVICE } else { "auto" }),
    [int]$Frames = 30,
    [int]$Warmup = 5
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = if ([System.IO.Path]::IsPathRooted($VenvDir)) { $VenvDir } else { Join-Path $ProjectRoot $VenvDir }
$VenvPython = Join-Path $VenvPath "bin/python"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = Join-Path $VenvPath "Scripts/python.exe"
}

if (-not (Test-Path $VenvPython)) {
    throw "ROCm virtual environment not found. Run: pwsh -File scripts/setup-rocm.ps1"
}

Set-Location $ProjectRoot
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

& $VenvPython scripts/benchmark_yolo.py --model $ModelPath --backend $Backend --prompts $Prompts --device $Device --frames $Frames --warmup $Warmup
