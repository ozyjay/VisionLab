param(
    [string]$VenvDir = ".venv-rocm312",
    [string]$ModelPath = $(if ($env:VISION_OBJECT_MODEL_PATH) { $env:VISION_OBJECT_MODEL_PATH } else { "models/yolo11s.pt" }),
    [string]$Device = $(if ($env:VISION_OBJECT_DEVICE) { $env:VISION_OBJECT_DEVICE } else { "auto" }),
    [ValidateSet("fast", "quality")]
    [string]$ResolutionMode = $(if ($env:VISION_CAMERA_RESOLUTION_MODE) { $env:VISION_CAMERA_RESOLUTION_MODE } else { "fast" }),
    [switch]$FaceDetection
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
$env:VISION_ENABLE_OBJECT_DETECTION = "true"
$env:VISION_OBJECT_MODEL_PATH = $ModelPath
$env:VISION_OBJECT_DEVICE = $Device
$env:VISION_CAMERA_RESOLUTION_MODE = $ResolutionMode

if ($FaceDetection) {
    $env:VISION_ENABLE_FACE_DETECTION = "true"
}

& $VenvPython -m src.main run
