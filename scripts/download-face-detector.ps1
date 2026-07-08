$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ModelsDir = Join-Path $ProjectRoot "models"
$TargetPath = Join-Path $ModelsDir "face_detection_yunet_2026may.onnx"
$SourceUrl = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

if (Test-Path $TargetPath) {
    $Header = Get-Content -Path $TargetPath -TotalCount 1 -ErrorAction SilentlyContinue
    if ($Header -match "git-lfs") {
        Write-Host "Replacing Git LFS pointer with the actual YuNet model..."
        Remove-Item -Path $TargetPath -Force
    }
}

if (Test-Path $TargetPath) {
    Write-Host "Face detector already exists: $TargetPath"
    exit 0
}

Write-Host "Downloading OpenCV YuNet face detector..."
Invoke-WebRequest -Uri $SourceUrl -OutFile $TargetPath
Write-Host "Face detector ready: $TargetPath"
