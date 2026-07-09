param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "web", "run", "health", "config", "download-models", "face-model", "benchmark", "gpu")]
    [string]$Command = "web",

    [switch]$ObjectDetection,
    [switch]$DownloadModels,
    [ValidateSet("demo", "all-useful")]
    [string]$ModelBundle = $(if ($env:VISION_MODEL_BUNDLE) { $env:VISION_MODEL_BUNDLE } else { "demo" }),

    [string]$ModelPath = $(if ($env:VISION_OBJECT_MODEL_PATH) { $env:VISION_OBJECT_MODEL_PATH } else { "models/yoloe-26s-seg.pt" }),
    [string]$Backend = $(if ($env:VISION_OBJECT_DETECTOR_BACKEND) { $env:VISION_OBJECT_DETECTOR_BACKEND } else { "auto" }),
    [string]$Prompts = $(if ($env:VISION_OBJECT_PROMPTS) { $env:VISION_OBJECT_PROMPTS } else { "person,mobile phone,computer mouse,pen,watch,keys,wallet,mug,keyboard,cable" }),
    [string]$Device = $(if ($env:VISION_OBJECT_DEVICE) { $env:VISION_OBJECT_DEVICE } else { "auto" }),
    [string]$HostName = $(if ($env:VISION_WEB_HOST) { $env:VISION_WEB_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:VISION_WEB_PORT) { [int]$env:VISION_WEB_PORT } else { 8019 }),
    [ValidateSet("fast", "quality")]
    [string]$ResolutionMode = $(if ($env:VISION_CAMERA_RESOLUTION_MODE) { $env:VISION_CAMERA_RESOLUTION_MODE } else { "quality" }),

    [int]$Frames = 30,
    [int]$Warmup = 5,
    [string]$VenvDir = $(if ($env:VISION_VENV_DIR) { $env:VISION_VENV_DIR } else { ".venv-rocm312" })
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Get-VenvPython {
    $VenvPath = if ([System.IO.Path]::IsPathRooted($VenvDir)) { $VenvDir } else { Join-Path $ProjectRoot $VenvDir }
    $VenvPython = Join-Path $VenvPath "bin/python"
    if (-not (Test-Path $VenvPython)) {
        $VenvPython = Join-Path $VenvPath "Scripts/python.exe"
    }
    if (-not (Test-Path $VenvPython)) {
        throw "ROCm virtual environment not found. Run: pwsh -File scripts/vision.ps1 setup"
    }
    return $VenvPython
}

function Set-DemoObjectEnvironment {
    $env:VISION_ENABLE_OBJECT_DETECTION = "true"
    $env:VISION_OBJECT_MODEL_PATH = $ModelPath
    $env:VISION_OBJECT_DETECTOR_BACKEND = $Backend
    $env:VISION_OBJECT_PROMPTS = $Prompts
    $env:VISION_OBJECT_DEVICE = $Device
    $env:VISION_CAMERA_RESOLUTION_MODE = $ResolutionMode
}

switch ($Command) {
    "setup" {
        & pwsh -File (Join-Path $PSScriptRoot "setup-rocm.ps1") -VenvDir $VenvDir
        if ($DownloadModels) {
            $VenvPython = Get-VenvPython
            & $VenvPython scripts/download_yolo_model.py $ModelBundle
            & pwsh -File (Join-Path $PSScriptRoot "download-face-detector.ps1")
        }
        break
    }

    "web" {
        $VenvPython = Get-VenvPython
        Set-DemoObjectEnvironment
        $env:VISION_WEB_HOST = $HostName
        $env:VISION_WEB_PORT = [string]$Port
        Write-Host "Open http://$HostName`:$Port in your browser."
        & $VenvPython -m src.main web
        break
    }

    "run" {
        $VenvPython = Get-VenvPython
        & $VenvPython -m src.main run
        break
    }

    "health" {
        $VenvPython = Get-VenvPython
        & $VenvPython -m src.main health
        break
    }

    "config" {
        $VenvPython = Get-VenvPython
        & $VenvPython -m src.main config
        break
    }

    "download-models" {
        $VenvPython = Get-VenvPython
        & $VenvPython scripts/download_yolo_model.py $ModelBundle
        break
    }

    "face-model" {
        & pwsh -File (Join-Path $PSScriptRoot "download-face-detector.ps1")
        break
    }

    "benchmark" {
        $VenvPython = Get-VenvPython
        & $VenvPython scripts/benchmark_yolo.py --model $ModelPath --backend $Backend --prompts $Prompts --device $Device --frames $Frames --warmup $Warmup
        break
    }

    "gpu" {
        & pwsh -File (Join-Path $PSScriptRoot "run-gpu.ps1") -ModelPath $ModelPath -Backend $Backend -Prompts $Prompts -Device $Device -ResolutionMode $ResolutionMode
        break
    }
}
