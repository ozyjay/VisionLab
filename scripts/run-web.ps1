param(
    [string]$ModelPath = $(if ($env:VISION_OBJECT_MODEL_PATH) { $env:VISION_OBJECT_MODEL_PATH } else { "models/yoloe-26s-seg.pt" }),
    [string]$Backend = $(if ($env:VISION_OBJECT_DETECTOR_BACKEND) { $env:VISION_OBJECT_DETECTOR_BACKEND } else { "auto" }),
    [string]$Prompts = $(if ($env:VISION_OBJECT_PROMPTS) { $env:VISION_OBJECT_PROMPTS } else { "person,mobile phone,computer mouse,pen,watch,keys,wallet,mug,keyboard,cable" }),
    [string]$Device = $(if ($env:VISION_OBJECT_DEVICE) { $env:VISION_OBJECT_DEVICE } else { "cpu" }),
    [string]$HostName = $(if ($env:VISION_WEB_HOST) { $env:VISION_WEB_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:VISION_WEB_PORT) { [int]$env:VISION_WEB_PORT } else { 8019 }),
    [ValidateSet("fast", "quality")]
    [string]$ResolutionMode = $(if ($env:VISION_CAMERA_RESOLUTION_MODE) { $env:VISION_CAMERA_RESOLUTION_MODE } else { "quality" })
)

$ErrorActionPreference = "Stop"

$env:VISION_ENABLE_OBJECT_DETECTION = "true"
$env:VISION_OBJECT_MODEL_PATH = $ModelPath
$env:VISION_OBJECT_DETECTOR_BACKEND = $Backend
$env:VISION_OBJECT_PROMPTS = $Prompts
$env:VISION_OBJECT_DEVICE = $Device
$env:VISION_WEB_HOST = $HostName
$env:VISION_WEB_PORT = [string]$Port
$env:VISION_CAMERA_RESOLUTION_MODE = $ResolutionMode

Write-Host "Open http://$HostName`:$Port in your browser."
python -m src.main web
