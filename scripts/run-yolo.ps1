param(
    [string]$ModelPath = $(if ($env:VISION_OBJECT_MODEL_PATH) { $env:VISION_OBJECT_MODEL_PATH } else { "models/yolo11s.pt" }),
    [string]$Backend = $(if ($env:VISION_OBJECT_DETECTOR_BACKEND) { $env:VISION_OBJECT_DETECTOR_BACKEND } else { "auto" }),
    [string]$Prompts = $(if ($env:VISION_OBJECT_PROMPTS) { $env:VISION_OBJECT_PROMPTS } else { "" }),
    [string]$Device = $(if ($env:VISION_OBJECT_DEVICE) { $env:VISION_OBJECT_DEVICE } else { "cpu" }),
    [ValidateSet("fast", "quality")]
    [string]$ResolutionMode = $(if ($env:VISION_CAMERA_RESOLUTION_MODE) { $env:VISION_CAMERA_RESOLUTION_MODE } else { "fast" })
)

$ErrorActionPreference = "Stop"

$env:VISION_ENABLE_OBJECT_DETECTION = "true"
$env:VISION_OBJECT_MODEL_PATH = $ModelPath
$env:VISION_OBJECT_DETECTOR_BACKEND = $Backend
if ($Prompts) {
    $env:VISION_OBJECT_PROMPTS = $Prompts
}
$env:VISION_OBJECT_DEVICE = $Device
$env:VISION_CAMERA_RESOLUTION_MODE = $ResolutionMode

python -m src.main run
