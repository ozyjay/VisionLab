param(
    [ValidateSet("fast", "quality")]
    [string]$ResolutionMode = $(if ($env:VISION_CAMERA_RESOLUTION_MODE) { $env:VISION_CAMERA_RESOLUTION_MODE } else { "fast" })
)

$ErrorActionPreference = "Stop"

$env:VISION_CAMERA_RESOLUTION_MODE = $ResolutionMode

python -m src.main run
