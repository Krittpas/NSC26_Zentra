# pack_for_release.ps1 - build a shippable ZENTRA zip for another machine.
# ----------------------------------------------------------------------------
# Strips heavy / secret files automatically (venv, __pycache__, archive, the
# SQLite DB, evidence snapshots, unused weights) but KEEPS everything needed to
# run (person_v2 / pose models, backend/.env, data/demo.mp4, zones.json).
# Kept ASCII-only on purpose so Windows PowerShell 5.1 never mis-parses it.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\pack_for_release.ps1
#   -> produces ZENTRA_release_<timestamp>.zip in the parent folder
# ----------------------------------------------------------------------------
param(
    [string]$OutDir = (Split-Path $PSScriptRoot -Parent)   # default: parent folder
)

$ErrorActionPreference = "Stop"
$src     = $PSScriptRoot
$stamp   = Get-Date -Format "yyyyMMdd_HHmm"
$staging = Join-Path $env:TEMP "zentra_pack_$stamp"
$zipPath = Join-Path $OutDir "ZENTRA_release_$stamp.zip"

Write-Host "[pack] source : $src"
Write-Host "[pack] staging: $staging"

# --- verify required (normally git-ignored) files are present before packing ---
$must = @(
    "backend\models\person_v2.pt",
    "backend\models\yolo11n-pose.pt",
    "backend\.env",
    "data\demo.mp4"
)
$missing = $must | Where-Object { -not (Test-Path (Join-Path $src $_)) }
if ($missing) {
    Write-Warning "Missing required files - the target machine may not run fully:"
    $missing | ForEach-Object { Write-Warning "  - $_" }
    Write-Host  "See docs\DEPLOYMENT_AND_USER_GUIDE.md section A."
}

# --- copy to staging, excluding what must not ship (robocopy) ---
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging | Out-Null

# directories to drop (matched by name at any depth)
$xd = @(
    ".venv", "venv", ".git", "__pycache__", "runs", "dist",
    "archive", "model_archive", ".claude",
    "collected", "fall_clips", "demo_clips", "train_dataset",
    "roboflow_dl", "eval_out", "snapshots", "reports"
)
# files to drop (secrets + unused heavy weights)
$xf = @(
    "*.pyc", "*.pyo", "*.log", "*.zip",
    "settings.json", "zentra.db",
    "person_v1.pt",
    "yolo11l.pt", "yolo11m.pt", "yolo11m-pose.pt", "yolo11s-pose.pt", "yolo11x.pt"
)

$roboArgs = @($src, $staging, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
$roboArgs += "/XD"; $roboArgs += $xd
$roboArgs += "/XF"; $roboArgs += $xf

Write-Host "[pack] copying (excluding venv/archive/db/snapshots/token) ..."
& robocopy @roboArgs | Out-Null
# robocopy exit codes: 0-7 = success, 8+ = real error
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }

# --- belt-and-suspenders: never let settings.json (LINE token) escape ---
$leak = Join-Path $staging "data\settings.json"
if (Test-Path $leak) { Remove-Item $leak -Force }

# --- compress to zip ---
Write-Host "[pack] compressing -> $zipPath"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -Force

# --- cleanup + summary ---
$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Remove-Item $staging -Recurse -Force
Write-Host ""
Write-Host "DONE: $zipPath  ($sizeMB MB)" -ForegroundColor Green
Write-Host "  Recipient: unzip, then follow docs\DEPLOYMENT_AND_USER_GUIDE.md section B."
Write-Host "  Note: backend\.env is included (its LINE token is blank = safe);"
Write-Host "        data\settings.json is EXCLUDED (holds the real token) - a fresh"
Write-Host "        machine will generate default settings on first run."
