# run_zentra.ps1 — ZENTRA one-click launcher
# 1) make sure the Roboflow inference server is up on :9001
# 2) launch the desktop app
# ----------------------------------------------------------------
$Port = 9001
$Img  = 'roboflow/roboflow-inference-server-cpu:latest'
$Name = 'zentra-inference'

function Test-Inference {
    try {
        Invoke-WebRequest "http://localhost:$Port/" -TimeoutSec 3 -UseBasicParsing | Out-Null
        return $true
    } catch { return $false }
}

Write-Host "[ZENTRA] Checking inference server on port $Port ..."
if (Test-Inference) {
    Write-Host "[ZENTRA] Inference server already running."
} else {
    # Ensure the Docker engine is up
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ZENTRA] Starting Docker Desktop (first run can take ~1 min) ..."
        $dd = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (Test-Path $dd) { Start-Process $dd }
        for ($i = 0; $i -lt 40; $i++) {
            docker info *> $null
            if ($LASTEXITCODE -eq 0) { break }
            Start-Sleep -Seconds 3
        }
    }
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ZENTRA] ERROR: Docker engine not available. PPE/Zone need it."
    } else {
        # Reuse an existing container if present, otherwise create one
        $exists = docker ps -a --filter "name=$Name" --format '{{.Names}}'
        if ($exists -eq $Name) {
            docker start $Name | Out-Null
        } else {
            Write-Host "[ZENTRA] Starting inference server (pulls image on first run) ..."
            docker run -d --name $Name --restart unless-stopped -p "${Port}:9001" $Img | Out-Null
        }
        Write-Host "[ZENTRA] Waiting for inference server to respond ..."
        for ($i = 0; $i -lt 30; $i++) {
            if (Test-Inference) { break }
            Start-Sleep -Seconds 3
        }
        if (Test-Inference) { Write-Host "[ZENTRA] Inference server ready." }
        else { Write-Host "[ZENTRA] WARNING: server not responding yet; PPE/Zone may be idle until it is." }
    }
}

Write-Host "[ZENTRA] Launching desktop app ..."
Set-Location $PSScriptRoot
python app.py
