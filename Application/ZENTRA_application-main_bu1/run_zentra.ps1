# run_zentra.ps1 — ZENTRA one-click launcher
# All AI runs IN-PROCESS via ultralytics — there is NO Docker inference server
# and nothing listens on :9001 (that was an older architecture). Just launch the
# desktop app; it starts uvicorn (127.0.0.1:7788) + the PyWebView window itself.
# ----------------------------------------------------------------
Set-Location $PSScriptRoot

# Prefer a project virtualenv's Python if present, else fall back to `python`.
$py = "python"
foreach ($cand in @(".venv\Scripts\python.exe", "venv\Scripts\python.exe")) {
    if (Test-Path $cand) { $py = $cand; break }
}

Write-Host "[ZENTRA] Launching desktop app ($py app.py) ..."
& $py app.py
