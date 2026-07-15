#!/usr/bin/env bash
# run_native.sh — run ZENTRA natively on macOS so inference uses the Apple GPU
# (MPS) instead of Docker's CPU-only Linux VM. On this Mac the model does ~73 FPS
# @640 on MPS vs ~7 FPS in Docker — same code, same model, just the GPU.
#
#   ./run_native.sh              # port 7788, STREAM_FPS=30
#   STREAM_FPS=60 ./run_native.sh
#
# Then open http://127.0.0.1:7788/ in a browser. Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-7788}"
export STREAM_FPS="${STREAM_FPS:-30}"   # displayed video fps (detection runs faster)
export PPE_IMGSZ="${PPE_IMGSZ:-640}"    # 640 = deployed size; MPS handles 960 too
export INFERENCE_CONFIDENCE="${INFERENCE_CONFIDENCE:-0.30}"
export USE_LOCAL_MODEL=true             # use models/ppe_finetuned.pt
# PPE_INFER_DEVICE is intentionally UNSET → detect_track auto-picks MPS on macOS.

# Free the port: a running Docker container would hold 7788.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^zentra-app$'; then
  echo "[run_native] stopping Docker container zentra-app (frees port $PORT)…"
  docker stop zentra-app >/dev/null
fi

VENV_PY="./.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "ERROR: $VENV_PY not found — create the venv first"; exit 1; }

echo "[run_native] starting native server on http://127.0.0.1:$PORT  (MPS, STREAM_FPS=$STREAM_FPS, imgsz=$PPE_IMGSZ)"
exec "$VENV_PY" -m uvicorn server.api:app --host 127.0.0.1 --port "$PORT" --log-level warning
