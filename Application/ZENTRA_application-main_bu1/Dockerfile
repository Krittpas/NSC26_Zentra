# ZENTRA PPE app — single container: web UI + pipeline + engine (ultralytics
# ByteTrack in-process) using our own ppe_finetuned.pt. Cameras via RTSP.
FROM python:3.11-slim

# ffmpeg = RTSP decode; libglib = opencv-headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first (layer cache); slim set — no mediapipe/roboflow/inference-sdk.
# docker/wheels holds the prebuilt lapx wheel (the `lap` module ultralytics
# ByteTrack requires) so tracking installs even if the build host's network is
# flaky; --find-links prefers it, the rest come from PyPI.
COPY docker/requirements-app.txt ./req.txt
COPY docker/wheels/ ./wheels/
RUN pip install --no-cache-dir --find-links=./wheels -r req.txt

# app code + backend engine + the deployed model (heavy dirs excluded via .dockerignore)
COPY . .

ENV PYTHONUTF8=1 \
    PPE_INFER_DEVICE=cpu \
    USE_LOCAL_MODEL=true \
    YOLO_CONFIG_DIR=/tmp/ultralytics

EXPOSE 7788
# web app (uvicorn) — browser views the live annotated feed at :7788
CMD ["python3", "-m", "uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "7788"]
