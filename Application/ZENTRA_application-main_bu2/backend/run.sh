#!/bin/bash
# run.sh — Start ZENTRA with correct Python version
# Usage: ./run.sh

PYTHON=python3.11

echo "🔍 ตรวจสอบ Python..."
if ! command -v $PYTHON &>/dev/null; then
    echo "❌ ไม่พบ $PYTHON — ติดตั้งด้วย: brew install python@3.11"
    exit 1
fi

echo "🔍 ตรวจสอบ inference-sdk..."
if ! $PYTHON -c "import inference_sdk" &>/dev/null; then
    echo "📦 ติดตั้ง dependencies..."
    pip3.11 install inference-sdk aiortc opencv-python requests
fi

echo "🚀 เริ่มระบบ ZENTRA..."
cd "$(dirname "$0")"
$PYTHON main.py
