#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/third_party" "$ROOT/models" "$ROOT/assets"

ORT_VER="${ORT_VER:-1.18.0}"
ORT_USE_CUDA="${ORT_USE_CUDA:-0}"
UNAME="$(uname -s || echo unknown)"

download_linux_cpu() {
  local B="onnxruntime-linux-x64-${ORT_VER}"
  echo "[*] Downloading ONNX Runtime (CPU) ${ORT_VER}"
  curl -L --fail -o /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${B}.tgz"
  mkdir -p "$ROOT/third_party/onnxruntime"
  tar -xzf /tmp/ort.tgz -C "$ROOT/third_party"
  mv "$ROOT/third_party/${B}"/* "$ROOT/third_party/onnxruntime"/
  rm -f /tmp/ort.tgz
}

download_linux_gpu() {
  local B="onnxruntime-linux-x64-gpu-${ORT_VER}"
  echo "[*] Downloading ONNX Runtime (CUDA) ${ORT_VER}"
  curl -L --fail -o /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${B}.tgz"
  mkdir -p "$ROOT/third_party/onnxruntime"
  tar -xzf /tmp/ort.tgz -C "$ROOT/third_party"
  mv "$ROOT/third_party/${B}"/* "$ROOT/third_party/onnxruntime"/
  rm -f /tmp/ort.tgz
}

download_windows() {
  local B="onnxruntime-win-x64-${ORT_VER}"
  echo "[*] Downloading ONNX Runtime (Windows CPU) ${ORT_VER}"
  local ZIP="/tmp/ort_win.zip"
  curl -L --fail -o "$ZIP" "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${B}.zip"
  mkdir -p "$ROOT/third_party"
  powershell -Command "Expand-Archive -Path '${ZIP}' -DestinationPath '${ROOT}/third_party' -Force"
  mkdir -p "$ROOT/third_party/onnxruntime"
  if [ -d "$ROOT/third_party/${B}" ]; then
    cp -r "$ROOT/third_party/${B}/"* "$ROOT/third_party/onnxruntime"/
  fi
  rm -f "$ZIP"
}

if [ ! -d "$ROOT/third_party/onnxruntime/include" ]; then
  case "$UNAME" in
    Linux*) if [ "$ORT_USE_CUDA" = "1" ]; then download_linux_gpu || download_linux_cpu; else download_linux_cpu; fi ;;
    MINGW*|MSYS*|CYGWIN*) download_windows ;;
    *) echo "Skipping ONNX Runtime fetch for $UNAME" ;;
  esac
fi

# SqueezeNet 1.1 (validated path)
MODEL="$ROOT/models/squeezenet1.1.onnx"
URL="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
if [ ! -f "$MODEL" ] || [ ! -s "$MODEL" ] || [ $(stat -c%s "$MODEL" 2>/dev/null || echo 0) -lt 1000000 ]; then
  echo "[*] Downloading SqueezeNet 1.1 (ONNX)"
  rm -f "$MODEL"
  curl -L --fail --retry 4 --retry-connrefused -o "$MODEL" "$URL"
  if head -c 200 "$MODEL" | grep -qi "<!DOCTYPE html>"; then
    echo "[!] Model download returned HTML (rate limit?). Please retry later."
    exit 1
  fi
fi

# Sample image and labels
if [ ! -f "$ROOT/assets/sample.jpg" ]; then
  echo "[*] Downloading sample image"
  curl -L --fail --retry 4 --retry-connrefused \
    -o "$ROOT/assets/sample.jpg" \
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
fi

if [ ! -f "$ROOT/assets/imagenet_labels.txt" ]; then
  echo "[*] Downloading ImageNet labels"
  curl -L --fail --retry 4 --retry-connrefused \
    -o "$ROOT/assets/imagenet_labels.txt" \
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
fi

echo "[*] Assets ready."
