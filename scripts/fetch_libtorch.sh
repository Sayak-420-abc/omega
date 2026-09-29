#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"
CUDA="${LIBTORCH_CUDA:-0}"

if [ "$CUDA" = "1" ]; then
  URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.3.1%2Bcu121.zip}"
else
  URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.3.1%2Bcpu.zip}"
fi

echo "[*] Downloading LibTorch from: $URL"
TMP="/tmp/libtorch.zip"
curl -L --fail -o "$TMP" "$URL"
unzip -q -o "$TMP" -d "$ROOT/third_party"
rm -f "$TMP"
echo "[*] LibTorch ready at third_party/libtorch"
