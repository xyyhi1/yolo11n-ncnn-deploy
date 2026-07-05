#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCNN_DIR="${NCNN_DIR:-$HOME/local/ncnn/lib/cmake/ncnn}"

configure_command=(
  cmake
  -S "$ROOT"
  -B "$ROOT/build"
  -DCMAKE_BUILD_TYPE=Release
  -Dncnn_DIR="$NCNN_DIR"
)
"${configure_command[@]}"
cmake --build "$ROOT/build" -j"$(nproc)"

echo "Built: $ROOT/build/yolo_ncnn"
