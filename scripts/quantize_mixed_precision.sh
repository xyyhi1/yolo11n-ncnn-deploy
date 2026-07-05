#!/usr/bin/env bash
# 根据两张混合精度校准表生成“输出层 FP32”和“检测头 FP32”两种模型。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCNN_TOOLS_DIR="${NCNN_TOOLS_DIR:-$HOME/projects/ncnn/build-tools/tools/quantize}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FP32_DIR="$ROOT/models/yolo11n_hit_uav_ncnn"

"$PYTHON_BIN" "$ROOT/scripts/create_mixed_precision_tables.py"

# 定义一个可复用函数，根据表文件生成对应混合精度模型。
quantize_profile() {
  local profile="$1"
  local table="$2"
  local output_dir="$ROOT/models/yolo11n_hit_uav_ncnn_int8_${profile}"
  mkdir -p "$output_dir"
  "$NCNN_TOOLS_DIR/ncnn2int8" \
    "$FP32_DIR/model.ncnn.param" \
    "$FP32_DIR/model.ncnn.bin" \
    "$output_dir/model.ncnn.param" \
    "$output_dir/model.ncnn.bin" \
    "$table"
  ls -lh "$output_dir/model.ncnn.bin"
}

quantize_profile "mixed_output_fp32" "$ROOT/results/int8_calibration_mixed_output_fp32.table"
quantize_profile "mixed_head_fp32" "$ROOT/results/int8_calibration_mixed_head_fp32.table"
