#!/usr/bin/env bash
# 使用 NCNN 官方 ncnn2table 和 ncnn2int8 完成 INT8 训练后量化。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCNN_TOOLS_DIR="${NCNN_TOOLS_DIR:-$HOME/projects/ncnn/build-tools/tools/quantize}"
FP32_DIR="$ROOT/models/yolo11n_hit_uav_ncnn"
INT8_DIR="$ROOT/models/yolo11n_hit_uav_ncnn_int8"
CALIBRATION_LIST="$ROOT/calibration/int8_calibration_images.txt"
CALIBRATION_TABLE="$ROOT/results/int8_calibration.table"

test -x "$NCNN_TOOLS_DIR/ncnn2table"
test -x "$NCNN_TOOLS_DIR/ncnn2int8"
test -f "$CALIBRATION_LIST"
mkdir -p "$INT8_DIR" "$ROOT/results"

table_command=(
  "$NCNN_TOOLS_DIR/ncnn2table"
  "$FP32_DIR/model.ncnn.param"
  "$FP32_DIR/model.ncnn.bin"
  "$CALIBRATION_LIST"
  "$CALIBRATION_TABLE"
  "mean=[0,0,0]"
  "norm=[0.003921568627,0.003921568627,0.003921568627]"
  "shape=[640,640,3]"
  "pixel=RGB"
  "thread=8"
  "method=kl"
)
"${table_command[@]}"

int8_command=(
  "$NCNN_TOOLS_DIR/ncnn2int8"
  "$FP32_DIR/model.ncnn.param"
  "$FP32_DIR/model.ncnn.bin"
  "$INT8_DIR/model.ncnn.param"
  "$INT8_DIR/model.ncnn.bin"
  "$CALIBRATION_TABLE"
)
"${int8_command[@]}"

ls -lh "$FP32_DIR/model.ncnn.bin" "$INT8_DIR/model.ncnn.bin"
