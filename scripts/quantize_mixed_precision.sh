#!/usr/bin/env bash
# 文件作用：根据两张混合精度校准表生成“输出层 FP32”和“检测头 FP32”两种模型。
# 中文说明：启用 Shell 严格模式。
set -euo pipefail

# 中文说明：取得项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：允许覆盖 NCNN 量化工具目录。
NCNN_TOOLS_DIR="${NCNN_TOOLS_DIR:-$HOME/projects/ncnn/build-tools/tools/quantize}"
# 中文说明：允许通过环境变量选择 Python 解释器，默认使用系统 python3。
PYTHON_BIN="${PYTHON_BIN:-python3}"
# 中文说明：定义 FP32 输入模型目录。
FP32_DIR="$ROOT/models/yolo11n_hit_uav_ncnn"

# 中文说明：执行 Python 脚本，生成两张注释敏感层权重尺度的量化表。
"$PYTHON_BIN" "$ROOT/scripts/create_mixed_precision_tables.py"

# 中文说明：定义一个可复用函数，根据表文件生成对应混合精度模型。
quantize_profile() {
  # 中文说明：第一个参数是模型名称后缀。
  local profile="$1"
  # 中文说明：第二个参数是混合精度校准表路径。
  local table="$2"
  # 中文说明：构造当前模型输出目录。
  local output_dir="$ROOT/models/yolo11n_hit_uav_ncnn_int8_${profile}"
  # 中文说明：创建模型输出目录。
  mkdir -p "$output_dir"
  # 中文说明：调用 ncnn2int8；被注释权重尺度的层会保留 FP32。
  "$NCNN_TOOLS_DIR/ncnn2int8" \
    "$FP32_DIR/model.ncnn.param" \
    "$FP32_DIR/model.ncnn.bin" \
    "$output_dir/model.ncnn.param" \
    "$output_dir/model.ncnn.bin" \
    "$table"
  # 中文说明：打印当前混合精度权重文件大小。
  ls -lh "$output_dir/model.ncnn.bin"
}

# 中文说明：生成只回退最终输出卷积的模型。
quantize_profile "mixed_output_fp32" "$ROOT/results/int8_calibration_mixed_output_fp32.table"
# 中文说明：生成回退整个检测头的模型。
quantize_profile "mixed_head_fp32" "$ROOT/results/int8_calibration_mixed_head_fp32.table"
