#!/usr/bin/env bash
# 文件作用：使用 NCNN 官方 ncnn2table 和 ncnn2int8 完成 INT8 训练后量化。
# 中文说明：启用严格模式，任何失败、未定义变量或管道错误都会终止脚本。
set -euo pipefail

# 中文说明：计算当前项目根目录的绝对路径。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：允许通过环境变量覆盖 NCNN 工具目录，否则使用本机已编译位置。
NCNN_TOOLS_DIR="${NCNN_TOOLS_DIR:-$HOME/projects/ncnn/build-tools/tools/quantize}"
# 中文说明：定义 FP32 模型目录。
FP32_DIR="$ROOT/models/yolo11n_hit_uav_ncnn"
# 中文说明：定义 INT8 模型输出目录。
INT8_DIR="$ROOT/models/yolo11n_hit_uav_ncnn_int8"
# 中文说明：定义校准图片绝对路径清单。
CALIBRATION_LIST="$ROOT/calibration/int8_calibration_images.txt"
# 中文说明：定义 ncnn2table 生成的激活和权重量化尺度表。
CALIBRATION_TABLE="$ROOT/results/int8_calibration.table"

# 中文说明：确认 ncnn2table 已经成功编译且可以执行。
test -x "$NCNN_TOOLS_DIR/ncnn2table"
# 中文说明：确认 ncnn2int8 已经成功编译且可以执行。
test -x "$NCNN_TOOLS_DIR/ncnn2int8"
# 中文说明：确认校准清单已经由 prepare_int8_calibration.py 生成。
test -f "$CALIBRATION_LIST"
# 中文说明：创建 INT8 模型输出目录和结果目录。
mkdir -p "$INT8_DIR" "$ROOT/results"

# 中文说明：构造校准表生成命令。
table_command=(
  # 中文说明：调用 NCNN 官方校准工具。
  "$NCNN_TOOLS_DIR/ncnn2table"
  # 中文说明：传入 FP32 网络结构文件。
  "$FP32_DIR/model.ncnn.param"
  # 中文说明：传入 FP32 权重文件。
  "$FP32_DIR/model.ncnn.bin"
  # 中文说明：传入经过 Letterbox 的校准图片清单。
  "$CALIBRATION_LIST"
  # 中文说明：指定校准表输出位置。
  "$CALIBRATION_TABLE"
  # 中文说明：部署前处理没有减均值，因此三个通道均为零。
  "mean=[0,0,0]"
  # 中文说明：除以 255 等价于每个通道乘以 0.003921568627。
  "norm=[0.003921568627,0.003921568627,0.003921568627]"
  # 中文说明：校准图片已经是 640×640，工具按三通道输入模型。
  "shape=[640,640,3]"
  # 中文说明：OpenCV 文件是 BGR，pixel=RGB 会在输入 NCNN 前完成 BGR 转 RGB。
  "pixel=RGB"
  # 中文说明：使用八个 CPU 线程加速校准统计。
  "thread=8"
  # 中文说明：使用 KL 散度寻找激活截断阈值。
  "method=kl"
)
# 中文说明：运行校准过程并生成每层量化尺度表。
"${table_command[@]}"

# 中文说明：构造 FP32 到 INT8 的模型转换命令。
int8_command=(
  # 中文说明：调用 NCNN 官方 INT8 转换工具。
  "$NCNN_TOOLS_DIR/ncnn2int8"
  # 中文说明：输入 FP32 网络结构。
  "$FP32_DIR/model.ncnn.param"
  # 中文说明：输入 FP32 权重。
  "$FP32_DIR/model.ncnn.bin"
  # 中文说明：输出 INT8 网络结构。
  "$INT8_DIR/model.ncnn.param"
  # 中文说明：输出 INT8 权重。
  "$INT8_DIR/model.ncnn.bin"
  # 中文说明：使用前一步生成的量化尺度表。
  "$CALIBRATION_TABLE"
)
# 中文说明：执行模型量化转换。
"${int8_command[@]}"

# 中文说明：打印量化完成后的模型文件大小，便于与 FP32 对比。
ls -lh "$FP32_DIR/model.ncnn.bin" "$INT8_DIR/model.ncnn.bin"
