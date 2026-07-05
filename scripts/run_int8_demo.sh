#!/usr/bin/env bash
# 文件作用：运行最终推荐的“骨干与颈部 INT8、检测头 FP32”混合精度模型。
# 中文说明：启用严格模式，任意错误都会停止脚本。
set -euo pipefail

# 中文说明：取得项目根目录并切换过去。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：切换工作目录，保证相对路径稳定。
cd "$ROOT"

# 中文说明：使用数组保存完整命令，避免反斜杠续行与注释冲突。
command=(
  # 中文说明：调用已经编译好的 C++ 推理程序。
  ./build/yolo_ncnn
  # 中文说明：使用混合精度 NCNN 网络结构。
  --param models/yolo11n_hit_uav_ncnn_int8_mixed_head_fp32/model.ncnn.param
  # 中文说明：使用混合精度权重文件。
  --bin models/yolo11n_hit_uav_ncnn_int8_mixed_head_fp32/model.ncnn.bin
  # 中文说明：读取五个类别名称。
  --classes models/classes.txt
  # 中文说明：读取固定测试图片。
  --image assets/test.jpg
  # 中文说明：把混合精度画框结果保存为独立文件。
  --output outputs/result_int8_mixed_head.jpg
  # 中文说明：模型输入边长保持 640。
  --imgsz 640
  # 中文说明：置信度阈值与 FP32 基线一致。
  --conf 0.25
  # 中文说明：NMS IoU 阈值与 FP32 基线一致。
  --iou 0.45
  # 中文说明：使用四个 CPU 线程，保证性能对比公平。
  --threads 4
)
# 中文说明：展开数组并执行混合精度推理。
"${command[@]}"
