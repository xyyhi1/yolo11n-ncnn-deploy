#!/usr/bin/env bash
# 运行最终推荐的“骨干与颈部 INT8、检测头 FP32”混合精度模型。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command=(
  ./build/yolo_ncnn
  --param models/yolo11n_hit_uav_ncnn_int8_mixed_head_fp32/model.ncnn.param
  --bin models/yolo11n_hit_uav_ncnn_int8_mixed_head_fp32/model.ncnn.bin
  --classes models/classes.txt
  --image assets/test.jpg
  --output outputs/result_int8_mixed_head.jpg
  --imgsz 640
  --conf 0.25
  --iou 0.45
  --threads 4
)
"${command[@]}"
