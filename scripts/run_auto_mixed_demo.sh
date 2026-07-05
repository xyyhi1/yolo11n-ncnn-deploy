#!/usr/bin/env bash
# 运行自动搜索选出的最优NCNN混合精度模型。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command=(
  ./build/yolo_ncnn
  --param models/yolo11n_hit_uav_ncnn_auto_mixed/model.ncnn.param
  --bin models/yolo11n_hit_uav_ncnn_auto_mixed/model.ncnn.bin
  --classes models/classes.txt
  --image assets/test.jpg
  --output outputs/result_auto_mixed.jpg
  --imgsz 640
  --conf 0.25
  --iou 0.45
  --threads 4
)
"${command[@]}"
