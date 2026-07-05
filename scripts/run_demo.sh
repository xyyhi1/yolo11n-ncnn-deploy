#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

demo_command=(
  ./build/yolo_ncnn
  --param models/yolo11n_hit_uav_ncnn/model.ncnn.param
  --bin models/yolo11n_hit_uav_ncnn/model.ncnn.bin
  --classes models/classes.txt
  --image assets/test.jpg
  --output outputs/result.jpg
  --imgsz 640
  --conf 0.25
  --iou 0.45
  --threads 4
)
"${demo_command[@]}"
