#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

index=1
for image in assets/validation_images/*.jpg; do
  output=$(printf "outputs/result_%03d.jpg" "$index")
  echo "[$index] $image -> $output"
  validation_command=(
    ./build/yolo_ncnn
    --param models/yolo11n_hit_uav_ncnn/model.ncnn.param
    --bin models/yolo11n_hit_uav_ncnn/model.ncnn.bin
    --classes models/classes.txt
    --image "$image"
    --output "$output"
    --imgsz 640
    --conf 0.25
    --iou 0.45
    --threads 4
  )
  "${validation_command[@]}"
  index=$((index + 1))
done

echo "Generated $((index - 1)) validation outputs."
