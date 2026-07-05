#!/usr/bin/env bash
# 中文说明：该脚本批量处理 validation_images 目录中的所有 JPG 图片。
# 中文说明：启用严格模式，避免单张图片失败后仍继续产生不完整结果。
set -euo pipefail

# 中文说明：计算项目根目录的绝对路径。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：切换到项目根目录，统一后续相对路径的基准。
cd "$ROOT"

# 中文说明：结果编号从 1 开始。
index=1
# 中文说明：按文件名顺序遍历所有验证图片。
for image in assets/validation_images/*.jpg; do
  # 中文说明：把当前编号格式化成 result_001.jpg 这样的三位数文件名。
  output=$(printf "outputs/result_%03d.jpg" "$index")
  # 中文说明：打印当前输入图片及其输出路径。
  echo "[$index] $image -> $output"
  # 中文说明：为当前图片构造一次完整的推理命令。
  validation_command=(
    # 中文说明：指定 C++ 推理程序。
    ./build/yolo_ncnn
    # 中文说明：指定 NCNN 网络结构文件。
    --param models/yolo11n_hit_uav_ncnn/model.ncnn.param
    # 中文说明：指定 NCNN 二进制权重文件。
    --bin models/yolo11n_hit_uav_ncnn/model.ncnn.bin
    # 中文说明：指定五个类别名称所在的文本文件。
    --classes models/classes.txt
    # 中文说明：把循环中的当前图片作为输入。
    --image "$image"
    # 中文说明：把编号后的文件名作为输出。
    --output "$output"
    # 中文说明：把模型输入边长设置为 640。
    --imgsz 640
    # 中文说明：把置信度阈值设置为 0.25。
    --conf 0.25
    # 中文说明：把分类 NMS 的 IoU 阈值设置为 0.45。
    --iou 0.45
    # 中文说明：使用四个 CPU 线程执行 NCNN 推理。
    --threads 4
  )
  # 中文说明：执行当前图片的推理命令。
  "${validation_command[@]}"
  # 中文说明：编号加一，为下一张图片生成新的输出文件名。
  index=$((index + 1))
# 中文说明：结束图片遍历循环。
done

# 中文说明：打印实际生成的结果图片数量。
echo "Generated $((index - 1)) validation outputs."
