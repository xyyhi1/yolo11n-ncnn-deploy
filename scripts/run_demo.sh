#!/usr/bin/env bash
# 中文说明：该脚本用固定参数运行一张图片，作为最简单的演示入口。
# 中文说明：启用 Shell 严格模式，任意一步失败都会停止脚本。
set -euo pipefail

# 中文说明：计算项目根目录的绝对路径。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：切换到项目根目录，使后续相对路径保持稳定。
cd "$ROOT"

# 中文说明：把程序和所有参数放入数组，便于逐项阅读。
demo_command=(
  # 中文说明：指定要执行的 C++ 推理程序。
  ./build/yolo_ncnn
  # 中文说明：指定 NCNN 网络结构文件。
  --param models/yolo11n_hit_uav_ncnn/model.ncnn.param
  # 中文说明：指定 NCNN 二进制权重文件。
  --bin models/yolo11n_hit_uav_ncnn/model.ncnn.bin
  # 中文说明：指定类别名称文件。
  --classes models/classes.txt
  # 中文说明：指定输入测试图片。
  --image assets/test.jpg
  # 中文说明：指定画框结果的保存位置。
  --output outputs/result.jpg
  # 中文说明：指定模型输入边长为 640。
  --imgsz 640
  # 中文说明：只保留置信度不低于 0.25 的候选框。
  --conf 0.25
  # 中文说明：NMS 使用 0.45 作为 IoU 抑制阈值。
  --iou 0.45
  # 中文说明：NCNN CPU 推理使用四个线程。
  --threads 4
)
# 中文说明：展开数组并执行完整推理命令。
"${demo_command[@]}"
