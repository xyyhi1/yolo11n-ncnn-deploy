#!/usr/bin/env bash
# 文件作用：运行自动搜索选出的最优NCNN混合精度模型。
# 中文说明：启用Shell严格模式，任一错误都会停止脚本。
set -euo pipefail

# 中文说明：取得项目根目录绝对路径。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：切换到项目根目录，保证相对路径稳定。
cd "$ROOT"

# 中文说明：用数组保存完整推理命令，便于逐项阅读。
command=(
  # 中文说明：调用独立C++推理程序。
  ./build/yolo_ncnn
  # 中文说明：使用自动搜索导出的网络结构。
  --param models/yolo11n_hit_uav_ncnn_auto_mixed/model.ncnn.param
  # 中文说明：使用自动搜索导出的混合精度权重。
  --bin models/yolo11n_hit_uav_ncnn_auto_mixed/model.ncnn.bin
  # 中文说明：读取五个类别名称。
  --classes models/classes.txt
  # 中文说明：使用固定测试图片。
  --image assets/test.jpg
  # 中文说明：保存自动混合精度画框结果。
  --output outputs/result_auto_mixed.jpg
  # 中文说明：模型输入边长为640。
  --imgsz 640
  # 中文说明：置信度阈值保持0.25。
  --conf 0.25
  # 中文说明：NMS IoU阈值保持0.45。
  --iou 0.45
  # 中文说明：使用四个CPU线程。
  --threads 4
)
# 中文说明：展开命令数组并执行推理。
"${command[@]}"
