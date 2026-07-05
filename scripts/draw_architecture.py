#!/usr/bin/env python3
# 中文说明：指定运行该脚本所使用的解释器。
# 文件作用：使用 OpenCV 绘制项目推理流程架构图。
# 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
# 中文说明：从 Python 模块 pathlib 导入后续需要的对象。
from pathlib import Path
# 中文说明：导入 Python 模块 cv2。
import cv2
# 中文说明：导入 Python 模块 numpy as np。
import numpy as np

# 中文说明：更新变量或对象 root 的值。
root = Path(__file__).resolve().parents[1]
# 中文说明：更新变量或对象 output 的值。
output = root / "docs/project_architecture.png"

# 中文说明：更新变量或对象 canvas 的值。
canvas = np.full((420, 1500, 3), 255, dtype=np.uint8)
# 中文说明：更新变量或对象 steps 的值。
steps = [
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "YOLO11n\nbest.pt",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "Ultralytics\n+ PNNX",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "NCNN\n.param + .bin",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "Letterbox\nRGB / 255",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "NCNN CPU\nExtractor",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "Decode + NMS\nCoordinate restore",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "result.jpg",
# 中文说明：结束当前跨行的列表、表达式或函数调用。
]
# 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
box_w, box_h, gap, x0, y = 175, 100, 35, 30, 155

# 中文说明：按顺序循环处理：index, text in enumerate(steps):
for index, text in enumerate(steps):
    # 中文说明：更新变量或对象 x 的值。
    x = x0 + index * (box_w + gap)
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (64, 104, 145), 2)
    # 中文说明：更新变量或对象 lines 的值。
    lines = text.split("\n")
    # 中文说明：按顺序循环处理：line_index, line in enumerate(lines):
    for line_index, line in enumerate(lines):
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # 中文说明：更新变量或对象 tx 的值。
        tx = x + (box_w - size[0]) // 2
        # 中文说明：更新变量或对象 ty 的值。
        ty = y + 40 + line_index * 28
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        cv2.putText(canvas, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    # 中文说明：判断条件：index + 1 < len(steps):
    if index + 1 < len(steps):
        # 中文说明：更新变量或对象 start 的值。
        start = (x + box_w + 5, y + box_h // 2)
        # 中文说明：更新变量或对象 end 的值。
        end = (x + box_w + gap - 5, y + box_h // 2)
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        cv2.arrowedLine(canvas, start, end, (64, 104, 145), 2, tipLength=0.25)

# 中文说明：调用函数或构造对象，并传入括号中的参数。
cv2.putText(
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    canvas,
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "YOLO11n HIT-UAV -> NCNN C++ Deployment Pipeline",
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    (350, 75),
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    cv2.FONT_HERSHEY_SIMPLEX,
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    1.0,
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    (27, 64, 92),
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    2,
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    cv2.LINE_AA,
# 中文说明：结束当前跨行的 CMake 命令或函数调用。
)
# 中文说明：调用函数或构造对象，并传入括号中的参数。
output.parent.mkdir(parents=True, exist_ok=True)
# 中文说明：调用函数或构造对象，并传入括号中的参数。
cv2.imwrite(str(output), canvas)
# 中文说明：向终端输出进度或结果，方便观察程序运行状态。
print(output)
