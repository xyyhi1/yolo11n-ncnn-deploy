#!/usr/bin/env python3
# 使用 OpenCV 绘制项目推理流程架构图。
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
output = root / "docs/project_architecture.png"

canvas = np.full((420, 1500, 3), 255, dtype=np.uint8)
steps = [
    "YOLO11n\nbest.pt",
    "Ultralytics\n+ PNNX",
    "NCNN\n.param + .bin",
    "Letterbox\nRGB / 255",
    "NCNN CPU\nExtractor",
    "Decode + NMS\nCoordinate restore",
    "result.jpg",
]
box_w, box_h, gap, x0, y = 175, 100, 35, 30, 155

for index, text in enumerate(steps):
    x = x0 + index * (box_w + gap)
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (64, 104, 145), 2)
    lines = text.split("\n")
    for line_index, line in enumerate(lines):
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = x + (box_w - size[0]) // 2
        ty = y + 40 + line_index * 28
        cv2.putText(canvas, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    if index + 1 < len(steps):
        start = (x + box_w + 5, y + box_h // 2)
        end = (x + box_w + gap - 5, y + box_h // 2)
        cv2.arrowedLine(canvas, start, end, (64, 104, 145), 2, tipLength=0.25)

cv2.putText(
    canvas,
    "YOLO11n HIT-UAV -> NCNN C++ Deployment Pipeline",
    (350, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (27, 64, 92),
    2,
    cv2.LINE_AA,
)
output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output), canvas)
print(output)
