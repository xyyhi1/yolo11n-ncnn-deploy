#!/usr/bin/env python3
# 生成不拉伸内容的竖图和正方形测试样例。
"""在不拉伸原图内容的前提下，创建竖图和正方形几何测试样例。"""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/validation_images/1_100_30_0_05095.jpg"
OUTPUT_DIR = ROOT / "assets/validation_images"


# 定义 Python 函数 fit_on_canvas。
def fit_on_canvas(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


# 定义 Python 函数 main。
def main():
    image = cv2.imread(str(SOURCE))
    if image is None:
        raise RuntimeError(f"cannot read {SOURCE}")
    cases = {
        "shape_portrait_512x640.jpg": fit_on_canvas(image, 512, 640),
        "shape_square_640x640.jpg": fit_on_canvas(image, 640, 640),
    }
    for name, output in cases.items():
        path = OUTPUT_DIR / name
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"cannot write {path}")
        print(f"{path}: {output.shape[1]}x{output.shape[0]}")


if __name__ == "__main__":
    main()
