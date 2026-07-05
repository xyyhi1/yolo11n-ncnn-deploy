#!/usr/bin/env python3
# 中文说明：指定运行该脚本所使用的解释器。
# 文件作用：生成不拉伸内容的竖图和正方形测试样例。
# 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
# 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
"""在不拉伸原图内容的前提下，创建竖图和正方形几何测试样例。"""

# 中文说明：从 Python 模块 pathlib 导入后续需要的对象。
from pathlib import Path

# 中文说明：导入 Python 模块 cv2。
import cv2
# 中文说明：导入 Python 模块 numpy as np。
import numpy as np


# 中文说明：更新变量或对象 ROOT 的值。
ROOT = Path(__file__).resolve().parents[1]
# 中文说明：更新变量或对象 SOURCE 的值。
SOURCE = ROOT / "assets/validation_images/1_100_30_0_05095.jpg"
# 中文说明：更新变量或对象 OUTPUT_DIR 的值。
OUTPUT_DIR = ROOT / "assets/validation_images"


# 中文说明：定义 Python 函数 fit_on_canvas。
def fit_on_canvas(image, width, height):
    # 中文说明：更新变量或对象 scale 的值。
    scale = min(width / image.shape[1], height / image.shape[0])
    # 中文说明：更新变量或对象 resized 的值。
    resized = cv2.resize(
        # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        image,
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        # 中文说明：更新变量或对象 interpolation 的值。
        interpolation=cv2.INTER_LINEAR,
    # 中文说明：结束当前跨行的 CMake 命令或函数调用。
    )
    # 中文说明：更新变量或对象 canvas 的值。
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    # 中文说明：更新变量或对象 x 的值。
    x = (width - resized.shape[1]) // 2
    # 中文说明：更新变量或对象 y 的值。
    y = (height - resized.shape[0]) // 2
    # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    # 中文说明：结束当前函数并返回结果：canvas
    return canvas


# 中文说明：定义 Python 函数 main。
def main():
    # 中文说明：更新变量或对象 image 的值。
    image = cv2.imread(str(SOURCE))
    # 中文说明：判断条件：image is None:
    if image is None:
        # 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        raise RuntimeError(f"cannot read {SOURCE}")
    # 中文说明：更新变量或对象 cases 的值。
    cases = {
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "shape_portrait_512x640.jpg": fit_on_canvas(image, 512, 640),
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "shape_square_640x640.jpg": fit_on_canvas(image, 640, 640),
    # 中文说明：结束当前函数、类型或代码块。
    }
    # 中文说明：按顺序循环处理：name, output in cases.items():
    for name, output in cases.items():
        # 中文说明：更新变量或对象 path 的值。
        path = OUTPUT_DIR / name
        # 中文说明：判断条件：not cv2.imwrite(str(path), output):
        if not cv2.imwrite(str(path), output):
            # 中文说明：发现非法状态，抛出异常并停止当前处理流程。
            raise RuntimeError(f"cannot write {path}")
        # 中文说明：向终端输出进度或结果，方便观察程序运行状态。
        print(f"{path}: {output.shape[1]}x{output.shape[0]}")


# 中文说明：只有直接运行该脚本时才进入主函数。
if __name__ == "__main__":
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    main()
