#!/usr/bin/env python3
# 中文说明：指定运行该脚本所使用的解释器。
# 文件作用：逐图比较 PyTorch 与 NCNN 的类别、框 IoU 和置信度差异。
# 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
# 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
"""在验证图片上比较 PyTorch 与导出的 NCNN 模型预测结果。"""

# 中文说明：导入 Python 模块 csv。
import csv
# 中文说明：从 Python 模块 pathlib 导入后续需要的对象。
from pathlib import Path

# 中文说明：导入 Python 模块 numpy as np。
import numpy as np
# 中文说明：从 Python 模块 ultralytics 导入后续需要的对象。
from ultralytics import YOLO


# 中文说明：更新变量或对象 ROOT 的值。
ROOT = Path(__file__).resolve().parents[1]
# 中文说明：更新变量或对象 IMAGES 的值。
IMAGES = sorted((ROOT / "assets/validation_images").glob("*.jpg"))


# 中文说明：定义 Python 函数 iou。
def iou(a: np.ndarray, b: np.ndarray) -> float:
    # 中文说明：更新变量或对象 left_top 的值。
    left_top = np.maximum(a[:2], b[:2])
    # 中文说明：更新变量或对象 right_bottom 的值。
    right_bottom = np.minimum(a[2:], b[2:])
    # 中文说明：更新变量或对象 intersection 的值。
    intersection = np.prod(np.maximum(0.0, right_bottom - left_top))
    # 中文说明：更新变量或对象 area_a 的值。
    area_a = np.prod(np.maximum(0.0, a[2:] - a[:2]))
    # 中文说明：更新变量或对象 area_b 的值。
    area_b = np.prod(np.maximum(0.0, b[2:] - b[:2]))
    # 中文说明：更新变量或对象 union 的值。
    union = area_a + area_b - intersection
    # 中文说明：结束当前函数并返回结果：float(intersection / union) if union > 0 else 0.0
    return float(intersection / union) if union > 0 else 0.0


# 中文说明：定义 Python 函数 unpack。
def unpack(result):
    # 中文说明：更新变量或对象 boxes 的值。
    boxes = result.boxes
    # 中文说明：判断条件：boxes is None or len(boxes) == 0:
    if boxes is None or len(boxes) == 0:
        # 中文说明：结束当前函数并返回结果：np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
    # 中文说明：结束当前函数并返回结果：(
    return (
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        boxes.xyxy.cpu().numpy(),
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        boxes.conf.cpu().numpy(),
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        boxes.cls.cpu().numpy().astype(np.int32),
    # 中文说明：结束当前跨行的 CMake 命令或函数调用。
    )


# 中文说明：定义 Python 函数 compare。
def compare(pt_result, ncnn_result):
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    pt_xyxy, pt_conf, pt_cls = unpack(pt_result)
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    nc_xyxy, nc_conf, nc_cls = unpack(ncnn_result)
    # 中文说明：更新变量或对象 used 的值。
    used = set()
    # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    matched_ious, confidence_deltas = [], []
    # 中文说明：按顺序循环处理：p_index, p_box in enumerate(pt_xyxy):
    for p_index, p_box in enumerate(pt_xyxy):
        # 中文说明：更新变量或对象 candidates 的值。
        candidates = [
            # 中文说明：调用函数或构造对象，并传入括号中的参数。
            (iou(p_box, n_box), n_index)
            # 中文说明：按顺序循环处理：n_index, n_box in enumerate(nc_xyxy)
            for n_index, n_box in enumerate(nc_xyxy)
            # 中文说明：判断条件：n_index not in used and nc_cls[n_index] == pt_cls[p_index]
            if n_index not in used and nc_cls[n_index] == pt_cls[p_index]
        # 中文说明：结束当前跨行的列表、表达式或函数调用。
        ]
        # 中文说明：判断条件：not candidates:
        if not candidates:
            # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
            continue
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        best_iou, best_index = max(candidates)
        # 中文说明：判断条件：best_iou >= 0.5:
        if best_iou >= 0.5:
            # 中文说明：调用函数或构造对象，并传入括号中的参数。
            used.add(best_index)
            # 中文说明：调用函数或构造对象，并传入括号中的参数。
            matched_ious.append(best_iou)
            # 中文说明：调用函数或构造对象，并传入括号中的参数。
            confidence_deltas.append(abs(float(pt_conf[p_index] - nc_conf[best_index])))
    # 中文说明：结束当前函数并返回结果：{
    return {
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "pytorch_detections": len(pt_xyxy),
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "ncnn_detections": len(nc_xyxy),
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "matched_iou_ge_0_5": len(matched_ious),
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "mean_matched_iou": np.mean(matched_ious) if matched_ious else 1.0,
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        "mean_abs_conf_delta": np.mean(confidence_deltas) if confidence_deltas else 0.0,
    # 中文说明：结束当前函数、类型或代码块。
    }


# 中文说明：定义 Python 函数 main。
def main():
    # 中文说明：判断条件：len(IMAGES) < 10:
    if len(IMAGES) < 10:
        # 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        raise RuntimeError(f"expected at least 10 images, found {len(IMAGES)}")
    # 中文说明：更新变量或对象 pt_model 的值。
    pt_model = YOLO(ROOT / "models/source/yolo11n_hit_uav.pt", task="detect")
    # Ultralytics 依靠 `_ncnn_model` 后缀识别 NCNN 模型目录。
    # 中文说明：更新变量或对象 ncnn_model 的值。
    ncnn_model = YOLO(
        # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        ROOT / "models/source/yolo11n_hit_uav_ncnn_model", task="detect"
    # 中文说明：结束当前跨行的 CMake 命令或函数调用。
    )
    # 中文说明：更新变量或对象 common 的值。
    common = dict(imgsz=640, conf=0.25, iou=0.45, verbose=False)
    # 当前 Ultralytics 版本的 NCNN AutoBackend 每次只能处理一张图片。
    # 中文说明：更新变量或对象 pt_results 的值。
    pt_results = [pt_model(str(path), **common)[0] for path in IMAGES]
    # 中文说明：更新变量或对象 ncnn_results 的值。
    ncnn_results = [ncnn_model(str(path), **common)[0] for path in IMAGES]

    # 中文说明：更新变量或对象 rows 的值。
    rows = []
    # 中文说明：按顺序循环处理：path, pt_result, ncnn_result in zip(IMAGES, pt_results, ncnn_results):
    for path, pt_result, ncnn_result in zip(IMAGES, pt_results, ncnn_results):
        # 中文说明：更新变量或对象 row 的值。
        row = {"image": path.name, **compare(pt_result, ncnn_result)}
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        rows.append(row)
        # 中文说明：向终端输出进度或结果，方便观察程序运行状态。
        print(row)

    # 中文说明：更新变量或对象 output 的值。
    output = ROOT / "results/backend_comparison.csv"
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    with output.open("w", newline="", encoding="utf-8") as handle:
        # 中文说明：更新变量或对象 writer 的值。
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        writer.writeheader()
        # 中文说明：调用函数或构造对象，并传入括号中的参数。
        writer.writerows(rows)
    # 中文说明：向终端输出进度或结果，方便观察程序运行状态。
    print(output)


# 中文说明：只有直接运行该脚本时才进入主函数。
if __name__ == "__main__":
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    main()
