#!/usr/bin/env python3
# 逐图比较 PyTorch 与 NCNN 的类别、框 IoU 和置信度差异。
"""在验证图片上比较 PyTorch 与导出的 NCNN 模型预测结果。"""

import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGES = sorted((ROOT / "assets/validation_images").glob("*.jpg"))


# 定义 Python 函数 iou。
def iou(a: np.ndarray, b: np.ndarray) -> float:
    left_top = np.maximum(a[:2], b[:2])
    right_bottom = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(0.0, right_bottom - left_top))
    area_a = np.prod(np.maximum(0.0, a[2:] - a[:2]))
    area_b = np.prod(np.maximum(0.0, b[2:] - b[:2]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


# 定义 Python 函数 unpack。
def unpack(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
    return (
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(np.int32),
    )


# 定义 Python 函数 compare。
def compare(pt_result, ncnn_result):
    pt_xyxy, pt_conf, pt_cls = unpack(pt_result)
    nc_xyxy, nc_conf, nc_cls = unpack(ncnn_result)
    used = set()
    matched_ious, confidence_deltas = [], []
    for p_index, p_box in enumerate(pt_xyxy):
        candidates = [
            (iou(p_box, n_box), n_index)
            for n_index, n_box in enumerate(nc_xyxy)
            if n_index not in used and nc_cls[n_index] == pt_cls[p_index]
        ]
        if not candidates:
            continue
        best_iou, best_index = max(candidates)
        if best_iou >= 0.5:
            used.add(best_index)
            matched_ious.append(best_iou)
            confidence_deltas.append(abs(float(pt_conf[p_index] - nc_conf[best_index])))
    return {
        "pytorch_detections": len(pt_xyxy),
        "ncnn_detections": len(nc_xyxy),
        "matched_iou_ge_0_5": len(matched_ious),
        "mean_matched_iou": np.mean(matched_ious) if matched_ious else 1.0,
        "mean_abs_conf_delta": np.mean(confidence_deltas) if confidence_deltas else 0.0,
    }


# 定义 Python 函数 main。
def main():
    if len(IMAGES) < 10:
        raise RuntimeError(f"expected at least 10 images, found {len(IMAGES)}")
    pt_model = YOLO(ROOT / "models/source/yolo11n_hit_uav.pt", task="detect")
    # Ultralytics 依靠 `_ncnn_model` 后缀识别 NCNN 模型目录。
    ncnn_model = YOLO(
        ROOT / "models/source/yolo11n_hit_uav_ncnn_model", task="detect"
    )
    common = dict(imgsz=640, conf=0.25, iou=0.45, verbose=False)
    # 当前 Ultralytics 版本的 NCNN AutoBackend 每次只能处理一张图片。
    pt_results = [pt_model(str(path), **common)[0] for path in IMAGES]
    ncnn_results = [ncnn_model(str(path), **common)[0] for path in IMAGES]

    rows = []
    for path, pt_result, ncnn_result in zip(IMAGES, pt_results, ncnn_results):
        row = {"image": path.name, **compare(pt_result, ncnn_result)}
        rows.append(row)
        print(row)

    output = ROOT / "results/backend_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
