#!/usr/bin/env python3
# 在完整 HIT-UAV 验证集上分别评估 FP32 与 INT8 NCNN 模型，并保存精度对比表。

import csv
import os
from shutil import copy2
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = Path(os.environ.get("HIT_UAV_DATA_YAML", "HIT-UAV.yaml")).expanduser()
EXPORT_DIR = ROOT / "models/source/yolo11n_hit_uav_ncnn_model"
MODEL_SOURCES = {
    "fp32": ROOT / "models/yolo11n_hit_uav_ncnn",
    "int8": ROOT / "models/yolo11n_hit_uav_ncnn_int8",
    "int8_mixed_output_fp32": ROOT / "models/yolo11n_hit_uav_ncnn_int8_mixed_output_fp32",
    "int8_mixed_head_fp32": ROOT / "models/yolo11n_hit_uav_ncnn_int8_mixed_head_fp32",
}


# 定义模型目录准备函数，因为 Ultralytics 依赖 `_ncnn_model` 后缀识别格式。
def prepare_ultralytics_directory(name: str, source: Path) -> Path:
    destination = ROOT / "models/eval" / f"yolo11n_hit_uav_{name}_ncnn_model"
    destination.mkdir(parents=True, exist_ok=True)
    copy2(source / "model.ncnn.param", destination / "model.ncnn.param")
    copy2(source / "model.ncnn.bin", destination / "model.ncnn.bin")
    copy2(EXPORT_DIR / "metadata.yaml", destination / "metadata.yaml")
    return destination


# 定义完整验证集评估函数。
def evaluate(name: str, model_directory: Path) -> dict:
    model = YOLO(model_directory, task="detect")
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=1,
        workers=0,
        plots=False,
        project=str(ROOT / "results/validation"),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    return {
        "precision": name,
        "validation_images": 290,
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


# 定义脚本主流程。
def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(DATA_YAML)
    rows = []
    for name, source in MODEL_SOURCES.items():
        if not (source / "model.ncnn.param").is_file():
            raise FileNotFoundError(source / "model.ncnn.param")
        if not (source / "model.ncnn.bin").is_file():
            raise FileNotFoundError(source / "model.ncnn.bin")
        model_directory = prepare_ultralytics_directory(name, source)
        row = evaluate(name, model_directory)
        rows.append(row)
        print(row)

    output = ROOT / "results/quantization_accuracy.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
