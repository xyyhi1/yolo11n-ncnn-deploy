#!/usr/bin/env python3
# 文件作用：在完整 HIT-UAV 验证集上分别评估 FP32 与 INT8 NCNN 模型，并保存精度对比表。

# 中文说明：导入 CSV 模块，用于写出结构化指标。
import csv
# 中文说明：导入环境变量模块，让数据集路径可以由使用者配置。
import os
# 中文说明：导入文件复制函数，用于构造 Ultralytics 能识别的 NCNN 模型目录。
from shutil import copy2
# 中文说明：导入 Path，统一管理项目和模型路径。
from pathlib import Path

# 中文说明：导入 Ultralytics YOLO 接口，复用其完整验证集指标实现。
from ultralytics import YOLO


# 中文说明：取得项目根目录。
ROOT = Path(__file__).resolve().parents[1]
# 中文说明：优先读取环境变量，未设置时使用当前目录下的 HIT-UAV.yaml。
DATA_YAML = Path(os.environ.get("HIT_UAV_DATA_YAML", "HIT-UAV.yaml")).expanduser()
# 中文说明：指定原始 PNNX 导出目录，主要用于复制 metadata.yaml。
EXPORT_DIR = ROOT / "models/source/yolo11n_hit_uav_ncnn_model"
# 中文说明：定义两种精度模型的真实文件来源。
MODEL_SOURCES = {
    # 中文说明：FP32 模型作为量化前基线。
    "fp32": ROOT / "models/yolo11n_hit_uav_ncnn",
    # 中文说明：INT8 模型用于观察量化后的精度变化。
    "int8": ROOT / "models/yolo11n_hit_uav_ncnn_int8",
    # 中文说明：只让七个最终输出卷积保留 FP32，用于测试最小回退是否恢复精度。
    "int8_mixed_output_fp32": ROOT / "models/yolo11n_hit_uav_ncnn_int8_mixed_output_fp32",
    # 中文说明：让整个检测头保留 FP32，用于判断精度损失是否主要来自检测头。
    "int8_mixed_head_fp32": ROOT / "models/yolo11n_hit_uav_ncnn_int8_mixed_head_fp32",
}


# 中文说明：定义模型目录准备函数，因为 Ultralytics 依赖 `_ncnn_model` 后缀识别格式。
def prepare_ultralytics_directory(name: str, source: Path) -> Path:
    # 中文说明：为当前精度模型创建独立评估目录。
    destination = ROOT / "models/eval" / f"yolo11n_hit_uav_{name}_ncnn_model"
    # 中文说明：递归创建目录，已存在时继续覆盖模型文件。
    destination.mkdir(parents=True, exist_ok=True)
    # 中文说明：复制模型结构并保持 Ultralytics 约定的文件名。
    copy2(source / "model.ncnn.param", destination / "model.ncnn.param")
    # 中文说明：复制对应的 FP32 或 INT8 权重。
    copy2(source / "model.ncnn.bin", destination / "model.ncnn.bin")
    # 中文说明：复制类别、输入尺寸等导出元数据。
    copy2(EXPORT_DIR / "metadata.yaml", destination / "metadata.yaml")
    # 中文说明：返回可以直接传给 YOLO 的目录。
    return destination


# 中文说明：定义完整验证集评估函数。
def evaluate(name: str, model_directory: Path) -> dict:
    # 中文说明：显式指定 detect 任务，避免框架根据目录猜测任务类型。
    model = YOLO(model_directory, task="detect")
    # 中文说明：在全部 290 张验证图片上计算检测指标。
    metrics = model.val(
        # 中文说明：传入数据集路径和五个类别定义。
        data=str(DATA_YAML),
        # 中文说明：明确评估 validation split。
        split="val",
        # 中文说明：与部署程序保持 640 输入尺寸一致。
        imgsz=640,
        # 中文说明：NCNN AutoBackend 当前按单图执行，因此 batch 固定为 1。
        batch=1,
        # 中文说明：不启动额外数据加载进程，减少 WSL 环境变量影响。
        workers=0,
        # 中文说明：关闭绘图，评估只保留数值指标。
        plots=False,
        # 中文说明：把验证日志和中间结果集中放到项目 results 目录。
        project=str(ROOT / "results/validation"),
        # 中文说明：FP32 和 INT8 使用不同子目录。
        name=name,
        # 中文说明：重复运行时覆盖同名目录，不自动生成 fp322 等目录。
        exist_ok=True,
        # 中文说明：显示验证进度和最终指标。
        verbose=True,
    )
    # 中文说明：提取简历和对比表需要的主要检测指标。
    return {
        # 中文说明：记录模型精度类型。
        "precision": name,
        # 中文说明：记录验证集图片数量，当前固定为 290。
        "validation_images": 290,
        # 中文说明：记录所有类别平均 Precision。
        "precision_mean": float(metrics.box.mp),
        # 中文说明：记录所有类别平均 Recall。
        "recall_mean": float(metrics.box.mr),
        # 中文说明：记录 IoU=0.5 时的 mAP。
        "map50": float(metrics.box.map50),
        # 中文说明：记录 IoU 0.5 到 0.95 的 COCO 风格 mAP。
        "map50_95": float(metrics.box.map),
    }


# 中文说明：定义脚本主流程。
def main():
    # 中文说明：验证数据集配置文件存在，避免跑到一半才失败。
    if not DATA_YAML.is_file():
        # 中文说明：缺少配置文件时抛出明确异常。
        raise FileNotFoundError(DATA_YAML)
    # 中文说明：保存两种模型的验证指标。
    rows = []
    # 中文说明：依次评估 FP32 基线和 INT8 模型。
    for name, source in MODEL_SOURCES.items():
        # 中文说明：检查当前模型结构文件存在。
        if not (source / "model.ncnn.param").is_file():
            # 中文说明：缺少结构文件时停止并报告路径。
            raise FileNotFoundError(source / "model.ncnn.param")
        # 中文说明：检查当前模型权重文件存在。
        if not (source / "model.ncnn.bin").is_file():
            # 中文说明：缺少权重文件时停止并报告路径。
            raise FileNotFoundError(source / "model.ncnn.bin")
        # 中文说明：创建符合 Ultralytics 命名约定的评估目录。
        model_directory = prepare_ultralytics_directory(name, source)
        # 中文说明：执行验证并保存当前模型指标。
        row = evaluate(name, model_directory)
        # 中文说明：把当前结果追加到最终表格。
        rows.append(row)
        # 中文说明：实时打印结果，便于发现明显精度下降。
        print(row)

    # 中文说明：定义量化精度对比 CSV 路径。
    output = ROOT / "results/quantization_accuracy.csv"
    # 中文说明：以 UTF-8 编码创建 CSV 文件。
    with output.open("w", newline="", encoding="utf-8") as handle:
        # 中文说明：使用字典键作为 CSV 表头。
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        # 中文说明：写入第一行表头。
        writer.writeheader()
        # 中文说明：写入 FP32、全 INT8 和两种混合精度实验指标。
        writer.writerows(rows)
    # 中文说明：打印最终对比表位置。
    print(output)


# 中文说明：直接运行脚本时进入主流程。
if __name__ == "__main__":
    # 中文说明：调用主函数。
    main()
