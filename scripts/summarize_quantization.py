#!/usr/bin/env python3
# 文件作用：汇总四种精度模型的文件大小、验证集精度和 100 次 CPU 性能数据。

# 中文说明：导入 CSV 读写模块。
import csv
# 中文说明：导入 Path 处理项目文件。
from pathlib import Path

# 中文说明：导入 NumPy 计算平均值、P50 和 P95。
import numpy as np


# 中文说明：取得项目根目录。
ROOT = Path(__file__).resolve().parents[1]
# 中文说明：定义四种模型和对应 Benchmark 文件。
PROFILES = {
    # 中文说明：FP32 是未量化基线。
    "fp32": ("yolo11n_hit_uav_ncnn", "benchmark_ncnn_fp32_cpu_rerun.csv"),
    # 中文说明：int8 是全部可量化卷积都转换的模型。
    "int8": ("yolo11n_hit_uav_ncnn_int8", "benchmark_ncnn_int8_cpu.csv"),
    # 中文说明：该模型只让最终输出卷积回退 FP32。
    "int8_mixed_output_fp32": ("yolo11n_hit_uav_ncnn_int8_mixed_output_fp32", "benchmark_ncnn_int8_mixed_output_cpu.csv"),
    # 中文说明：该模型让完整检测头回退 FP32，是当前推荐方案。
    "int8_mixed_head_fp32": ("yolo11n_hit_uav_ncnn_int8_mixed_head_fp32", "benchmark_ncnn_int8_mixed_head_cpu.csv"),
}


# 中文说明：读取完整验证集精度 CSV，并按模型名称建立字典。
def load_accuracy():
    # 中文说明：打开量化精度结果文件。
    with (ROOT / "results/quantization_accuracy.csv").open(encoding="utf-8") as handle:
        # 中文说明：把每行转为字典，并使用 precision 字段作为索引。
        return {row["precision"]: row for row in csv.DictReader(handle)}


# 中文说明：读取单个模型的 100 次性能数据并计算统计量。
def load_benchmark(filename: str):
    # 中文说明：打开 C++ 程序生成的逐次计时 CSV。
    with (ROOT / "results" / filename).open(encoding="utf-8") as handle:
        # 中文说明：把全部正式运行记录读取到内存。
        rows = list(csv.DictReader(handle))
    # 中文说明：提取前处理耗时数组。
    preprocess = np.array([float(row["preprocess_ms"]) for row in rows])
    # 中文说明：提取 NCNN 推理耗时数组。
    inference = np.array([float(row["inference_ms"]) for row in rows])
    # 中文说明：提取后处理耗时数组。
    postprocess = np.array([float(row["postprocess_ms"]) for row in rows])
    # 中文说明：提取端到端耗时数组。
    end_to_end = np.array([float(row["end_to_end_ms"]) for row in rows])
    # 中文说明：计算并返回统一性能指标。
    return {
        # 中文说明：记录正式运行次数。
        "benchmark_runs": len(rows),
        # 中文说明：计算平均前处理耗时。
        "preprocess_mean_ms": float(preprocess.mean()),
        # 中文说明：计算平均模型推理耗时。
        "inference_mean_ms": float(inference.mean()),
        # 中文说明：计算平均后处理耗时。
        "postprocess_mean_ms": float(postprocess.mean()),
        # 中文说明：计算平均端到端耗时。
        "end_to_end_mean_ms": float(end_to_end.mean()),
        # 中文说明：计算端到端中位数。
        "end_to_end_p50_ms": float(np.percentile(end_to_end, 50)),
        # 中文说明：计算端到端慢尾 P95。
        "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
        # 中文说明：通过 1000 除以平均毫秒得到 FPS。
        "fps": float(1000.0 / end_to_end.mean()),
    }


# 中文说明：定义汇总脚本主流程。
def main():
    # 中文说明：读取四种模型的验证集精度。
    accuracy = load_accuracy()
    # 中文说明：读取 FP32 权重字节数，作为压缩率分母。
    fp32_size = (ROOT / "models/yolo11n_hit_uav_ncnn/model.ncnn.bin").stat().st_size
    # 中文说明：保存最终四行汇总结果。
    summary_rows = []
    # 中文说明：依次处理四种精度方案。
    for name, (model_directory, benchmark_file) in PROFILES.items():
        # 中文说明：读取当前权重文件字节数。
        model_size = (ROOT / "models" / model_directory / "model.ncnn.bin").stat().st_size
        # 中文说明：读取当前模型性能统计量。
        benchmark = load_benchmark(benchmark_file)
        # 中文说明：读取当前模型完整验证集指标。
        metric = accuracy[name]
        # 中文说明：组合模型大小、精度和性能为一行。
        summary_rows.append({
            # 中文说明：记录方案名称。
            "profile": name,
            # 中文说明：记录原始字节数，方便严格复核。
            "model_bin_bytes": model_size,
            # 中文说明：换算为 MiB。
            "model_bin_mib": model_size / 1024 / 1024,
            # 中文说明：计算相对 FP32 权重减少百分比。
            "size_reduction_percent": (1.0 - model_size / fp32_size) * 100.0,
            # 中文说明：记录验证集 Precision。
            "precision_mean": float(metric["precision_mean"]),
            # 中文说明：记录验证集 Recall。
            "recall_mean": float(metric["recall_mean"]),
            # 中文说明：记录 mAP50。
            "map50": float(metric["map50"]),
            # 中文说明：记录 mAP50-95。
            "map50_95": float(metric["map50_95"]),
            # 中文说明：合并当前性能指标。
            **benchmark,
        })

    # 中文说明：指定最终汇总 CSV 路径。
    output = ROOT / "results/quantization_summary.csv"
    # 中文说明：创建并写入汇总文件。
    with output.open("w", newline="", encoding="utf-8") as handle:
        # 中文说明：使用第一行字典键作为表头。
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        # 中文说明：写入 CSV 表头。
        writer.writeheader()
        # 中文说明：写入四种方案的数据。
        writer.writerows(summary_rows)
    # 中文说明：打印便于阅读的关键指标。
    for row in summary_rows:
        # 中文说明：输出方案、大小、精度、延迟和 FPS。
        print(
            f"{row['profile']}: {row['model_bin_mib']:.2f} MiB, "
            f"mAP50={row['map50']:.4f}, mAP50-95={row['map50_95']:.4f}, "
            f"E2E={row['end_to_end_mean_ms']:.2f} ms, FPS={row['fps']:.2f}"
        )
    # 中文说明：打印汇总 CSV 的绝对路径。
    print(output)


# 中文说明：直接运行脚本时进入主流程。
if __name__ == "__main__":
    # 中文说明：调用主函数。
    main()
