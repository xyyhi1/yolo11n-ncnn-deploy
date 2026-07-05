#!/usr/bin/env python3
# 汇总四种精度模型的文件大小、验证集精度和 100 次 CPU 性能数据。

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "fp32": ("yolo11n_hit_uav_ncnn", "benchmark_ncnn_fp32_cpu_rerun.csv"),
    "int8": ("yolo11n_hit_uav_ncnn_int8", "benchmark_ncnn_int8_cpu.csv"),
    "int8_mixed_output_fp32": ("yolo11n_hit_uav_ncnn_int8_mixed_output_fp32", "benchmark_ncnn_int8_mixed_output_cpu.csv"),
    "int8_mixed_head_fp32": ("yolo11n_hit_uav_ncnn_int8_mixed_head_fp32", "benchmark_ncnn_int8_mixed_head_cpu.csv"),
}


# 读取完整验证集精度 CSV，并按模型名称建立字典。
def load_accuracy():
    with (ROOT / "results/quantization_accuracy.csv").open(encoding="utf-8") as handle:
        return {row["precision"]: row for row in csv.DictReader(handle)}


# 读取单个模型的 100 次性能数据并计算统计量。
def load_benchmark(filename: str):
    with (ROOT / "results" / filename).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    preprocess = np.array([float(row["preprocess_ms"]) for row in rows])
    inference = np.array([float(row["inference_ms"]) for row in rows])
    postprocess = np.array([float(row["postprocess_ms"]) for row in rows])
    end_to_end = np.array([float(row["end_to_end_ms"]) for row in rows])
    return {
        "benchmark_runs": len(rows),
        "preprocess_mean_ms": float(preprocess.mean()),
        "inference_mean_ms": float(inference.mean()),
        "postprocess_mean_ms": float(postprocess.mean()),
        "end_to_end_mean_ms": float(end_to_end.mean()),
        "end_to_end_p50_ms": float(np.percentile(end_to_end, 50)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
        "fps": float(1000.0 / end_to_end.mean()),
    }


# 定义汇总脚本主流程。
def main():
    accuracy = load_accuracy()
    fp32_size = (ROOT / "models/yolo11n_hit_uav_ncnn/model.ncnn.bin").stat().st_size
    summary_rows = []
    for name, (model_directory, benchmark_file) in PROFILES.items():
        model_size = (ROOT / "models" / model_directory / "model.ncnn.bin").stat().st_size
        benchmark = load_benchmark(benchmark_file)
        metric = accuracy[name]
        summary_rows.append({
            "profile": name,
            "model_bin_bytes": model_size,
            "model_bin_mib": model_size / 1024 / 1024,
            "size_reduction_percent": (1.0 - model_size / fp32_size) * 100.0,
            "precision_mean": float(metric["precision_mean"]),
            "recall_mean": float(metric["recall_mean"]),
            "map50": float(metric["map50"]),
            "map50_95": float(metric["map50_95"]),
            **benchmark,
        })

    output = ROOT / "results/quantization_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(
            f"{row['profile']}: {row['model_bin_mib']:.2f} MiB, "
            f"mAP50={row['map50']:.4f}, mAP50-95={row['map50_95']:.4f}, "
            f"E2E={row['end_to_end_mean_ms']:.2f} ms, FPS={row['fps']:.2f}"
        )
    print(output)


if __name__ == "__main__":
    main()
