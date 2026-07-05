#!/usr/bin/env python3
# 调用 C++ 程序执行 20 次预热和 100 次正式性能测试。
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
command = [
    str(root / "build/yolo_ncnn"),
    "--param", str(root / "models/yolo11n_hit_uav_ncnn/model.ncnn.param"),
    "--bin", str(root / "models/yolo11n_hit_uav_ncnn/model.ncnn.bin"),
    "--classes", str(root / "models/classes.txt"),
    "--image", str(root / "assets/test.jpg"),
    "--output", str(root / "outputs/result_benchmark.jpg"),
    "--imgsz", "640",
    "--conf", "0.25",
    "--iou", "0.45",
    "--threads", "4",
    "--warmup", "20",
    "--runs", "100",
    "--benchmark-csv", str(root / "results/benchmark_ncnn_cpu.csv"),
]
subprocess.run(command, check=True, cwd=root)
