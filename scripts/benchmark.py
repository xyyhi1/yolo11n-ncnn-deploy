#!/usr/bin/env python3
# 中文说明：指定运行该脚本所使用的解释器。
# 文件作用：调用 C++ 程序执行 20 次预热和 100 次正式性能测试。
# 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
# 中文说明：从 Python 模块 pathlib 导入后续需要的对象。
from pathlib import Path
# 中文说明：导入 Python 模块 subprocess。
import subprocess

# 中文说明：更新变量或对象 root 的值。
root = Path(__file__).resolve().parents[1]
# 中文说明：更新变量或对象 command 的值。
command = [
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    str(root / "build/yolo_ncnn"),
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    "--param", str(root / "models/yolo11n_hit_uav_ncnn/model.ncnn.param"),
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    "--bin", str(root / "models/yolo11n_hit_uav_ncnn/model.ncnn.bin"),
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    "--classes", str(root / "models/classes.txt"),
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    "--image", str(root / "assets/test.jpg"),
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    "--output", str(root / "outputs/result_benchmark.jpg"),
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "--imgsz", "640",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "--conf", "0.25",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "--iou", "0.45",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "--threads", "4",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "--warmup", "20",
    # 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    "--runs", "100",
    # 中文说明：调用函数或构造对象，并传入括号中的参数。
    "--benchmark-csv", str(root / "results/benchmark_ncnn_cpu.csv"),
# 中文说明：结束当前跨行的列表、表达式或函数调用。
]
# 中文说明：调用函数或构造对象，并传入括号中的参数。
subprocess.run(command, check=True, cwd=root)
