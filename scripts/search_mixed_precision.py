#!/usr/bin/env python3
# 自动解析 NCNN 图、生成混合精度候选、评估精度与延迟，并选择满足约束的 Pareto 最优策略。

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


CONV_TYPES = {"Convolution", "ConvolutionDepthWise"}


# 使用数据类保存一行 NCNN param 对应的图节点。
@dataclass
class Layer:
    layer_type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: list[str]
    raw: str


# 使用数据类保存一个候选混合精度策略。
@dataclass
class Candidate:
    name: str
    fp32_layers: set[str]
    groups: list[str]


# 解析单个 NCNN 层参数中的整数值。
def attr_int(layer: Layer, key: str, default: int = -1) -> int:
    for token in layer.attrs:
        if token.startswith(key + "="):
            value = token.split("=", 1)[1].split(",", 1)[0]
            try:
                return int(value)
            except ValueError:
                return default
    return default


# 解析 NCNN `.param` 文件并建立图连接关系。
def parse_ncnn_param(param_path: Path):
    lines = [line.strip() for line in param_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"无效 NCNN param：{param_path}")
    layers = []
    for raw in lines[2:]:
        parts = raw.split()
        if len(parts) < 4:
            continue
        bottom_count = int(parts[2])
        top_count = int(parts[3])
        cursor = 4
        inputs = parts[cursor : cursor + bottom_count]
        cursor += bottom_count
        outputs = parts[cursor : cursor + top_count]
        cursor += top_count
        layers.append(Layer(parts[0], parts[1], inputs, outputs, parts[cursor:], raw))
    layer_by_name = {layer.name: layer for layer in layers}
    producer = {blob: layer for layer in layers for blob in layer.outputs}
    consumers = {}
    for layer in layers:
        for blob in layer.inputs:
            consumers.setdefault(blob, []).append(layer)
    return layers, layer_by_name, producer, consumers


# 判断一个 Split 是否是检测头入口边界。
def is_head_boundary_split(layer: Layer, consumers: dict[str, list[Layer]]) -> bool:
    if layer.layer_type != "Split":
        return False
    return any(
        consumer.layer_type in CONV_TYPES
        for blob in layer.outputs
        for consumer in consumers.get(blob, [])
    )


# 从最终输出反向遍历，自动识别检测头中的卷积层。
def discover_head_layers(layers: list[Layer], producer: dict[str, Layer], consumers: dict[str, list[Layer]]):
    final_layers = [layer for layer in layers if "out0" in layer.outputs]
    final_layer = final_layers[-1] if final_layers else layers[-1]
    stack = [final_layer]
    visited = set()
    head_convolutions = set()
    while stack:
        layer = stack.pop()
        if layer.name in visited:
            continue
        visited.add(layer.name)
        if is_head_boundary_split(layer, consumers):
            continue
        if layer.layer_type in CONV_TYPES:
            head_convolutions.add(layer.name)
        for blob in layer.inputs:
            parent = producer.get(blob)
            if parent is not None:
                stack.append(parent)
    return head_convolutions, final_layer


# 识别卷积后直接连接 Reshape 的最终预测卷积。
def discover_terminal_convolutions(head_layers: set[str], layer_by_name: dict[str, Layer], consumers: dict[str, list[Layer]]):
    terminals = []
    for name in sorted(head_layers):
        layer = layer_by_name[name]
        reshape_consumers = [
            consumer
            for blob in layer.outputs
            for consumer in consumers.get(blob, [])
            if consumer.layer_type == "Reshape"
        ]
        if reshape_consumers:
            terminals.append((layer, reshape_consumers[0]))
    return terminals


# 从一个最终预测卷积反向收集同一分支的卷积链。
def trace_branch_convolutions(terminal: Layer, producer: dict[str, Layer], consumers: dict[str, list[Layer]]):
    if attr_int(terminal, "0") == 1:
        return {terminal.name}
    branch = {terminal.name}
    stack = [producer[blob] for blob in terminal.inputs if blob in producer]
    visited = set()
    while stack:
        layer = stack.pop()
        if layer.name in visited:
            continue
        visited.add(layer.name)
        if is_head_boundary_split(layer, consumers):
            continue
        if layer.layer_type in CONV_TYPES:
            branch.add(layer.name)
        for blob in layer.inputs:
            parent = producer.get(blob)
            if parent is not None:
                stack.append(parent)
    return branch


# 根据输出通道、候选点数量和分支链自动建立搜索分组。
def build_search_groups(head_layers, terminals, producer, consumers, num_classes):
    groups = {}
    scale_groups = {}
    classification = set()
    regression = set()
    dfl = set()
    output_layers = set()
    for terminal, reshape in terminals:
        output_channels = attr_int(terminal, "0")
        anchor_count = attr_int(reshape, "0")
        branch = trace_branch_convolutions(terminal, producer, consumers)
        output_layers.add(terminal.name)
        if output_channels == num_classes:
            classification |= branch
            scale_groups.setdefault(anchor_count, set()).update(branch)
            groups[f"classification_{anchor_count}"] = branch
        elif output_channels == 64:
            regression |= branch
            scale_groups.setdefault(anchor_count, set()).update(branch)
            groups[f"regression_{anchor_count}"] = branch
        elif output_channels == 1:
            dfl |= branch
    groups["output"] = output_layers
    groups["classification"] = classification
    groups["regression"] = regression
    groups["dfl"] = dfl
    groups["head"] = set(head_layers)
    sorted_scales = sorted((count for count in scale_groups if count > 0), reverse=True)
    for index, anchor_count in enumerate(sorted_scales):
        scale_name = f"p{index + 3}"
        groups[scale_name] = scale_groups[anchor_count]
    return groups, sorted_scales


# 创建去重后的候选策略列表。
def build_candidates(groups: dict[str, set[str]]):
    candidates = []
    seen = set()

    # 定义内部函数，按分组名称组合FP32层。
    def add(name: str, group_names: list[str]):
        layers = set().union(*(groups[group] for group in group_names if group in groups)) if group_names else set()
        key = frozenset(layers)
        if key in seen:
            return
        seen.add(key)
        candidates.append(Candidate(name, layers, group_names))

    add("full_int8", [])
    add("output_fp32", ["output"])
    add("classification_fp32", ["classification"])
    add("regression_fp32", ["regression"])
    add("dfl_fp32", ["dfl"])
    add("regression_dfl_fp32", ["regression", "dfl"])
    add("classification_dfl_fp32", ["classification", "dfl"])
    add("head_fp32", ["head"])
    for scale in ("p3", "p4", "p5"):
        if scale in groups:
            add(f"{scale}_fp32", [scale])
    for first, second in (("p3", "p4"), ("p3", "p5"), ("p4", "p5")):
        if first in groups and second in groups:
            add(f"{first}_{second}_fp32", [first, second])
    if "p3" in groups:
        add("p3_output_fp32", ["p3", "output"])
    return candidates


# 把图片路径转换成对应YOLO标签路径。
def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


# 读取单张图片标签并计算数量和平均归一化面积。
def label_statistics(image_path: Path):
    label_path = image_to_label_path(image_path)
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return 0, 1.0
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    areas = []
    for line in lines:
        values = line.split()
        if len(values) >= 5:
            areas.append(float(values[3]) * float(values[4]))
    return len(areas), float(np.mean(areas)) if areas else 1.0


# 构造兼顾背景、密集和小目标场景的确定性代理验证集。
def create_proxy_dataset(data_yaml: Path, output_dir: Path, proxy_count: int):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"]).expanduser().resolve()
    val_entry = data["val"]
    val_path = Path(val_entry)
    if not val_path.is_absolute():
        val_path = dataset_root / val_path
    if val_path.is_dir():
        images = sorted(path.resolve() for path in val_path.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    else:
        images = [Path(line.strip()).resolve() for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    proxy_count = min(proxy_count, len(images))
    records = [(path, *label_statistics(path)) for path in images]
    backgrounds = [record for record in records if record[1] == 0]
    foregrounds = [record for record in records if record[1] > 0]
    selected = []
    # 定义内部函数安全追加图片。
    def append_unique(record):
        if record[0] not in selected and len(selected) < proxy_count:
            selected.append(record[0])
    for record in backgrounds[: max(1, proxy_count // 10)]:
        append_unique(record)
    for record in sorted(foregrounds, key=lambda item: (item[2], -item[1]))[: proxy_count // 3]:
        append_unique(record)
    for record in sorted(foregrounds, key=lambda item: (-item[1], item[2]))[: proxy_count // 3]:
        append_unique(record)
    for record in records:
        append_unique(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_list = output_dir / "proxy_images.txt"
    image_list.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
    proxy_data = dict(data)
    proxy_data["val"] = str(image_list.resolve())
    proxy_yaml = output_dir / "proxy_data.yaml"
    proxy_yaml.write_text(yaml.safe_dump(proxy_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return proxy_yaml, selected


# 根据FP32层集合生成新的量化表。
def create_candidate_table(source_table: Path, destination: Path, fp32_layers: set[str]):
    lines = source_table.read_text(encoding="utf-8").splitlines()
    output = []
    commented = set()
    for line in lines:
        if not line.strip():
            output.append(line)
            continue
        key = line.split(maxsplit=1)[0]
        layer_name = key.removesuffix("_param_0")
        if key.endswith("_param_0") and layer_name in fp32_layers:
            output.append("#" + line)
            commented.add(layer_name)
        else:
            output.append(line)
    missing = fp32_layers - commented
    if missing:
        raise RuntimeError(f"校准表缺少以下权重尺度：{sorted(missing)}")
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


# 运行ncnn2int8生成一个候选模型，并支持文件缓存。
def quantize_candidate(candidate, args, artifact_dir, metadata_path):
    model_dir = artifact_dir / "models" / f"{candidate.name}_ncnn_model"
    output_param = model_dir / "model.ncnn.param"
    output_bin = model_dir / "model.ncnn.bin"
    if output_param.is_file() and output_bin.is_file() and not args.force:
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    table_path = artifact_dir / "tables" / f"{candidate.name}.table"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    create_candidate_table(args.table, table_path, candidate.fp32_layers)
    command = [
        str(args.ncnn2int8),
        str(args.param),
        str(args.bin),
        str(output_param),
        str(output_bin),
        str(table_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    (model_dir / "quantize.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    shutil.copy2(metadata_path, model_dir / "metadata.yaml")
    return model_dir


# 准备FP32评估目录，使其同样满足Ultralytics命名约定。
def prepare_fp32_eval_model(args, artifact_dir, metadata_path):
    model_dir = artifact_dir / "models" / "fp32_ncnn_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.param, model_dir / "model.ncnn.param")
    shutil.copy2(args.bin, model_dir / "model.ncnn.bin")
    shutil.copy2(metadata_path, model_dir / "metadata.yaml")
    return model_dir


# 评估单个NCNN模型并把指标缓存为JSON。
def evaluate_model(model_dir, data_yaml, name, scope, artifact_dir, force=False):
    cache_path = artifact_dir / "metrics" / f"{name}_{scope}.json"
    if cache_path.is_file() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_dir, task="detect")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=1,
        workers=0,
        plots=False,
        project=str(artifact_dir / "val_runs"),
        name=f"{name}_{scope}",
        exist_ok=True,
        verbose=False,
    )
    result = {
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# 运行C++程序的20次预热和100次正式Benchmark。
def benchmark_model(model_dir, name, args, artifact_dir):
    cache_path = artifact_dir / "metrics" / f"{name}_benchmark.json"
    if cache_path.is_file() and not args.force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    csv_path = artifact_dir / "benchmarks" / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = artifact_dir / "benchmarks" / f"{name}.jpg"
    command = [
        str(args.executable),
        "--param", str(model_dir / "model.ncnn.param"),
        "--bin", str(model_dir / "model.ncnn.bin"),
        "--classes", str(args.classes),
        "--image", str(args.benchmark_image),
        "--output", str(output_image),
        "--imgsz", "640",
        "--conf", "0.25",
        "--iou", "0.45",
        "--threads", str(args.threads),
        "--warmup", "20",
        "--runs", "100",
        "--benchmark-csv", str(csv_path),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    total = np.array([float(row["end_to_end_ms"]) for row in rows])
    inference = np.array([float(row["inference_ms"]) for row in rows])
    result = {
        "runs": len(rows),
        "inference_mean_ms": float(inference.mean()),
        "end_to_end_mean_ms": float(total.mean()),
        "end_to_end_p50_ms": float(np.percentile(total, 50)),
        "end_to_end_p95_ms": float(np.percentile(total, 95)),
        "fps": float(1000.0 / total.mean()),
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# 判断一条结果是否被其他候选在精度、延迟和大小三个维度支配。
def mark_pareto(rows):
    valid_rows = [row for row in rows if row.get("full_map50_95") is not None and row.get("end_to_end_mean_ms") is not None]
    for row in valid_rows:
        dominated = False
        for other in valid_rows:
            if other is row:
                continue
            no_worse = (
                other["full_map50_95"] >= row["full_map50_95"]
                and other["end_to_end_mean_ms"] <= row["end_to_end_mean_ms"]
                and other["model_mib"] <= row["model_mib"]
            )
            strictly_better = (
                other["full_map50_95"] > row["full_map50_95"]
                or other["end_to_end_mean_ms"] < row["end_to_end_mean_ms"]
                or other["model_mib"] < row["model_mib"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        row["pareto"] = not dominated


# 生成延迟—精度Pareto散点图。
def draw_pareto(rows, output_path):
    valid = [row for row in rows if row.get("full_map50_95") is not None and row.get("end_to_end_mean_ms") is not None]
    plt.figure(figsize=(12, 7))
    valid.sort(key=lambda row: row["end_to_end_mean_ms"])
    for index, row in enumerate(valid):
        color = "#d62728" if row.get("pareto") else "#1f77b4"
        size = 80 + row["model_mib"] * 25
        plt.scatter(
            row["end_to_end_mean_ms"],
            row["full_map50_95"],
            s=size,
            c=color,
            alpha=0.8,
            label=f"{index + 1}: {row['name']}",
        )
        plt.annotate(
            str(index + 1),
            (row["end_to_end_mean_ms"], row["full_map50_95"]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    plt.xlabel("End-to-end latency (ms, lower is better)")
    plt.ylabel("Validation mAP50-95 (higher is better)")
    plt.title("NCNN Mixed-Precision Search Pareto Frontier")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


# 把所有候选结果写入CSV。
def write_results_csv(rows, output_path):
    fieldnames = [
        "name", "groups", "fp32_layer_count", "fp32_layers", "model_mib",
        "proxy_map50", "proxy_map50_95", "full_precision", "full_recall",
        "full_map50", "full_map50_95", "map50_95_drop", "inference_mean_ms",
        "end_to_end_mean_ms", "end_to_end_p50_ms", "end_to_end_p95_ms", "fps", "pareto",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# 生成便于阅读的中文Markdown实验报告。
def write_report(rows, best, baseline, args, output_path, proxy_count):
    evaluated = sorted(
        (row for row in rows if row.get("full_map50_95") is not None),
        key=lambda row: row["full_map50_95"],
        reverse=True,
    )
    lines = [
        "# NCNN 自动混合精度搜索报告",
        "",
        "## 搜索设置",
        "",
        f"- 代理验证集：{proxy_count} 张，覆盖背景、小目标和密集场景。",
        f"- 完整验证集：290 张。",
        f"- 最大允许 mAP50-95 下降：{args.max_map_drop:.4f}。",
        f"- 最大允许模型大小：{args.max_model_mib:.2f} MiB。",
        f"- 性能测试：{args.threads}线程，20次预热，100次正式运行。",
        "",
        "## 完整评估结果",
        "",
        "| 策略 | FP32层数 | 大小/MiB | mAP50-95 | 相对FP32下降 | 延迟/ms | FPS | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluated:
        lines.append(
            f"| {row['name']} | {row['fp32_layer_count']} | {row['model_mib']:.2f} | "
            f"{row['full_map50_95']:.4f} | {row['map50_95_drop']:.4f} | "
            f"{row['end_to_end_mean_ms']:.2f} | {row['fps']:.2f} | "
            f"{'是' if row.get('pareto') else '否'} |"
        )
    lines.extend([
        "",
        "## 自动选择结果",
        "",
        f"最优策略：`{best['name']}`。",
        "",
        f"- FP32层数：{best['fp32_layer_count']}。",
        f"- 模型大小：{best['model_mib']:.2f} MiB。",
        f"- mAP50-95：{best['full_map50_95']:.4f}。",
        f"- 相对FP32下降：{best['map50_95_drop']:.4f}。",
        f"- 端到端延迟：{best['end_to_end_mean_ms']:.2f} ms。",
        f"- FPS：{best['fps']:.2f}。",
        "",
        "选择规则是在满足精度下降和模型大小约束的候选中优先选择端到端延迟最低者；若没有候选满足约束，则选择mAP50-95最高者。",
        "",
        "## FP32基线",
        "",
        f"- mAP50-95：{baseline['full_map50_95']:.4f}。",
        f"- 端到端延迟：{baseline['end_to_end_mean_ms']:.2f} ms。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 定义完整自动搜索主流程。
def run_search(args):
    args.param = args.param.resolve()
    args.bin = args.bin.resolve()
    args.table = args.table.resolve()
    args.data = args.data.resolve()
    args.ncnn2int8 = args.ncnn2int8.resolve()
    args.executable = args.executable.resolve()
    args.classes = args.classes.resolve()
    args.benchmark_image = args.benchmark_image.resolve()
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / "search_artifacts"
    results_dir = root / "results"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "models/source/yolo11n_hit_uav_ncnn_model/metadata.yaml"
    for path in (args.param, args.bin, args.table, args.data, args.ncnn2int8, args.executable, args.classes, args.benchmark_image, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    layers, layer_by_name, producer, consumers = parse_ncnn_param(args.param)
    num_classes = len([line for line in args.classes.read_text(encoding="utf-8").splitlines() if line.strip()])
    head_layers, final_layer = discover_head_layers(layers, producer, consumers)
    terminals = discover_terminal_convolutions(head_layers, layer_by_name, consumers)
    groups, scales = build_search_groups(head_layers, terminals, producer, consumers, num_classes)
    candidates = build_candidates(groups)
    print(f"最终输出层：{final_layer.name}")
    print(f"自动识别检测头卷积：{len(head_layers)} 层")
    print(f"最终预测卷积：{[layer.name for layer, _ in terminals]}")
    print(f"自动识别尺度：{scales}")
    print(f"候选策略：{len(candidates)} 个")
    if args.dry_run:
        for candidate in candidates:
            print(candidate.name, len(candidate.fp32_layers), sorted(candidate.fp32_layers))
        return

    proxy_yaml, proxy_images = create_proxy_dataset(args.data, artifact_dir / "proxy", args.proxy_images)
    fp32_model = prepare_fp32_eval_model(args, artifact_dir, metadata_path)
    fp32_proxy = evaluate_model(fp32_model, proxy_yaml, "fp32", "proxy", artifact_dir, args.force)
    fp32_full = evaluate_model(fp32_model, args.data, "fp32", "full", artifact_dir, args.force)
    fp32_benchmark = benchmark_model(fp32_model, "fp32", args, artifact_dir)
    candidate_models = {}
    proxy_rows = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] 量化并代理评估 {candidate.name}")
        model_dir = quantize_candidate(candidate, args, artifact_dir, metadata_path)
        candidate_models[candidate.name] = model_dir
        proxy_metric = evaluate_model(model_dir, proxy_yaml, candidate.name, "proxy", artifact_dir, args.force)
        proxy_rows.append({
            "candidate": candidate,
            "model_dir": model_dir,
            "metric": proxy_metric,
            "model_mib": (model_dir / "model.ncnn.bin").stat().st_size / 1024 / 1024,
        })

    ranked = sorted(proxy_rows, key=lambda row: (-row["metric"]["map50_95"], row["model_mib"]))
    selected_names = {row["candidate"].name for row in ranked[: args.full_top_k]}
    selected_names |= {"full_int8", "output_fp32", "head_fp32"}
    result_rows = []
    baseline_row = {
        "name": "fp32",
        "groups": "baseline",
        "fp32_layer_count": sum(layer.layer_type in CONV_TYPES for layer in layers),
        "fp32_layers": "all",
        "model_mib": args.bin.stat().st_size / 1024 / 1024,
        "proxy_map50": fp32_proxy["map50"],
        "proxy_map50_95": fp32_proxy["map50_95"],
        "full_precision": fp32_full["precision_mean"],
        "full_recall": fp32_full["recall_mean"],
        "full_map50": fp32_full["map50"],
        "full_map50_95": fp32_full["map50_95"],
        "map50_95_drop": 0.0,
        **fp32_benchmark,
    }
    result_rows.append(baseline_row)

    for proxy_row in proxy_rows:
        candidate = proxy_row["candidate"]
        row = {
            "name": candidate.name,
            "groups": ";".join(candidate.groups),
            "fp32_layer_count": len(candidate.fp32_layers),
            "fp32_layers": ";".join(sorted(candidate.fp32_layers)),
            "model_mib": proxy_row["model_mib"],
            "proxy_map50": proxy_row["metric"]["map50"],
            "proxy_map50_95": proxy_row["metric"]["map50_95"],
        }
        if candidate.name in selected_names:
            print(f"完整评估与Benchmark：{candidate.name}")
            full_metric = evaluate_model(proxy_row["model_dir"], args.data, candidate.name, "full", artifact_dir, args.force)
            benchmark = benchmark_model(proxy_row["model_dir"], candidate.name, args, artifact_dir)
            row.update({
                "full_precision": full_metric["precision_mean"],
                "full_recall": full_metric["recall_mean"],
                "full_map50": full_metric["map50"],
                "full_map50_95": full_metric["map50_95"],
                "map50_95_drop": fp32_full["map50_95"] - full_metric["map50_95"],
                **benchmark,
            })
        result_rows.append(row)

    mark_pareto(result_rows)
    feasible = [
        row for row in result_rows
        if row["name"] != "fp32"
        and row.get("full_map50_95") is not None
        and row["map50_95_drop"] <= args.max_map_drop
        and row["model_mib"] <= args.max_model_mib
    ]
    if feasible:
        best = sorted(feasible, key=lambda row: (row["end_to_end_mean_ms"], -row["full_map50_95"]))[0]
    else:
        evaluated_quantized = [row for row in result_rows if row["name"] != "fp32" and row.get("full_map50_95") is not None]
        best = sorted(evaluated_quantized, key=lambda row: -row["full_map50_95"])[0]

    result_rows.sort(key=lambda row: (row["name"] != "fp32", row.get("full_map50_95") is None, -(row.get("full_map50_95") or -1)))
    csv_output = results_dir / "mixed_precision_search.csv"
    write_results_csv(result_rows, csv_output)
    pareto_output = results_dir / "pareto_frontier.png"
    draw_pareto(result_rows, pareto_output)

    best_candidate = next(candidate for candidate in candidates if candidate.name == best["name"])
    policy = {
        "selected_policy": best["name"],
        "selection_rule": "在满足mAP50-95下降和模型大小约束的候选中选择端到端延迟最低者",
        "constraints": {"max_map_drop": args.max_map_drop, "max_model_mib": args.max_model_mib},
        "graph": {
            "final_layer": final_layer.name,
            "head_convolution_count": len(head_layers),
            "terminal_convolutions": [layer.name for layer, _ in terminals],
            "anchor_scales": scales,
        },
        "groups": best_candidate.groups,
        "fp32_layers": sorted(best_candidate.fp32_layers),
        "metrics": {key: value for key, value in best.items() if key not in {"fp32_layers"}},
        "fp32_baseline": baseline_row,
    }
    policy_output = results_dir / "best_quantization_policy.json"
    policy_output.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    final_model_dir = root / "models/yolo11n_hit_uav_ncnn_auto_mixed"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    source_model_dir = candidate_models[best["name"]]
    shutil.copy2(source_model_dir / "model.ncnn.param", final_model_dir / "model.ncnn.param")
    shutil.copy2(source_model_dir / "model.ncnn.bin", final_model_dir / "model.ncnn.bin")
    shutil.copy2(policy_output, final_model_dir / "policy.json")
    report_output = results_dir / "mixed_precision_search_report.md"
    write_report(result_rows, best, baseline_row, args, report_output, len(proxy_images))
    print(f"最优策略：{best['name']}")
    print(f"最优模型：{final_model_dir}")
    print(f"搜索结果：{csv_output}")
    print(f"Pareto图：{pareto_output}")
    print(f"策略JSON：{policy_output}")


# 定义命令行参数。
def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="面向红外小目标的NCNN混合精度自动搜索")
    parser.add_argument("--param", type=Path, default=root / "models/yolo11n_hit_uav_ncnn/model.ncnn.param")
    parser.add_argument("--bin", type=Path, default=root / "models/yolo11n_hit_uav_ncnn/model.ncnn.bin")
    parser.add_argument("--table", type=Path, default=root / "results/int8_calibration.table")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("HIT_UAV_DATA_YAML", "HIT-UAV.yaml")).expanduser(),
    )
    parser.add_argument("--ncnn2int8", type=Path, default=Path.home() / "projects/ncnn/build-tools/tools/quantize/ncnn2int8")
    parser.add_argument("--executable", type=Path, default=root / "build/yolo_ncnn")
    parser.add_argument("--classes", type=Path, default=root / "models/classes.txt")
    parser.add_argument("--benchmark-image", type=Path, default=root / "assets/test.jpg")
    parser.add_argument("--max-map-drop", type=float, default=0.04)
    parser.add_argument("--max-model-mib", type=float, default=3.5)
    parser.add_argument("--proxy-images", type=int, default=60)
    parser.add_argument("--full-top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_search(parse_args())
