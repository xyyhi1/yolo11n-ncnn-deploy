#!/usr/bin/env python3
# 文件作用：自动解析 NCNN 图、生成混合精度候选、评估精度与延迟，并选择满足约束的 Pareto 最优策略。

# 中文说明：导入命令行参数解析模块。
import argparse
# 中文说明：导入 CSV 模块，保存全部候选实验结果。
import csv
# 中文说明：导入 JSON 模块，实现指标缓存和最优策略导出。
import json
# 中文说明：导入环境变量模块，使数据集配置不绑定作者本机路径。
import os
# 中文说明：导入文件复制工具，用于准备模型目录和最终模型。
import shutil
# 中文说明：导入子进程模块，调用 ncnn2int8 和 C++ Benchmark。
import subprocess
# 中文说明：导入数据类，清晰表达 NCNN 层结构。
from dataclasses import dataclass
# 中文说明：导入 Path，统一处理文件和目录。
from pathlib import Path

# 中文说明：让 Matplotlib 使用无界面的 Agg 后端，适配 WSL 和自动化环境。
import matplotlib

# 中文说明：在导入 pyplot 之前选择 Agg 后端。
matplotlib.use("Agg")
# 中文说明：导入绘图库，生成 Pareto 前沿图片。
import matplotlib.pyplot as plt
# 中文说明：导入 NumPy，计算平均值和百分位延迟。
import numpy as np
# 中文说明：导入 YAML，读取和生成 Ultralytics 数据集配置。
import yaml
# 中文说明：导入 Ultralytics YOLO，复用其完整检测指标计算。
from ultralytics import YOLO


# 中文说明：NCNN 中可转换为 INT8 的两类卷积层。
CONV_TYPES = {"Convolution", "ConvolutionDepthWise"}


# 中文说明：使用数据类保存一行 NCNN param 对应的图节点。
@dataclass
class Layer:
    # 中文说明：保存算子类型，例如 Convolution、Reshape 或 Split。
    layer_type: str
    # 中文说明：保存 NCNN 层名，例如 conv_67。
    name: str
    # 中文说明：保存该层读取的所有输入 Blob。
    inputs: list[str]
    # 中文说明：保存该层产生的所有输出 Blob。
    outputs: list[str]
    # 中文说明：保存层参数，例如输出通道数和卷积核尺寸。
    attrs: list[str]
    # 中文说明：保存未经修改的原始 param 行，便于调试。
    raw: str


# 中文说明：使用数据类保存一个候选混合精度策略。
@dataclass
class Candidate:
    # 中文说明：策略名称用于目录、CSV 和图表标注。
    name: str
    # 中文说明：集合中的层会保留 FP32，其余可量化层转成 INT8。
    fp32_layers: set[str]
    # 中文说明：说明该候选由哪些自动识别分组组成。
    groups: list[str]


# 中文说明：解析单个 NCNN 层参数中的整数值。
def attr_int(layer: Layer, key: str, default: int = -1) -> int:
    # 中文说明：逐个检查形如 0=64 的属性字符串。
    for token in layer.attrs:
        # 中文说明：判断当前属性是否对应目标键。
        if token.startswith(key + "="):
            # 中文说明：取等号右边的第一个数值，忽略可能存在的逗号数组。
            value = token.split("=", 1)[1].split(",", 1)[0]
            # 中文说明：尝试把字符串转换为整数。
            try:
                # 中文说明：返回成功解析的整数。
                return int(value)
            # 中文说明：格式不符合整数时返回默认值。
            except ValueError:
                # 中文说明：立即使用默认值结束函数。
                return default
    # 中文说明：没有找到目标属性时返回默认值。
    return default


# 中文说明：解析 NCNN `.param` 文件并建立图连接关系。
def parse_ncnn_param(param_path: Path):
    # 中文说明：读取所有非空行。
    lines = [line.strip() for line in param_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 中文说明：NCNN param 至少包含 magic、层统计和一个网络层。
    if len(lines) < 3:
        # 中文说明：文件结构异常时停止搜索。
        raise RuntimeError(f"无效 NCNN param：{param_path}")
    # 中文说明：保存解析后的全部网络层。
    layers = []
    # 中文说明：前两行是 magic 和层/Blob数量，从第三行开始解析网络层。
    for raw in lines[2:]:
        # 中文说明：按照空白符拆分 param 行。
        parts = raw.split()
        # 中文说明：每层至少需要类型、名称、输入数量和输出数量。
        if len(parts) < 4:
            # 中文说明：忽略不完整的异常行。
            continue
        # 中文说明：读取输入 Blob 数量。
        bottom_count = int(parts[2])
        # 中文说明：读取输出 Blob 数量。
        top_count = int(parts[3])
        # 中文说明：计算输入 Blob 字段起点。
        cursor = 4
        # 中文说明：切片取得全部输入 Blob 名称。
        inputs = parts[cursor : cursor + bottom_count]
        # 中文说明：把游标移动到输出 Blob 字段。
        cursor += bottom_count
        # 中文说明：切片取得全部输出 Blob 名称。
        outputs = parts[cursor : cursor + top_count]
        # 中文说明：把游标移动到算子属性字段。
        cursor += top_count
        # 中文说明：构造并保存当前层对象。
        layers.append(Layer(parts[0], parts[1], inputs, outputs, parts[cursor:], raw))
    # 中文说明：通过层名快速查找层对象。
    layer_by_name = {layer.name: layer for layer in layers}
    # 中文说明：通过输出 Blob 查找负责生成它的层。
    producer = {blob: layer for layer in layers for blob in layer.outputs}
    # 中文说明：初始化 Blob 到消费者层列表的映射。
    consumers = {}
    # 中文说明：遍历全部层建立消费者映射。
    for layer in layers:
        # 中文说明：遍历当前层的全部输入 Blob。
        for blob in layer.inputs:
            # 中文说明：把当前层追加为该 Blob 的消费者。
            consumers.setdefault(blob, []).append(layer)
    # 中文说明：返回网络层和三种索引。
    return layers, layer_by_name, producer, consumers


# 中文说明：判断一个 Split 是否是检测头入口边界。
def is_head_boundary_split(layer: Layer, consumers: dict[str, list[Layer]]) -> bool:
    # 中文说明：普通层不是候选边界。
    if layer.layer_type != "Split":
        # 中文说明：非 Split 直接返回 false。
        return False
    # 中文说明：如果 Split 的输出直接送入卷积分支，就把它视为检测头特征入口。
    return any(
        consumer.layer_type in CONV_TYPES
        for blob in layer.outputs
        for consumer in consumers.get(blob, [])
    )


# 中文说明：从最终输出反向遍历，自动识别检测头中的卷积层。
def discover_head_layers(layers: list[Layer], producer: dict[str, Layer], consumers: dict[str, list[Layer]]):
    # 中文说明：优先寻找明确输出 `out0` 的最后一层。
    final_layers = [layer for layer in layers if "out0" in layer.outputs]
    # 中文说明：找不到 out0 时退化为使用 param 的最后一层。
    final_layer = final_layers[-1] if final_layers else layers[-1]
    # 中文说明：反向遍历栈从最终输出层开始。
    stack = [final_layer]
    # 中文说明：防止多分支图中重复访问同一层。
    visited = set()
    # 中文说明：保存检测头范围内遇到的卷积层名。
    head_convolutions = set()
    # 中文说明：持续处理尚未访问的上游层。
    while stack:
        # 中文说明：取出一个待处理层。
        layer = stack.pop()
        # 中文说明：已经访问过的层不再重复处理。
        if layer.name in visited:
            # 中文说明：跳过当前重复节点。
            continue
        # 中文说明：记录该层已经访问。
        visited.add(layer.name)
        # 中文说明：到达向检测头分发特征的 Split 时停止向骨干和颈部回溯。
        if is_head_boundary_split(layer, consumers):
            # 中文说明：不把边界以前的层加入检测头。
            continue
        # 中文说明：检测头范围内的卷积层加入集合。
        if layer.layer_type in CONV_TYPES:
            # 中文说明：保存当前卷积层名。
            head_convolutions.add(layer.name)
        # 中文说明：继续沿当前层所有输入 Blob 向上游遍历。
        for blob in layer.inputs:
            # 中文说明：查找产生当前输入 Blob 的层。
            parent = producer.get(blob)
            # 中文说明：只有真实存在的上游层才加入栈。
            if parent is not None:
                # 中文说明：把上游层加入待处理列表。
                stack.append(parent)
    # 中文说明：返回检测头卷积集合与最终输出层。
    return head_convolutions, final_layer


# 中文说明：识别卷积后直接连接 Reshape 的最终预测卷积。
def discover_terminal_convolutions(head_layers: set[str], layer_by_name: dict[str, Layer], consumers: dict[str, list[Layer]]):
    # 中文说明：保存框回归、分类和DFL最终投影层。
    terminals = []
    # 中文说明：逐个检查自动识别出的检测头卷积。
    for name in sorted(head_layers):
        # 中文说明：取得当前卷积层对象。
        layer = layer_by_name[name]
        # 中文说明：检查任一输出是否直接被 Reshape 使用。
        reshape_consumers = [
            consumer
            for blob in layer.outputs
            for consumer in consumers.get(blob, [])
            if consumer.layer_type == "Reshape"
        ]
        # 中文说明：只有输出进入 Reshape 的卷积才是预测投影候选。
        if reshape_consumers:
            # 中文说明：记录卷积和对应的第一个 Reshape。
            terminals.append((layer, reshape_consumers[0]))
    # 中文说明：返回全部自动识别的最终预测卷积。
    return terminals


# 中文说明：从一个最终预测卷积反向收集同一分支的卷积链。
def trace_branch_convolutions(terminal: Layer, producer: dict[str, Layer], consumers: dict[str, list[Layer]]):
    # 中文说明：DFL投影层的上游是整个回归输出，不应把所有回归分支重复并入该组。
    if attr_int(terminal, "0") == 1:
        # 中文说明：DFL组只包含投影卷积自身。
        return {terminal.name}
    # 中文说明：分支集合从最终预测卷积开始。
    branch = {terminal.name}
    # 中文说明：从最终卷积的所有输入 Blob 开始反向遍历。
    stack = [producer[blob] for blob in terminal.inputs if blob in producer]
    # 中文说明：保存已经访问的分支层。
    visited = set()
    # 中文说明：持续处理上游层，直到到达特征 Split 边界。
    while stack:
        # 中文说明：取出一个上游层。
        layer = stack.pop()
        # 中文说明：跳过重复层。
        if layer.name in visited:
            # 中文说明：进入下一次循环。
            continue
        # 中文说明：标记当前层已访问。
        visited.add(layer.name)
        # 中文说明：遇到检测头输入特征 Split 时停止该分支。
        if is_head_boundary_split(layer, consumers):
            # 中文说明：不继续穿过边界。
            continue
        # 中文说明：把分支内的卷积和深度卷积加入回退集合。
        if layer.layer_type in CONV_TYPES:
            # 中文说明：保存当前卷积层名。
            branch.add(layer.name)
        # 中文说明：继续回溯当前层的输入。
        for blob in layer.inputs:
            # 中文说明：取得当前输入的生产者。
            parent = producer.get(blob)
            # 中文说明：存在上游层时加入栈。
            if parent is not None:
                # 中文说明：把上游层加入待处理列表。
                stack.append(parent)
    # 中文说明：返回该预测分支中的全部卷积层。
    return branch


# 中文说明：根据输出通道、候选点数量和分支链自动建立搜索分组。
def build_search_groups(head_layers, terminals, producer, consumers, num_classes):
    # 中文说明：保存所有可组合的层分组。
    groups = {}
    # 中文说明：保存每个尺度的分类与回归分支，后续合并为P3/P4/P5组。
    scale_groups = {}
    # 中文说明：保存全部分类分支层。
    classification = set()
    # 中文说明：保存全部回归分支层。
    regression = set()
    # 中文说明：保存DFL投影层。
    dfl = set()
    # 中文说明：保存所有最终输出卷积。
    output_layers = set()
    # 中文说明：逐个分析最终预测卷积。
    for terminal, reshape in terminals:
        # 中文说明：读取卷积输出通道数。
        output_channels = attr_int(terminal, "0")
        # 中文说明：读取后续 Reshape 的候选点数量，例如6400、1600或400。
        anchor_count = attr_int(reshape, "0")
        # 中文说明：反向收集当前预测分支的全部卷积。
        branch = trace_branch_convolutions(terminal, producer, consumers)
        # 中文说明：所有最终卷积都加入输出层集合。
        output_layers.add(terminal.name)
        # 中文说明：输出通道等于类别数时属于分类分支。
        if output_channels == num_classes:
            # 中文说明：把当前分支加入全部分类层。
            classification |= branch
            # 中文说明：按候选点数量记录该尺度分类分支。
            scale_groups.setdefault(anchor_count, set()).update(branch)
            # 中文说明：给单独分类尺度分支命名。
            groups[f"classification_{anchor_count}"] = branch
        # 中文说明：输出通道为64时属于YOLO11 DFL框回归分支。
        elif output_channels == 64:
            # 中文说明：把当前分支加入全部回归层。
            regression |= branch
            # 中文说明：按候选点数量记录该尺度回归分支。
            scale_groups.setdefault(anchor_count, set()).update(branch)
            # 中文说明：给单独回归尺度分支命名。
            groups[f"regression_{anchor_count}"] = branch
        # 中文说明：输出通道为1的卷积是DFL固定投影层。
        elif output_channels == 1:
            # 中文说明：保存DFL投影分支。
            dfl |= branch
    # 中文说明：保存全部最终预测卷积分组。
    groups["output"] = output_layers
    # 中文说明：保存完整分类头分组。
    groups["classification"] = classification
    # 中文说明：保存完整回归头分组。
    groups["regression"] = regression
    # 中文说明：保存DFL投影分组。
    groups["dfl"] = dfl
    # 中文说明：保存整个检测头分组，以自动识别结果为准。
    groups["head"] = set(head_layers)
    # 中文说明：按候选点从多到少排序，分别对应小、中、大尺度特征层。
    sorted_scales = sorted((count for count in scale_groups if count > 0), reverse=True)
    # 中文说明：使用P3、P4、P5为三个尺度分组命名。
    for index, anchor_count in enumerate(sorted_scales):
        # 中文说明：根据排序生成P3、P4或P5名称。
        scale_name = f"p{index + 3}"
        # 中文说明：保存该尺度的分类与回归分支并集。
        groups[scale_name] = scale_groups[anchor_count]
    # 中文说明：返回自动识别的全部分组和尺度顺序。
    return groups, sorted_scales


# 中文说明：创建去重后的候选策略列表。
def build_candidates(groups: dict[str, set[str]]):
    # 中文说明：保存候选策略。
    candidates = []
    # 中文说明：保存已经出现的FP32层集合，避免不同组合产生重复模型。
    seen = set()

    # 中文说明：定义内部函数，按分组名称组合FP32层。
    def add(name: str, group_names: list[str]):
        # 中文说明：合并所有存在的目标分组。
        layers = set().union(*(groups[group] for group in group_names if group in groups)) if group_names else set()
        # 中文说明：转换为不可变集合用于去重。
        key = frozenset(layers)
        # 中文说明：重复策略不再次加入搜索空间。
        if key in seen:
            # 中文说明：结束当前候选添加。
            return
        # 中文说明：记录该FP32层集合已经使用。
        seen.add(key)
        # 中文说明：创建候选对象并加入列表。
        candidates.append(Candidate(name, layers, group_names))

    # 中文说明：全INT8作为搜索下界基线。
    add("full_int8", [])
    # 中文说明：只回退最终预测卷积。
    add("output_fp32", ["output"])
    # 中文说明：分别测试分类头、回归头和DFL投影敏感度。
    add("classification_fp32", ["classification"])
    # 中文说明：添加回归头FP32候选。
    add("regression_fp32", ["regression"])
    # 中文说明：添加DFL投影FP32候选。
    add("dfl_fp32", ["dfl"])
    # 中文说明：测试回归头与DFL同时回退。
    add("regression_dfl_fp32", ["regression", "dfl"])
    # 中文说明：测试分类头与DFL同时回退。
    add("classification_dfl_fp32", ["classification", "dfl"])
    # 中文说明：整个检测头FP32是搜索上界候选。
    add("head_fp32", ["head"])
    # 中文说明：逐尺度测试P3、P4、P5敏感度。
    for scale in ("p3", "p4", "p5"):
        # 中文说明：只有成功识别的尺度才加入候选。
        if scale in groups:
            # 中文说明：添加单尺度FP32策略。
            add(f"{scale}_fp32", [scale])
    # 中文说明：测试两个尺度联合回退。
    for first, second in (("p3", "p4"), ("p3", "p5"), ("p4", "p5")):
        # 中文说明：确保两个尺度都存在。
        if first in groups and second in groups:
            # 中文说明：添加双尺度联合策略。
            add(f"{first}_{second}_fp32", [first, second])
    # 中文说明：测试小目标P3分支加所有输出层的组合。
    if "p3" in groups:
        # 中文说明：添加小目标优先候选。
        add("p3_output_fp32", ["p3", "output"])
    # 中文说明：返回去重后的候选列表。
    return candidates


# 中文说明：把图片路径转换成对应YOLO标签路径。
def image_to_label_path(image_path: Path) -> Path:
    # 中文说明：复制路径各级名称，避免直接替换字符串误伤其他目录。
    parts = list(image_path.parts)
    # 中文说明：找到数据集中的 images 目录位置。
    index = parts.index("images")
    # 中文说明：把 images 替换为 labels。
    parts[index] = "labels"
    # 中文说明：把图片扩展名改为 .txt 并返回标签路径。
    return Path(*parts).with_suffix(".txt")


# 中文说明：读取单张图片标签并计算数量和平均归一化面积。
def label_statistics(image_path: Path):
    # 中文说明：取得对应标签文件路径。
    label_path = image_to_label_path(image_path)
    # 中文说明：标签不存在或为空时作为背景图处理。
    if not label_path.is_file() or label_path.stat().st_size == 0:
        # 中文说明：背景图目标数量为0，平均面积设为1便于排序。
        return 0, 1.0
    # 中文说明：读取全部非空标签行。
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 中文说明：保存每个框的归一化面积。
    areas = []
    # 中文说明：遍历每个YOLO标签。
    for line in lines:
        # 中文说明：标签字段依次为类别、中心点、宽和高。
        values = line.split()
        # 中文说明：至少五个字段才是合法检测标签。
        if len(values) >= 5:
            # 中文说明：归一化宽乘高得到目标面积占比。
            areas.append(float(values[3]) * float(values[4]))
    # 中文说明：返回目标数和平均面积，没有合法框时面积使用1。
    return len(areas), float(np.mean(areas)) if areas else 1.0


# 中文说明：构造兼顾背景、密集和小目标场景的确定性代理验证集。
def create_proxy_dataset(data_yaml: Path, output_dir: Path, proxy_count: int):
    # 中文说明：读取原始Ultralytics数据集配置。
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    # 中文说明：解析数据集根目录。
    dataset_root = Path(data["path"]).expanduser().resolve()
    # 中文说明：取得验证集路径配置。
    val_entry = data["val"]
    # 中文说明：当前项目的val是目录，相对路径基于数据集根目录。
    val_path = Path(val_entry)
    # 中文说明：相对路径拼接到数据集根目录。
    if not val_path.is_absolute():
        # 中文说明：生成验证集绝对路径。
        val_path = dataset_root / val_path
    # 中文说明：支持验证集目录和文本清单两种格式。
    if val_path.is_dir():
        # 中文说明：收集目录中的常见图片文件。
        images = sorted(path.resolve() for path in val_path.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    # 中文说明：文本文件时逐行读取图片路径。
    else:
        # 中文说明：解析清单中的非空路径。
        images = [Path(line.strip()).resolve() for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 中文说明：代理集不能超过完整验证集大小。
    proxy_count = min(proxy_count, len(images))
    # 中文说明：为每张图计算目标数量和平均面积。
    records = [(path, *label_statistics(path)) for path in images]
    # 中文说明：背景图用于检查误检。
    backgrounds = [record for record in records if record[1] == 0]
    # 中文说明：非背景图用于小目标和密集场景排序。
    foregrounds = [record for record in records if record[1] > 0]
    # 中文说明：保存已经选中的图片并保持唯一。
    selected = []
    # 中文说明：定义内部函数安全追加图片。
    def append_unique(record):
        # 中文说明：只在未选择且数量未满时追加。
        if record[0] not in selected and len(selected) < proxy_count:
            # 中文说明：保存图片路径。
            selected.append(record[0])
    # 中文说明：最多选择代理集10%的背景图。
    for record in backgrounds[: max(1, proxy_count // 10)]:
        # 中文说明：加入背景场景。
        append_unique(record)
    # 中文说明：按平均面积从小到大选择小目标场景。
    for record in sorted(foregrounds, key=lambda item: (item[2], -item[1]))[: proxy_count // 3]:
        # 中文说明：加入小目标优先场景。
        append_unique(record)
    # 中文说明：按目标数量从多到少选择密集场景。
    for record in sorted(foregrounds, key=lambda item: (-item[1], item[2]))[: proxy_count // 3]:
        # 中文说明：加入密集场景。
        append_unique(record)
    # 中文说明：从剩余图片按固定顺序补足代理集，保证可复现。
    for record in records:
        # 中文说明：追加尚未选择的普通场景。
        append_unique(record)
    # 中文说明：创建搜索结果目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：定义代理图片清单文件。
    image_list = output_dir / "proxy_images.txt"
    # 中文说明：每行写入一个绝对图片路径。
    image_list.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
    # 中文说明：复制原数据配置并把val替换为代理清单。
    proxy_data = dict(data)
    # 中文说明：使用绝对清单路径，避免工作目录影响。
    proxy_data["val"] = str(image_list.resolve())
    # 中文说明：定义代理数据集YAML。
    proxy_yaml = output_dir / "proxy_data.yaml"
    # 中文说明：写出Ultralytics可读取的YAML。
    proxy_yaml.write_text(yaml.safe_dump(proxy_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # 中文说明：返回代理配置和选中图片。
    return proxy_yaml, selected


# 中文说明：根据FP32层集合生成新的量化表。
def create_candidate_table(source_table: Path, destination: Path, fp32_layers: set[str]):
    # 中文说明：读取完整KL校准表。
    lines = source_table.read_text(encoding="utf-8").splitlines()
    # 中文说明：保存候选表的全部行。
    output = []
    # 中文说明：记录实际注释的权重尺度层。
    commented = set()
    # 中文说明：逐行处理量化尺度。
    for line in lines:
        # 中文说明：空行直接保留。
        if not line.strip():
            # 中文说明：追加空行。
            output.append(line)
            # 中文说明：继续下一行。
            continue
        # 中文说明：第一列是层尺度名称。
        key = line.split(maxsplit=1)[0]
        # 中文说明：去掉权重尺度后缀得到NCNN层名。
        layer_name = key.removesuffix("_param_0")
        # 中文说明：目标FP32层的权重尺度行需要被注释。
        if key.endswith("_param_0") and layer_name in fp32_layers:
            # 中文说明：加#后ncnn2int8不会量化该层权重。
            output.append("#" + line)
            # 中文说明：记录成功匹配的层。
            commented.add(layer_name)
        # 中文说明：其他尺度保持不变。
        else:
            # 中文说明：追加原始行。
            output.append(line)
    # 中文说明：检查所有目标FP32层都在校准表中有权重尺度。
    missing = fp32_layers - commented
    # 中文说明：缺失层通常代表不可量化层或图识别错误。
    if missing:
        # 中文说明：抛出包含具体层名的异常。
        raise RuntimeError(f"校准表缺少以下权重尺度：{sorted(missing)}")
    # 中文说明：写出候选校准表。
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


# 中文说明：运行ncnn2int8生成一个候选模型，并支持文件缓存。
def quantize_candidate(candidate, args, artifact_dir, metadata_path):
    # 中文说明：候选目录使用_ncnn_model后缀，便于Ultralytics自动识别。
    model_dir = artifact_dir / "models" / f"{candidate.name}_ncnn_model"
    # 中文说明：定义候选模型结构路径。
    output_param = model_dir / "model.ncnn.param"
    # 中文说明：定义候选模型权重路径。
    output_bin = model_dir / "model.ncnn.bin"
    # 中文说明：已存在且未要求强制重建时直接复用。
    if output_param.is_file() and output_bin.is_file() and not args.force:
        # 中文说明：返回缓存模型目录。
        return model_dir
    # 中文说明：创建候选模型和量化表目录。
    model_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：定义候选量化表位置。
    table_path = artifact_dir / "tables" / f"{candidate.name}.table"
    # 中文说明：创建表目录。
    table_path.parent.mkdir(parents=True, exist_ok=True)
    # 中文说明：按照候选FP32层生成表。
    create_candidate_table(args.table, table_path, candidate.fp32_layers)
    # 中文说明：构造官方ncnn2int8命令。
    command = [
        # 中文说明：量化转换工具路径。
        str(args.ncnn2int8),
        # 中文说明：FP32网络结构。
        str(args.param),
        # 中文说明：FP32权重。
        str(args.bin),
        # 中文说明：候选网络结构输出。
        str(output_param),
        # 中文说明：候选权重输出。
        str(output_bin),
        # 中文说明：候选混合精度表。
        str(table_path),
    ]
    # 中文说明：运行模型转换并捕获日志。
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    # 中文说明：保存转换日志用于排查层尺度问题。
    (model_dir / "quantize.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    # 中文说明：复制Ultralytics模型元数据。
    shutil.copy2(metadata_path, model_dir / "metadata.yaml")
    # 中文说明：返回候选模型目录。
    return model_dir


# 中文说明：准备FP32评估目录，使其同样满足Ultralytics命名约定。
def prepare_fp32_eval_model(args, artifact_dir, metadata_path):
    # 中文说明：定义FP32评估目录。
    model_dir = artifact_dir / "models" / "fp32_ncnn_model"
    # 中文说明：创建目录。
    model_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：复制FP32网络结构。
    shutil.copy2(args.param, model_dir / "model.ncnn.param")
    # 中文说明：复制FP32权重。
    shutil.copy2(args.bin, model_dir / "model.ncnn.bin")
    # 中文说明：复制模型元数据。
    shutil.copy2(metadata_path, model_dir / "metadata.yaml")
    # 中文说明：返回评估目录。
    return model_dir


# 中文说明：评估单个NCNN模型并把指标缓存为JSON。
def evaluate_model(model_dir, data_yaml, name, scope, artifact_dir, force=False):
    # 中文说明：定义当前指标缓存文件。
    cache_path = artifact_dir / "metrics" / f"{name}_{scope}.json"
    # 中文说明：已有缓存且未强制重跑时直接读取。
    if cache_path.is_file() and not force:
        # 中文说明：返回缓存字典。
        return json.loads(cache_path.read_text(encoding="utf-8"))
    # 中文说明：创建指标目录。
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 中文说明：加载NCNN模型并显式指定检测任务。
    model = YOLO(model_dir, task="detect")
    # 中文说明：运行Ultralytics验证流程。
    metrics = model.val(
        # 中文说明：使用代理集或完整验证集YAML。
        data=str(data_yaml),
        # 中文说明：模型输入尺寸保持640。
        imgsz=640,
        # 中文说明：NCNN按单图推理。
        batch=1,
        # 中文说明：关闭额外数据加载进程。
        workers=0,
        # 中文说明：不生成混淆矩阵等图片。
        plots=False,
        # 中文说明：集中保存框架验证产物。
        project=str(artifact_dir / "val_runs"),
        # 中文说明：不同作用域使用独立目录。
        name=f"{name}_{scope}",
        # 中文说明：允许覆盖相同名称目录。
        exist_ok=True,
        # 中文说明：减少逐类别日志，但进度仍可观察。
        verbose=False,
    )
    # 中文说明：提取统一检测指标。
    result = {
        # 中文说明：平均Precision。
        "precision_mean": float(metrics.box.mp),
        # 中文说明：平均Recall。
        "recall_mean": float(metrics.box.mr),
        # 中文说明：IoU=0.5的mAP。
        "map50": float(metrics.box.map50),
        # 中文说明：IoU 0.5到0.95的mAP。
        "map50_95": float(metrics.box.map),
    }
    # 中文说明：把指标写入缓存。
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 中文说明：返回指标字典。
    return result


# 中文说明：运行C++程序的20次预热和100次正式Benchmark。
def benchmark_model(model_dir, name, args, artifact_dir):
    # 中文说明：定义性能指标缓存。
    cache_path = artifact_dir / "metrics" / f"{name}_benchmark.json"
    # 中文说明：已有缓存且未强制运行时直接复用。
    if cache_path.is_file() and not args.force:
        # 中文说明：返回缓存性能数据。
        return json.loads(cache_path.read_text(encoding="utf-8"))
    # 中文说明：定义逐次计时CSV路径。
    csv_path = artifact_dir / "benchmarks" / f"{name}.csv"
    # 中文说明：创建Benchmark目录。
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 中文说明：定义画框输出图片，主要用于确认程序成功完成。
    output_image = artifact_dir / "benchmarks" / f"{name}.jpg"
    # 中文说明：构造与FP32基线完全一致的C++命令。
    command = [
        # 中文说明：独立C++推理程序。
        str(args.executable),
        # 中文说明：候选模型结构。
        "--param", str(model_dir / "model.ncnn.param"),
        # 中文说明：候选模型权重。
        "--bin", str(model_dir / "model.ncnn.bin"),
        # 中文说明：五类别名称文件。
        "--classes", str(args.classes),
        # 中文说明：固定测试图片。
        "--image", str(args.benchmark_image),
        # 中文说明：候选画框结果。
        "--output", str(output_image),
        # 中文说明：固定640输入。
        "--imgsz", "640",
        # 中文说明：固定0.25置信度阈值。
        "--conf", "0.25",
        # 中文说明：固定0.45 NMS阈值。
        "--iou", "0.45",
        # 中文说明：使用用户指定CPU线程数。
        "--threads", str(args.threads),
        # 中文说明：执行20次预热。
        "--warmup", "20",
        # 中文说明：执行100次正式运行。
        "--runs", "100",
        # 中文说明：保存逐次耗时。
        "--benchmark-csv", str(csv_path),
    ]
    # 中文说明：运行C++ Benchmark，失败时立即抛出异常。
    subprocess.run(command, check=True, text=True, capture_output=True)
    # 中文说明：读取100次计时记录。
    with csv_path.open(encoding="utf-8") as handle:
        # 中文说明：把CSV全部行转换为字典。
        rows = list(csv.DictReader(handle))
    # 中文说明：提取端到端耗时数组。
    total = np.array([float(row["end_to_end_ms"]) for row in rows])
    # 中文说明：提取模型推理耗时数组。
    inference = np.array([float(row["inference_ms"]) for row in rows])
    # 中文说明：计算统一性能指标。
    result = {
        # 中文说明：记录正式运行次数。
        "runs": len(rows),
        # 中文说明：平均NCNN推理耗时。
        "inference_mean_ms": float(inference.mean()),
        # 中文说明：平均端到端耗时。
        "end_to_end_mean_ms": float(total.mean()),
        # 中文说明：端到端P50。
        "end_to_end_p50_ms": float(np.percentile(total, 50)),
        # 中文说明：端到端P95。
        "end_to_end_p95_ms": float(np.percentile(total, 95)),
        # 中文说明：根据平均延迟换算FPS。
        "fps": float(1000.0 / total.mean()),
    }
    # 中文说明：写入性能缓存。
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 中文说明：返回性能指标。
    return result


# 中文说明：判断一条结果是否被其他候选在精度、延迟和大小三个维度支配。
def mark_pareto(rows):
    # 中文说明：只处理同时拥有完整精度和延迟的数据。
    valid_rows = [row for row in rows if row.get("full_map50_95") is not None and row.get("end_to_end_mean_ms") is not None]
    # 中文说明：逐个判断候选是否位于Pareto前沿。
    for row in valid_rows:
        # 中文说明：初始假设当前候选未被支配。
        dominated = False
        # 中文说明：使用其他候选与当前候选比较。
        for other in valid_rows:
            # 中文说明：不与自身比较。
            if other is row:
                # 中文说明：跳过自身。
                continue
            # 中文说明：检查其他候选在三个目标上是否都不差。
            no_worse = (
                other["full_map50_95"] >= row["full_map50_95"]
                and other["end_to_end_mean_ms"] <= row["end_to_end_mean_ms"]
                and other["model_mib"] <= row["model_mib"]
            )
            # 中文说明：检查至少一个目标严格更好。
            strictly_better = (
                other["full_map50_95"] > row["full_map50_95"]
                or other["end_to_end_mean_ms"] < row["end_to_end_mean_ms"]
                or other["model_mib"] < row["model_mib"]
            )
            # 中文说明：同时满足时当前候选被支配。
            if no_worse and strictly_better:
                # 中文说明：记录支配状态。
                dominated = True
                # 中文说明：无需继续比较其他候选。
                break
        # 中文说明：未被任何候选支配的点属于Pareto前沿。
        row["pareto"] = not dominated


# 中文说明：生成延迟—精度Pareto散点图。
def draw_pareto(rows, output_path):
    # 中文说明：筛选拥有完整指标的候选。
    valid = [row for row in rows if row.get("full_map50_95") is not None and row.get("end_to_end_mean_ms") is not None]
    # 中文说明：创建清晰的宽屏画布。
    plt.figure(figsize=(12, 7))
    # 中文说明：按照延迟排序，使标签偏移选择稳定可复现。
    valid.sort(key=lambda row: row["end_to_end_mean_ms"])
    # 中文说明：逐个绘制候选点。
    for index, row in enumerate(valid):
        # 中文说明：Pareto点用红色，其他点用蓝色。
        color = "#d62728" if row.get("pareto") else "#1f77b4"
        # 中文说明：模型越大，散点面积越大。
        size = 80 + row["model_mib"] * 25
        # 中文说明：绘制当前候选，并把编号与名称加入图例。
        plt.scatter(
            row["end_to_end_mean_ms"],
            row["full_map50_95"],
            s=size,
            c=color,
            alpha=0.8,
            label=f"{index + 1}: {row['name']}",
        )
        # 中文说明：点附近只标注短编号，完整名称放到右侧图例，避免文字重叠。
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
    # 中文说明：横轴表示越低越好的端到端延迟。
    plt.xlabel("End-to-end latency (ms, lower is better)")
    # 中文说明：纵轴表示越高越好的完整验证集mAP50-95。
    plt.ylabel("Validation mAP50-95 (higher is better)")
    # 中文说明：设置图表标题。
    plt.title("NCNN Mixed-Precision Search Pareto Frontier")
    # 中文说明：显示网格方便比较候选点。
    plt.grid(True, linestyle="--", alpha=0.35)
    # 中文说明：把编号与完整策略名称的图例放在图外右侧。
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    # 中文说明：自动调整边距。
    plt.tight_layout()
    # 中文说明：保存高分辨率图片。
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    # 中文说明：关闭画布释放内存。
    plt.close()


# 中文说明：把所有候选结果写入CSV。
def write_results_csv(rows, output_path):
    # 中文说明：统一输出字段顺序。
    fieldnames = [
        "name", "groups", "fp32_layer_count", "fp32_layers", "model_mib",
        "proxy_map50", "proxy_map50_95", "full_precision", "full_recall",
        "full_map50", "full_map50_95", "map50_95_drop", "inference_mean_ms",
        "end_to_end_mean_ms", "end_to_end_p50_ms", "end_to_end_p95_ms", "fps", "pareto",
    ]
    # 中文说明：创建结果CSV。
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        # 中文说明：构造字典写入器。
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # 中文说明：写入表头。
        writer.writeheader()
        # 中文说明：逐行写入，缺少完整评估的候选使用空值。
        for row in rows:
            # 中文说明：只输出预定义字段。
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# 中文说明：生成便于阅读的中文Markdown实验报告。
def write_report(rows, best, baseline, args, output_path, proxy_count):
    # 中文说明：按是否完整评估和mAP排序展示候选。
    evaluated = sorted(
        (row for row in rows if row.get("full_map50_95") is not None),
        key=lambda row: row["full_map50_95"],
        reverse=True,
    )
    # 中文说明：构造报告文本行。
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
    # 中文说明：追加每个完整评估候选。
    for row in evaluated:
        # 中文说明：格式化一行Markdown表格。
        lines.append(
            f"| {row['name']} | {row['fp32_layer_count']} | {row['model_mib']:.2f} | "
            f"{row['full_map50_95']:.4f} | {row['map50_95_drop']:.4f} | "
            f"{row['end_to_end_mean_ms']:.2f} | {row['fps']:.2f} | "
            f"{'是' if row.get('pareto') else '否'} |"
        )
    # 中文说明：追加自动选择结论。
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
    # 中文说明：写出报告并保留末尾换行。
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 中文说明：定义完整自动搜索主流程。
def run_search(args):
    # 中文说明：确保所有输入路径均转换为绝对路径。
    args.param = args.param.resolve()
    # 中文说明：解析FP32权重绝对路径。
    args.bin = args.bin.resolve()
    # 中文说明：解析校准表绝对路径。
    args.table = args.table.resolve()
    # 中文说明：解析其他运行依赖路径。
    args.data = args.data.resolve()
    # 中文说明：解析量化工具路径。
    args.ncnn2int8 = args.ncnn2int8.resolve()
    # 中文说明：解析C++可执行程序路径。
    args.executable = args.executable.resolve()
    # 中文说明：解析类别文件路径。
    args.classes = args.classes.resolve()
    # 中文说明：解析Benchmark图片路径。
    args.benchmark_image = args.benchmark_image.resolve()
    # 中文说明：定义项目根目录。
    root = Path(__file__).resolve().parents[1]
    # 中文说明：定义可重新生成的搜索中间目录。
    artifact_dir = root / "search_artifacts"
    # 中文说明：定义最终结果目录。
    results_dir = root / "results"
    # 中文说明：创建搜索和结果目录。
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：确保结果目录存在。
    results_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：定位Ultralytics导出元数据。
    metadata_path = root / "models/source/yolo11n_hit_uav_ncnn_model/metadata.yaml"
    # 中文说明：检查搜索所需的每个输入文件。
    for path in (args.param, args.bin, args.table, args.data, args.ncnn2int8, args.executable, args.classes, args.benchmark_image, metadata_path):
        # 中文说明：任一文件缺失都会导致实验不可复现。
        if not path.is_file():
            # 中文说明：抛出缺失文件路径。
            raise FileNotFoundError(path)

    # 中文说明：解析NCNN网络图。
    layers, layer_by_name, producer, consumers = parse_ncnn_param(args.param)
    # 中文说明：从classes.txt读取类别数。
    num_classes = len([line for line in args.classes.read_text(encoding="utf-8").splitlines() if line.strip()])
    # 中文说明：自动识别检测头卷积层。
    head_layers, final_layer = discover_head_layers(layers, producer, consumers)
    # 中文说明：自动识别七个最终预测卷积。
    terminals = discover_terminal_convolutions(head_layers, layer_by_name, consumers)
    # 中文说明：构造分类、回归、DFL和三尺度分组。
    groups, scales = build_search_groups(head_layers, terminals, producer, consumers, num_classes)
    # 中文说明：根据分组生成去重候选策略。
    candidates = build_candidates(groups)
    # 中文说明：打印图解析摘要供用户核验。
    print(f"最终输出层：{final_layer.name}")
    # 中文说明：打印检测头卷积数量。
    print(f"自动识别检测头卷积：{len(head_layers)} 层")
    # 中文说明：打印最终预测卷积名称。
    print(f"最终预测卷积：{[layer.name for layer, _ in terminals]}")
    # 中文说明：打印尺度候选点数量。
    print(f"自动识别尺度：{scales}")
    # 中文说明：打印候选策略数量。
    print(f"候选策略：{len(candidates)} 个")
    # 中文说明：dry-run只验证图解析和策略生成，不执行量化评估。
    if args.dry_run:
        # 中文说明：逐个打印候选和FP32层。
        for candidate in candidates:
            # 中文说明：输出策略摘要。
            print(candidate.name, len(candidate.fp32_layers), sorted(candidate.fp32_layers))
        # 中文说明：结束dry-run。
        return

    # 中文说明：生成60张左右的分层代理验证集。
    proxy_yaml, proxy_images = create_proxy_dataset(args.data, artifact_dir / "proxy", args.proxy_images)
    # 中文说明：准备FP32评估模型。
    fp32_model = prepare_fp32_eval_model(args, artifact_dir, metadata_path)
    # 中文说明：评估FP32代理集，作为候选精度下降参考。
    fp32_proxy = evaluate_model(fp32_model, proxy_yaml, "fp32", "proxy", artifact_dir, args.force)
    # 中文说明：评估FP32完整验证集。
    fp32_full = evaluate_model(fp32_model, args.data, "fp32", "full", artifact_dir, args.force)
    # 中文说明：运行FP32公平Benchmark。
    fp32_benchmark = benchmark_model(fp32_model, "fp32", args, artifact_dir)
    # 中文说明：保存候选对象到模型目录映射。
    candidate_models = {}
    # 中文说明：保存代理评估结果。
    proxy_rows = []
    # 中文说明：逐个量化并评估所有候选。
    for index, candidate in enumerate(candidates, start=1):
        # 中文说明：输出当前搜索进度。
        print(f"[{index}/{len(candidates)}] 量化并代理评估 {candidate.name}")
        # 中文说明：生成或复用候选模型。
        model_dir = quantize_candidate(candidate, args, artifact_dir, metadata_path)
        # 中文说明：记录候选模型目录。
        candidate_models[candidate.name] = model_dir
        # 中文说明：运行代理验证集评估。
        proxy_metric = evaluate_model(model_dir, proxy_yaml, candidate.name, "proxy", artifact_dir, args.force)
        # 中文说明：保存代理排序所需数据。
        proxy_rows.append({
            # 中文说明：保存候选对象。
            "candidate": candidate,
            # 中文说明：保存模型目录。
            "model_dir": model_dir,
            # 中文说明：保存代理指标。
            "metric": proxy_metric,
            # 中文说明：保存权重MiB大小。
            "model_mib": (model_dir / "model.ncnn.bin").stat().st_size / 1024 / 1024,
        })

    # 中文说明：代理排序优先mAP50-95高，其次模型更小。
    ranked = sorted(proxy_rows, key=lambda row: (-row["metric"]["map50_95"], row["model_mib"]))
    # 中文说明：选取代理表现最好的top-k进入完整评估。
    selected_names = {row["candidate"].name for row in ranked[: args.full_top_k]}
    # 中文说明：强制加入三个关键基线，保证自动结果可与手工实验比较。
    selected_names |= {"full_int8", "output_fp32", "head_fp32"}
    # 中文说明：保存最终CSV的全部候选行。
    result_rows = []
    # 中文说明：先加入FP32基线行。
    baseline_row = {
        # 中文说明：FP32基线名称。
        "name": "fp32",
        # 中文说明：FP32不属于回退组合。
        "groups": "baseline",
        # 中文说明：FP32层数使用全部可量化卷积数量。
        "fp32_layer_count": sum(layer.layer_type in CONV_TYPES for layer in layers),
        # 中文说明：基线不展开全部层名。
        "fp32_layers": "all",
        # 中文说明：计算FP32权重MiB。
        "model_mib": args.bin.stat().st_size / 1024 / 1024,
        # 中文说明：记录代理mAP50。
        "proxy_map50": fp32_proxy["map50"],
        # 中文说明：记录代理mAP50-95。
        "proxy_map50_95": fp32_proxy["map50_95"],
        # 中文说明：记录完整Precision。
        "full_precision": fp32_full["precision_mean"],
        # 中文说明：记录完整Recall。
        "full_recall": fp32_full["recall_mean"],
        # 中文说明：记录完整mAP50。
        "full_map50": fp32_full["map50"],
        # 中文说明：记录完整mAP50-95。
        "full_map50_95": fp32_full["map50_95"],
        # 中文说明：基线自身精度下降为0。
        "map50_95_drop": 0.0,
        # 中文说明：合并FP32性能数据。
        **fp32_benchmark,
    }
    # 中文说明：把FP32基线加入结果。
    result_rows.append(baseline_row)

    # 中文说明：逐个整理候选，入围者执行完整评估和Benchmark。
    for proxy_row in proxy_rows:
        # 中文说明：取得候选对象。
        candidate = proxy_row["candidate"]
        # 中文说明：创建所有候选共有字段。
        row = {
            # 中文说明：候选名称。
            "name": candidate.name,
            # 中文说明：组合分组名称。
            "groups": ";".join(candidate.groups),
            # 中文说明：FP32回退层数量。
            "fp32_layer_count": len(candidate.fp32_layers),
            # 中文说明：具体FP32层清单。
            "fp32_layers": ";".join(sorted(candidate.fp32_layers)),
            # 中文说明：候选权重大小。
            "model_mib": proxy_row["model_mib"],
            # 中文说明：代理mAP50。
            "proxy_map50": proxy_row["metric"]["map50"],
            # 中文说明：代理mAP50-95。
            "proxy_map50_95": proxy_row["metric"]["map50_95"],
        }
        # 中文说明：只有代理筛选入围候选才运行完整评估。
        if candidate.name in selected_names:
            # 中文说明：提示当前完整评估策略。
            print(f"完整评估与Benchmark：{candidate.name}")
            # 中文说明：运行完整290张验证集。
            full_metric = evaluate_model(proxy_row["model_dir"], args.data, candidate.name, "full", artifact_dir, args.force)
            # 中文说明：运行100次C++ Benchmark。
            benchmark = benchmark_model(proxy_row["model_dir"], candidate.name, args, artifact_dir)
            # 中文说明：保存完整精度指标。
            row.update({
                # 中文说明：完整Precision。
                "full_precision": full_metric["precision_mean"],
                # 中文说明：完整Recall。
                "full_recall": full_metric["recall_mean"],
                # 中文说明：完整mAP50。
                "full_map50": full_metric["map50"],
                # 中文说明：完整mAP50-95。
                "full_map50_95": full_metric["map50_95"],
                # 中文说明：计算相对FP32的mAP50-95下降。
                "map50_95_drop": fp32_full["map50_95"] - full_metric["map50_95"],
                # 中文说明：合并性能统计。
                **benchmark,
            })
        # 中文说明：把候选加入总结果。
        result_rows.append(row)

    # 中文说明：计算完整评估候选的Pareto状态。
    mark_pareto(result_rows)
    # 中文说明：筛选满足用户精度和大小约束的量化候选。
    feasible = [
        row for row in result_rows
        if row["name"] != "fp32"
        and row.get("full_map50_95") is not None
        and row["map50_95_drop"] <= args.max_map_drop
        and row["model_mib"] <= args.max_model_mib
    ]
    # 中文说明：存在可行候选时选择端到端延迟最低者，精度作为次级条件。
    if feasible:
        # 中文说明：按延迟升序、mAP降序排序并取第一名。
        best = sorted(feasible, key=lambda row: (row["end_to_end_mean_ms"], -row["full_map50_95"]))[0]
    # 中文说明：没有候选满足约束时退化为选择完整mAP最高的量化模型。
    else:
        # 中文说明：只考虑已经完整评估的量化候选。
        evaluated_quantized = [row for row in result_rows if row["name"] != "fp32" and row.get("full_map50_95") is not None]
        # 中文说明：按mAP降序选择。
        best = sorted(evaluated_quantized, key=lambda row: -row["full_map50_95"])[0]

    # 中文说明：按FP32、完整评估、代理精度的顺序排序CSV。
    result_rows.sort(key=lambda row: (row["name"] != "fp32", row.get("full_map50_95") is None, -(row.get("full_map50_95") or -1)))
    # 中文说明：定义搜索CSV输出。
    csv_output = results_dir / "mixed_precision_search.csv"
    # 中文说明：写入全部候选结果。
    write_results_csv(result_rows, csv_output)
    # 中文说明：定义Pareto图输出。
    pareto_output = results_dir / "pareto_frontier.png"
    # 中文说明：绘制Pareto图。
    draw_pareto(result_rows, pareto_output)

    # 中文说明：取得最优候选对象，用于导出层策略。
    best_candidate = next(candidate for candidate in candidates if candidate.name == best["name"])
    # 中文说明：构造完整策略JSON。
    policy = {
        # 中文说明：记录自动选择名称。
        "selected_policy": best["name"],
        # 中文说明：记录选择规则。
        "selection_rule": "在满足mAP50-95下降和模型大小约束的候选中选择端到端延迟最低者",
        # 中文说明：保存搜索约束。
        "constraints": {"max_map_drop": args.max_map_drop, "max_model_mib": args.max_model_mib},
        # 中文说明：保存自动识别网络信息。
        "graph": {
            # 中文说明：最终输出层名称。
            "final_layer": final_layer.name,
            # 中文说明：检测头卷积数量。
            "head_convolution_count": len(head_layers),
            # 中文说明：最终预测卷积名称。
            "terminal_convolutions": [layer.name for layer, _ in terminals],
            # 中文说明：三个尺度候选点数量。
            "anchor_scales": scales,
        },
        # 中文说明：保存最优策略分组。
        "groups": best_candidate.groups,
        # 中文说明：保存所有FP32回退层。
        "fp32_layers": sorted(best_candidate.fp32_layers),
        # 中文说明：保存最优模型指标。
        "metrics": {key: value for key, value in best.items() if key not in {"fp32_layers"}},
        # 中文说明：保存FP32基线指标用于复核。
        "fp32_baseline": baseline_row,
    }
    # 中文说明：定义策略JSON路径。
    policy_output = results_dir / "best_quantization_policy.json"
    # 中文说明：写出UTF-8策略JSON。
    policy_output.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    # 中文说明：把最优模型复制到稳定的项目模型目录。
    final_model_dir = root / "models/yolo11n_hit_uav_ncnn_auto_mixed"
    # 中文说明：创建最终模型目录。
    final_model_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：取得搜索中最优候选模型目录。
    source_model_dir = candidate_models[best["name"]]
    # 中文说明：复制最优网络结构。
    shutil.copy2(source_model_dir / "model.ncnn.param", final_model_dir / "model.ncnn.param")
    # 中文说明：复制最优模型权重。
    shutil.copy2(source_model_dir / "model.ncnn.bin", final_model_dir / "model.ncnn.bin")
    # 中文说明：在模型目录保存对应策略。
    shutil.copy2(policy_output, final_model_dir / "policy.json")
    # 中文说明：生成中文自动搜索报告。
    report_output = results_dir / "mixed_precision_search_report.md"
    # 中文说明：写出报告。
    write_report(result_rows, best, baseline_row, args, report_output, len(proxy_images))
    # 中文说明：打印最终结果路径。
    print(f"最优策略：{best['name']}")
    # 中文说明：打印最优模型目录。
    print(f"最优模型：{final_model_dir}")
    # 中文说明：打印CSV路径。
    print(f"搜索结果：{csv_output}")
    # 中文说明：打印Pareto图路径。
    print(f"Pareto图：{pareto_output}")
    # 中文说明：打印策略JSON路径。
    print(f"策略JSON：{policy_output}")


# 中文说明：定义命令行参数。
def parse_args():
    # 中文说明：取得项目根目录，构造默认路径。
    root = Path(__file__).resolve().parents[1]
    # 中文说明：创建参数解析器。
    parser = argparse.ArgumentParser(description="面向红外小目标的NCNN混合精度自动搜索")
    # 中文说明：FP32 NCNN结构文件。
    parser.add_argument("--param", type=Path, default=root / "models/yolo11n_hit_uav_ncnn/model.ncnn.param")
    # 中文说明：FP32 NCNN权重文件。
    parser.add_argument("--bin", type=Path, default=root / "models/yolo11n_hit_uav_ncnn/model.ncnn.bin")
    # 中文说明：完整KL校准表。
    parser.add_argument("--table", type=Path, default=root / "results/int8_calibration.table")
    # 中文说明：HIT-UAV数据集配置。
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("HIT_UAV_DATA_YAML", "HIT-UAV.yaml")).expanduser(),
    )
    # 中文说明：NCNN官方量化工具路径。
    parser.add_argument("--ncnn2int8", type=Path, default=Path.home() / "projects/ncnn/build-tools/tools/quantize/ncnn2int8")
    # 中文说明：独立C++推理程序路径。
    parser.add_argument("--executable", type=Path, default=root / "build/yolo_ncnn")
    # 中文说明：类别名称文件。
    parser.add_argument("--classes", type=Path, default=root / "models/classes.txt")
    # 中文说明：固定性能测试图片。
    parser.add_argument("--benchmark-image", type=Path, default=root / "assets/test.jpg")
    # 中文说明：允许的最大完整mAP50-95下降，默认0.04。
    parser.add_argument("--max-map-drop", type=float, default=0.04)
    # 中文说明：允许的最大权重MiB大小，默认3.5。
    parser.add_argument("--max-model-mib", type=float, default=3.5)
    # 中文说明：代理验证集图片数，默认60。
    parser.add_argument("--proxy-images", type=int, default=60)
    # 中文说明：代理排序后进入完整评估的候选数量。
    parser.add_argument("--full-top-k", type=int, default=6)
    # 中文说明：C++ Benchmark使用的CPU线程数。
    parser.add_argument("--threads", type=int, default=4)
    # 中文说明：强制重新量化和评估，忽略缓存。
    parser.add_argument("--force", action="store_true")
    # 中文说明：只打印图解析和候选策略，不执行耗时搜索。
    parser.add_argument("--dry-run", action="store_true")
    # 中文说明：返回解析后的参数。
    return parser.parse_args()


# 中文说明：直接运行脚本时执行自动搜索。
if __name__ == "__main__":
    # 中文说明：解析参数并启动主流程。
    run_search(parse_args())
