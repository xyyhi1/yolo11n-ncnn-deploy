#!/usr/bin/env python3
# 文件作用：把训练图片预处理成与 C++ 推理完全相同的 640×640 Letterbox 校准图片，并生成文件清单。

# 中文说明：导入命令行参数解析模块。
import argparse
# 中文说明：导入 Path，用统一方式处理文件和目录路径。
from pathlib import Path

# 中文说明：导入 OpenCV，负责图片读取、缩放、填充和保存。
import cv2


# 中文说明：定义与 C++ preprocess.cpp 一致的 Letterbox 函数。
def letterbox(image, input_size: int):
    # 中文说明：读取原图高度和宽度；OpenCV shape 顺序是高度、宽度、通道。
    original_height, original_width = image.shape[:2]
    # 中文说明：取宽、高缩放比例中的较小值，避免任何一边超过模型输入尺寸。
    scale = min(input_size / original_width, input_size / original_height)
    # 中文说明：按照统一比例计算缩放后的宽度，并使用 round 与 C++ 保持一致。
    resized_width = round(original_width * scale)
    # 中文说明：按照统一比例计算缩放后的高度。
    resized_height = round(original_height * scale)
    # 中文说明：只有尺寸发生变化时才调用 resize，避免不必要的插值误差。
    if (resized_width, resized_height) != (original_width, original_height):
        # 中文说明：使用双线性插值缩放图片，与 NCNN 的常规 resize 行为对应。
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # 中文说明：计算水平方向需要平均分配的填充量。
    dw = (input_size - resized_width) / 2.0
    # 中文说明：计算垂直方向需要平均分配的填充量。
    dh = (input_size - resized_height) / 2.0
    # 中文说明：减去 0.1 后取整，复现 Ultralytics/C++ 的左侧取整方式。
    pad_left = round(dw - 0.1)
    # 中文说明：加上 0.1 后取整，复现右侧取整方式并保证总尺寸正确。
    pad_right = round(dw + 0.1)
    # 中文说明：计算上侧填充像素数。
    pad_top = round(dh - 0.1)
    # 中文说明：计算下侧填充像素数。
    pad_bottom = round(dh + 0.1)
    # 中文说明：使用常量 114 填充四周，这是 YOLO 系列常用的 Letterbox 填充值。
    return cv2.copyMakeBorder(
        # 中文说明：传入已经按比例缩放的图片。
        image,
        # 中文说明：传入上侧填充量。
        pad_top,
        # 中文说明：传入下侧填充量。
        pad_bottom,
        # 中文说明：传入左侧填充量。
        pad_left,
        # 中文说明：传入右侧填充量。
        pad_right,
        # 中文说明：指定使用固定像素值填充，而不是复制边缘像素。
        cv2.BORDER_CONSTANT,
        # 中文说明：BGR 三个通道都填充为 114。
        value=(114, 114, 114),
    )


# 中文说明：定义脚本主函数，负责遍历训练图片并写出校准数据。
def main():
    # 中文说明：创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="生成 NCNN INT8 PTQ 校准图片")
    # 中文说明：要求调用者提供原始训练图片目录。
    parser.add_argument("--input-dir", type=Path, required=True)
    # 中文说明：要求调用者提供 Letterbox 图片输出目录。
    parser.add_argument("--output-dir", type=Path, required=True)
    # 中文说明：要求调用者指定 ncnn2table 使用的绝对路径清单文件。
    parser.add_argument("--list-file", type=Path, required=True)
    # 中文说明：默认模型输入边长为 640。
    parser.add_argument("--imgsz", type=int, default=640)
    # 中文说明：默认处理全部图片；大于零时只处理前 limit 张，便于快速调试。
    parser.add_argument("--limit", type=int, default=0)
    # 中文说明：解析用户传入的全部参数。
    args = parser.parse_args()

    # 中文说明：输入尺寸必须为正数，否则无法生成有效模型输入。
    if args.imgsz <= 0:
        # 中文说明：参数非法时立即报错退出。
        raise ValueError("--imgsz 必须大于 0")
    # 中文说明：输入目录必须真实存在。
    if not args.input_dir.is_dir():
        # 中文说明：目录不存在时报告准确路径。
        raise FileNotFoundError(args.input_dir)

    # 中文说明：递归收集常见格式图片，并排序保证每次生成结果一致。
    image_paths = sorted(
        # 中文说明：只保留扩展名属于常见图片格式的文件。
        path for path in args.input_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    # 中文说明：如果设置了 limit，就截取前指定数量的图片。
    if args.limit > 0:
        # 中文说明：利用列表切片限制校准样本数。
        image_paths = image_paths[: args.limit]
    # 中文说明：没有找到图片时不能继续生成校准表。
    if not image_paths:
        # 中文说明：抛出异常并提示输入目录。
        raise RuntimeError(f"没有在 {args.input_dir} 中找到图片")

    # 中文说明：创建校准图片输出目录，已存在时不报错。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 中文说明：创建清单文件的父目录。
    args.list_file.parent.mkdir(parents=True, exist_ok=True)
    # 中文说明：保存所有成功生成的校准图片绝对路径。
    generated_paths = []

    # 中文说明：逐张处理训练集图片。
    for index, source_path in enumerate(image_paths, start=1):
        # 中文说明：以 BGR 三通道格式读取原图。
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        # 中文说明：读取失败时立即停止，避免校准集静默缺图。
        if image is None:
            # 中文说明：报告无法解码的图片路径。
            raise RuntimeError(f"OpenCV 无法读取图片：{source_path}")
        # 中文说明：执行与部署程序一致的等比例缩放和 114 填充。
        prepared = letterbox(image, args.imgsz)
        # 中文说明：使用六位编号保证文件名排序与生成顺序一致。
        destination = args.output_dir / f"calibration_{index:06d}.png"
        # 中文说明：使用 PNG 无损保存，避免 JPEG 二次压缩改变像素分布。
        if not cv2.imwrite(str(destination), prepared):
            # 中文说明：写入失败时报告具体输出路径。
            raise RuntimeError(f"无法保存校准图片：{destination}")
        # 中文说明：记录绝对路径，ncnn2table 可在任意工作目录读取。
        generated_paths.append(destination.resolve())
        # 中文说明：每处理 100 张或最后一张时输出一次进度。
        if index % 100 == 0 or index == len(image_paths):
            # 中文说明：打印当前完成数量和总数量。
            print(f"已生成 {index}/{len(image_paths)} 张校准图片")

    # 中文说明：每行写入一张校准图片的绝对路径。
    args.list_file.write_text(
        # 中文说明：用换行连接路径，并在文件末尾保留换行符。
        "\n".join(str(path) for path in generated_paths) + "\n",
        # 中文说明：显式使用 UTF-8 编码。
        encoding="utf-8",
    )
    # 中文说明：打印最终清单路径和样本数量。
    print(f"校准清单：{args.list_file.resolve()}，共 {len(generated_paths)} 张")


# 中文说明：只有直接运行脚本时才调用 main，作为模块导入时不会自动执行。
if __name__ == "__main__":
    # 中文说明：进入脚本主流程。
    main()
