#!/usr/bin/env python3
# 把训练图片预处理成与 C++ 推理完全相同的 640×640 Letterbox 校准图片，并生成文件清单。

import argparse
from pathlib import Path

import cv2


# 定义与 C++ preprocess.cpp 一致的 Letterbox 函数。
def letterbox(image, input_size: int):
    original_height, original_width = image.shape[:2]
    scale = min(input_size / original_width, input_size / original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    if (resized_width, resized_height) != (original_width, original_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    dw = (input_size - resized_width) / 2.0
    dh = (input_size - resized_height) / 2.0
    pad_left = round(dw - 0.1)
    pad_right = round(dw + 0.1)
    pad_top = round(dh - 0.1)
    pad_bottom = round(dh + 0.1)
    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


# 定义脚本主函数，负责遍历训练图片并写出校准数据。
def main():
    parser = argparse.ArgumentParser(description="生成 NCNN INT8 PTQ 校准图片")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)

    image_paths = sorted(
        path for path in args.input_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"没有在 {args.input_dir} 中找到图片")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.list_file.parent.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    for index, source_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV 无法读取图片：{source_path}")
        prepared = letterbox(image, args.imgsz)
        destination = args.output_dir / f"calibration_{index:06d}.png"
        if not cv2.imwrite(str(destination), prepared):
            raise RuntimeError(f"无法保存校准图片：{destination}")
        generated_paths.append(destination.resolve())
        if index % 100 == 0 or index == len(image_paths):
            print(f"已生成 {index}/{len(image_paths)} 张校准图片")

    args.list_file.write_text(
        "\n".join(str(path) for path in generated_paths) + "\n",
        encoding="utf-8",
    )
    print(f"校准清单：{args.list_file.resolve()}，共 {len(generated_paths)} 张")


if __name__ == "__main__":
    main()
