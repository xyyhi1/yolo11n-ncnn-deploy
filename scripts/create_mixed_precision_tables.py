#!/usr/bin/env python3
# 从完整 INT8 校准表生成两种混合精度表，让量化敏感层保留 FP32 权重。

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE = ROOT / "results/int8_calibration.table"
OUTPUT_LAYERS = {"conv_67", "conv_70", "conv_73", "conv_76", "conv_79", "conv_82", "conv_83"}
HEAD_LAYERS = {
    "conv_65", "conv_66", "conv_67",
    "conv_68", "conv_69", "conv_70",
    "conv_71", "conv_72", "conv_73",
    "convdw_183", "conv_74", "convdw_184", "conv_75", "conv_76",
    "convdw_185", "conv_77", "convdw_186", "conv_78", "conv_79",
    "convdw_187", "conv_80", "convdw_188", "conv_81", "conv_82",
    "conv_83",
}


# 定义混合精度表生成函数。
def create_table(layers: set[str], destination: Path):
    lines = SOURCE_TABLE.read_text(encoding="utf-8").splitlines()
    output_lines = []
    commented = 0
    for line in lines:
        key = line.split(maxsplit=1)[0] if line.strip() else ""
        layer_name = key.removesuffix("_param_0")
        if key.endswith("_param_0") and layer_name in layers:
            output_lines.append("#" + line)
            commented += 1
        else:
            output_lines.append(line)
    if commented != len(layers):
        raise RuntimeError(f"{destination.name}: 期望回退 {len(layers)} 层，实际 {commented} 层")
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"{destination}: 回退 {commented} 层为 FP32")


# 定义脚本主流程。
def main():
    if not SOURCE_TABLE.is_file():
        raise FileNotFoundError(SOURCE_TABLE)
    create_table(OUTPUT_LAYERS, ROOT / "results/int8_calibration_mixed_output_fp32.table")
    create_table(HEAD_LAYERS, ROOT / "results/int8_calibration_mixed_head_fp32.table")


if __name__ == "__main__":
    main()
