#!/usr/bin/env python3
# 文件作用：从完整 INT8 校准表生成两种混合精度表，让量化敏感层保留 FP32 权重。

# 中文说明：导入 Path 处理输入输出文件。
from pathlib import Path


# 中文说明：取得项目根目录。
ROOT = Path(__file__).resolve().parents[1]
# 中文说明：指定全量 KL 校准表。
SOURCE_TABLE = ROOT / "results/int8_calibration.table"
# 中文说明：最终框回归、分类和 DFL 投影层直接决定输出，先作为最小回退集合。
OUTPUT_LAYERS = {"conv_67", "conv_70", "conv_73", "conv_76", "conv_79", "conv_82", "conv_83"}
# 中文说明：检测头回退集合包含三尺度框回归和分类分支的全部卷积层。
HEAD_LAYERS = {
    # 中文说明：加入框回归分支第一尺度的三个卷积层。
    "conv_65", "conv_66", "conv_67",
    # 中文说明：加入框回归分支第二尺度的三个卷积层。
    "conv_68", "conv_69", "conv_70",
    # 中文说明：加入框回归分支第三尺度的三个卷积层。
    "conv_71", "conv_72", "conv_73",
    # 中文说明：加入三个分类分支的深度可分离卷积和点卷积。
    "convdw_183", "conv_74", "convdw_184", "conv_75", "conv_76",
    # 中文说明：加入第二个分类尺度的全部卷积。
    "convdw_185", "conv_77", "convdw_186", "conv_78", "conv_79",
    # 中文说明：加入第三个分类尺度的全部卷积。
    "convdw_187", "conv_80", "convdw_188", "conv_81", "conv_82",
    # 中文说明：加入 DFL 离散分布到连续坐标的投影卷积。
    "conv_83",
}


# 中文说明：定义混合精度表生成函数。
def create_table(layers: set[str], destination: Path):
    # 中文说明：读取原始校准表的每一行。
    lines = SOURCE_TABLE.read_text(encoding="utf-8").splitlines()
    # 中文说明：保存修改后的全部行。
    output_lines = []
    # 中文说明：记录实际回退的权重尺度数量，用于检查层名是否匹配。
    commented = 0
    # 中文说明：逐行检查当前记录属于哪个层。
    for line in lines:
        # 中文说明：取第一个字段，例如 conv_67_param_0 或 conv_67。
        key = line.split(maxsplit=1)[0] if line.strip() else ""
        # 中文说明：去掉 `_param_0` 得到 NCNN 层名。
        layer_name = key.removesuffix("_param_0")
        # 中文说明：只注释权重尺度行；激活尺度行继续保留给相邻 INT8 层使用。
        if key.endswith("_param_0") and layer_name in layers:
            # 中文说明：NCNN 看到被注释的权重尺度后，会让该层继续使用 FP32 权重。
            output_lines.append("#" + line)
            # 中文说明：回退层计数加一。
            commented += 1
        # 中文说明：不属于回退集合的记录保持原样。
        else:
            # 中文说明：追加未修改的校准表行。
            output_lines.append(line)
    # 中文说明：确保每个目标层都找到了对应权重尺度。
    if commented != len(layers):
        # 中文说明：数量不一致说明层名或表格式发生变化，禁止静默生成错误模型。
        raise RuntimeError(f"{destination.name}: 期望回退 {len(layers)} 层，实际 {commented} 层")
    # 中文说明：写出混合精度校准表，并保留文件末尾换行。
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    # 中文说明：打印表名和回退层数量。
    print(f"{destination}: 回退 {commented} 层为 FP32")


# 中文说明：定义脚本主流程。
def main():
    # 中文说明：检查完整校准表存在。
    if not SOURCE_TABLE.is_file():
        # 中文说明：校准表缺失时停止执行。
        raise FileNotFoundError(SOURCE_TABLE)
    # 中文说明：生成只回退最终输出层的混合精度表。
    create_table(OUTPUT_LAYERS, ROOT / "results/int8_calibration_mixed_output_fp32.table")
    # 中文说明：生成回退完整检测头的混合精度表。
    create_table(HEAD_LAYERS, ROOT / "results/int8_calibration_mixed_head_fp32.table")


# 中文说明：直接运行脚本时进入主流程。
if __name__ == "__main__":
    # 中文说明：调用主函数。
    main()
