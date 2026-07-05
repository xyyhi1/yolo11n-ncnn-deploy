#!/usr/bin/env python3
# 文件作用：验证自动搜索中的图解析、候选生成、混合精度表和Pareto判断，防止后续修改破坏核心逻辑。

# 中文说明：导入临时目录工具，测试时不污染项目文件。
import tempfile
# 中文说明：导入unittest标准测试框架。
import unittest
# 中文说明：导入Path处理测试路径。
from pathlib import Path
# 中文说明：导入sys，把项目scripts目录加入模块搜索路径。
import sys

# 中文说明：取得项目根目录。
ROOT = Path(__file__).resolve().parents[1]
# 中文说明：把scripts目录放到Python导入路径首位。
sys.path.insert(0, str(ROOT / "scripts"))

# 中文说明：导入需要验证的自动搜索函数。
from search_mixed_precision import (  # noqa: E402
    build_candidates,
    build_search_groups,
    create_candidate_table,
    discover_head_layers,
    discover_terminal_convolutions,
    mark_pareto,
    parse_ncnn_param,
)


# 中文说明：定义自动搜索核心测试集合。
class SearchMixedPrecisionTest(unittest.TestCase):
    # 中文说明：每个测试前解析一次真实YOLO11n NCNN计算图。
    def setUp(self):
        # 中文说明：指定FP32模型结构文件。
        self.param = ROOT / "models/yolo11n_hit_uav_ncnn/model.ncnn.param"
        # 中文说明：解析真实图和索引。
        self.layers, self.layer_by_name, self.producer, self.consumers = parse_ncnn_param(self.param)

    # 中文说明：验证工具能自动恢复当前模型的检测头结构。
    def test_graph_discovery(self):
        # 中文说明：自动发现检测头层。
        head, final_layer = discover_head_layers(self.layers, self.producer, self.consumers)
        # 中文说明：自动发现最终预测卷积。
        terminals = discover_terminal_convolutions(head, self.layer_by_name, self.consumers)
        # 中文说明：最终输出层应为cat_20。
        self.assertEqual(final_layer.name, "cat_20")
        # 中文说明：当前模型检测头应有25个卷积。
        self.assertEqual(len(head), 25)
        # 中文说明：七个预测卷积名称必须与真实图一致。
        self.assertEqual(
            [layer.name for layer, _ in terminals],
            ["conv_67", "conv_70", "conv_73", "conv_76", "conv_79", "conv_82", "conv_83"],
        )

    # 中文说明：验证搜索分组和候选数量稳定。
    def test_candidate_generation(self):
        # 中文说明：识别检测头和预测卷积。
        head, _ = discover_head_layers(self.layers, self.producer, self.consumers)
        # 中文说明：取得终端卷积。
        terminals = discover_terminal_convolutions(head, self.layer_by_name, self.consumers)
        # 中文说明：构造搜索分组。
        groups, scales = build_search_groups(head, terminals, self.producer, self.consumers, 5)
        # 中文说明：生成去重候选。
        candidates = build_candidates(groups)
        # 中文说明：三个尺度应按6400、1600、400排序。
        self.assertEqual(scales, [6400, 1600, 400])
        # 中文说明：完整检测头分组应有25层。
        self.assertEqual(len(groups["head"]), 25)
        # 中文说明：当前规则应生成15个不同策略。
        self.assertEqual(len(candidates), 15)

    # 中文说明：验证注释权重尺度可以正确生成混合精度表。
    def test_candidate_table(self):
        # 中文说明：创建临时目录。
        with tempfile.TemporaryDirectory() as directory:
            # 中文说明：定义临时候选表。
            output = Path(directory) / "mixed.table"
            # 中文说明：只让conv_83回退FP32。
            create_candidate_table(ROOT / "results/int8_calibration.table", output, {"conv_83"})
            # 中文说明：读取生成内容。
            content = output.read_text(encoding="utf-8")
            # 中文说明：权重尺度行必须被注释。
            self.assertIn("#conv_83_param_0", content)
            # 中文说明：激活尺度行仍然保留。
            self.assertIn("\nconv_83 ", content)

    # 中文说明：验证Pareto支配判断。
    def test_pareto_marking(self):
        # 中文说明：A比B精度更高、延迟更低且更小，应支配B。
        rows = [
            {"name": "A", "full_map50_95": 0.56, "end_to_end_mean_ms": 24.0, "model_mib": 3.0},
            {"name": "B", "full_map50_95": 0.55, "end_to_end_mean_ms": 25.0, "model_mib": 3.1},
        ]
        # 中文说明：执行Pareto标记。
        mark_pareto(rows)
        # 中文说明：A应位于Pareto前沿。
        self.assertTrue(rows[0]["pareto"])
        # 中文说明：B应被A支配。
        self.assertFalse(rows[1]["pareto"])


# 中文说明：直接运行文件时执行全部测试。
if __name__ == "__main__":
    # 中文说明：使用详细模式输出每个测试名称。
    unittest.main(verbosity=2)
