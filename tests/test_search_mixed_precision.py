#!/usr/bin/env python3
# 验证自动搜索中的图解析、候选生成、混合精度表和Pareto判断，防止后续修改破坏核心逻辑。

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_mixed_precision import (  # noqa: E402
    build_candidates,
    build_search_groups,
    create_candidate_table,
    discover_head_layers,
    discover_terminal_convolutions,
    mark_pareto,
    parse_ncnn_param,
)


# 定义自动搜索核心测试集合。
class SearchMixedPrecisionTest(unittest.TestCase):
    # 每个测试前解析一次真实YOLO11n NCNN计算图。
    def setUp(self):
        self.param = ROOT / "models/yolo11n_hit_uav_ncnn/model.ncnn.param"
        self.layers, self.layer_by_name, self.producer, self.consumers = parse_ncnn_param(self.param)

    # 验证工具能自动恢复当前模型的检测头结构。
    def test_graph_discovery(self):
        head, final_layer = discover_head_layers(self.layers, self.producer, self.consumers)
        terminals = discover_terminal_convolutions(head, self.layer_by_name, self.consumers)
        self.assertEqual(final_layer.name, "cat_20")
        self.assertEqual(len(head), 25)
        self.assertEqual(
            [layer.name for layer, _ in terminals],
            ["conv_67", "conv_70", "conv_73", "conv_76", "conv_79", "conv_82", "conv_83"],
        )

    # 验证搜索分组和候选数量稳定。
    def test_candidate_generation(self):
        head, _ = discover_head_layers(self.layers, self.producer, self.consumers)
        terminals = discover_terminal_convolutions(head, self.layer_by_name, self.consumers)
        groups, scales = build_search_groups(head, terminals, self.producer, self.consumers, 5)
        candidates = build_candidates(groups)
        self.assertEqual(scales, [6400, 1600, 400])
        self.assertEqual(len(groups["head"]), 25)
        self.assertEqual(len(candidates), 15)

    # 验证注释权重尺度可以正确生成混合精度表。
    def test_candidate_table(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mixed.table"
            create_candidate_table(ROOT / "results/int8_calibration.table", output, {"conv_83"})
            content = output.read_text(encoding="utf-8")
            self.assertIn("#conv_83_param_0", content)
            self.assertIn("\nconv_83 ", content)

    # 验证Pareto支配判断。
    def test_pareto_marking(self):
        rows = [
            {"name": "A", "full_map50_95": 0.56, "end_to_end_mean_ms": 24.0, "model_mib": 3.0},
            {"name": "B", "full_map50_95": 0.55, "end_to_end_mean_ms": 25.0, "model_mib": 3.1},
        ]
        mark_pareto(rows)
        self.assertTrue(rows[0]["pareto"])
        self.assertFalse(rows[1]["pareto"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
