from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "linux" / "benchmark_locateanything.py"
SPEC = importlib.util.spec_from_file_location("benchmark_locateanything", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LocateAnythingPerformanceBenchmarkTests(TestCase):
    def test_parses_standard_and_batch_cases(self) -> None:
        standard = MODULE.parse_case("standard-slow")
        batch = MODULE.parse_case("batch-hybrid-4")
        self.assertEqual((standard.runtime, standard.generation_mode, standard.batch_size), ("standard", "slow", 1))
        self.assertEqual((batch.runtime, batch.generation_mode, batch.batch_size), ("batch", "hybrid", 4))

    def test_parses_labeled_boxes_in_original_image_coordinates(self) -> None:
        boxes = MODULE.parse_boxes(
            "<ref>person</ref><box><100><200><500><800></box><box><600><100><900><400></box>",
            2000,
            1000,
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0]["label"], "person")
        self.assertEqual(boxes[0]["bbox_xyxy"], [200.0, 200.0, 1000.0, 800.0])
        self.assertEqual(boxes[1]["label"], "person")

    def test_compares_boxes_greedily_at_iou_threshold(self) -> None:
        reference = [{"label": "person", "bbox_xyxy": [0, 0, 100, 100]}]
        candidate = [
            {"label": "person", "bbox_xyxy": [5, 5, 95, 95]},
            {"label": "car", "bbox_xyxy": [0, 0, 100, 100]},
        ]
        self.assertEqual(MODULE.compare_box_sets(reference, candidate), (1, 1, 0))

    def test_detects_cuda_oom_without_misclassifying_other_errors(self) -> None:
        self.assertTrue(MODULE.is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(MODULE.is_cuda_oom(RuntimeError("disk out of memory")))

    def test_defaults_standard_attention_to_sdpa(self) -> None:
        args = MODULE.build_parser().parse_args([])
        self.assertEqual(args.standard_attn, "sdpa")

    def test_pins_attention_on_nested_composite_config(self) -> None:
        text_config = SimpleNamespace()
        config = SimpleNamespace(text_config=text_config)
        MODULE._set_config_attention(config, "text_config", "sdpa")
        self.assertEqual(text_config._attn_implementation, "sdpa")
        self.assertEqual(text_config._attn_implementation_internal, "sdpa")
        self.assertTrue(text_config._attn_implementation_autoset)

    def test_attention_snapshot_reads_language_model_core(self) -> None:
        config = SimpleNamespace(
            _attn_implementation="magi",
            text_config=SimpleNamespace(_attn_implementation="sdpa"),
            vision_config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        )
        worker = SimpleNamespace(
            model=SimpleNamespace(
                config=config,
                language_model=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="sdpa"),
                    model=SimpleNamespace(_attn_implementation="sdpa"),
                ),
            )
        )
        snapshot = MODULE._standard_attention_snapshot(worker)
        self.assertEqual(snapshot["language_model.model"], "sdpa")
