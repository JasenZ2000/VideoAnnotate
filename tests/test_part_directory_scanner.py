from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflow_platform.part_directory_scanner import scan_part_directories


class PartDirectoryScannerTests(unittest.TestCase):
    def test_detects_work_directories_and_stops_before_marker_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "split_001" / "images").mkdir(parents=True)
            (root / "split_001" / "images" / "nested").mkdir()
            (root / "camera_a" / "split_002" / "labels").mkdir(parents=True)
            (root / "camera_a" / "split_002" / "annotations").mkdir()
            (root / "not_a_part" / "nested").mkdir(parents=True)

            result = scan_part_directories(root, max_depth=4)

            self.assertEqual(result.manifest_lines(), ["camera_a/split_002", "split_001"])
            self.assertEqual([item.depth for item in result.items], [2, 1])
            self.assertNotIn("split_001/images", result.manifest_lines())

    def test_depth_and_minimum_marker_count_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "level1" / "level2" / "split"
            (target / "images").mkdir(parents=True)
            self.assertEqual(scan_part_directories(root, max_depth=2).items, [])
            self.assertEqual(len(scan_part_directories(root, max_depth=3).items), 1)
            self.assertEqual(
                scan_part_directories(root, max_depth=3, minimum_marker_count=2).items,
                [],
            )

    def test_root_itself_can_be_a_work_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            result = scan_part_directories(root, max_depth=0)
            self.assertEqual(result.manifest_lines(), ["."])
            self.assertEqual(result.manifest_lines(full_paths=True), [str(root.resolve())])


if __name__ == "__main__":
    unittest.main()
