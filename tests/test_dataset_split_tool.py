from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_split_tool.core import scan_samples, split_dataset


class DatasetSplitToolTests(unittest.TestCase):
    def _sample(self, root: Path, dataset: str, stem: str) -> None:
        base = root / dataset
        for name in ("images", "labels", "annotations"):
            (base / name).mkdir(parents=True, exist_ok=True)
        (base / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (base / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
        (base / "annotations" / f"{stem}.xml").write_text("<annotation/>", encoding="utf-8")

    def test_scan_and_split_preserves_triplet_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ori = root / "ori"
            for index in range(5):
                self._sample(ori, f"dataset_{index % 2}", f"video_frame_{index:06d}")

            scan = scan_samples(ori)
            self.assertEqual(len(scan.dataset_roots), 2)
            self.assertEqual(len(scan.samples), 5)

            output = root / "split_output"
            result = split_dataset(ori, output, 2)
            self.assertEqual(result["counts"], [3, 2])
            for part in (output / "part_001", output / "part_002"):
                image_stems = {path.stem for path in (part / "images").iterdir()}
                label_stems = {path.stem for path in (part / "labels").iterdir()}
                xml_stems = {path.stem for path in (part / "annotations").iterdir()}
                self.assertEqual(image_stems, label_stems)
                self.assertEqual(image_stems, xml_stems)

    def test_missing_annotation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ori = Path(temporary) / "ori"
            self._sample(ori, "dataset", "sample")
            (ori / "dataset" / "annotations" / "sample.xml").unlink()
            with self.assertRaisesRegex(ValueError, "annotation XML"):
                scan_samples(ori)

    def test_duplicate_stem_across_datasets_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ori = Path(temporary) / "ori"
            self._sample(ori, "one", "sample")
            self._sample(ori, "two", "sample")
            with self.assertRaisesRegex(ValueError, "重复 stem"):
                scan_samples(ori)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ori = root / "ori"
            self._sample(ori, "dataset", "sample")
            output = root / "split_output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "输出目录已存在"):
                split_dataset(ori, output, 1)


if __name__ == "__main__":
    unittest.main()
