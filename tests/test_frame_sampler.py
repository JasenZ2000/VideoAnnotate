from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.frame_sampler.logic import (
    SamplingPlan,
    SamplingSegment,
    build_sampled_frame_indices,
    export_sampled_yolo_dataset,
    validate_sampling_plan,
)


class FrameSamplerTests(unittest.TestCase):
    def test_build_sampled_frame_indices_uses_default_and_segment_overrides(self) -> None:
        plan = SamplingPlan(
            default_interval=5,
            include_empty_frames=True,
            file_prefix="frame",
            segments=[
                SamplingSegment(start_frame=2, end_frame=8, interval=2, label="dense"),
                SamplingSegment(start_frame=15, end_frame=19, interval=3, label="sparse-end"),
            ],
        )
        frames = build_sampled_frame_indices(20, plan)
        self.assertEqual(frames, [0, 2, 4, 6, 8, 10, 15, 18, 19])

    def test_validate_sampling_plan_rejects_overlaps(self) -> None:
        plan = SamplingPlan(
            default_interval=0,
            include_empty_frames=True,
            file_prefix="frame",
            segments=[
                SamplingSegment(start_frame=1, end_frame=4, interval=1),
                SamplingSegment(start_frame=4, end_frame=6, interval=1),
            ],
        )
        with self.assertRaises(ValueError):
            validate_sampling_plan(plan, 10)

    def test_export_sampled_yolo_dataset_writes_images_labels_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "demo.avi"
            tracking_path = root / "tracking_results.json"
            output_dir = root / "out"

            self._write_demo_video(video_path)
            tracking_path.write_text(
                """
{
  "metadata": {
    "video_path": "demo.avi",
    "fps": 10,
    "width": 64,
    "height": 48,
    "frame_count": 6,
    "frame_offset": 0,
    "num_tracks": 1
  },
  "tracks": [
    {
      "track_id": 1,
      "class_id": 2,
      "frames": [
        { "frame_id": 0, "video_frame_idx": 0, "bbox_xyxy": [16, 12, 32, 24] },
        { "frame_id": 2, "video_frame_idx": 2, "bbox_xyxy": [8, 8, 24, 20] },
        { "frame_id": 5, "video_frame_idx": 5, "bbox_xyxy": [20, 10, 44, 30] }
      ]
    }
  ]
}
                """.strip(),
                encoding="utf-8",
            )

            plan = SamplingPlan(
                default_interval=0,
                include_empty_frames=True,
                file_prefix="frame",
                segments=[SamplingSegment(start_frame=0, end_frame=5, interval=2, label="dense")],
            )
            result = export_sampled_yolo_dataset(video_path, tracking_path, output_dir, plan)

            self.assertEqual(result["selected_frames"], 4)
            self.assertEqual(result["frames_exported"], 4)
            self.assertTrue((output_dir / "images" / "frame_000000.jpg").exists())
            self.assertTrue((output_dir / "images" / "frame_000005.jpg").exists())
            self.assertTrue((output_dir / "labels" / "frame_000004.txt").exists())
            self.assertEqual((output_dir / "labels" / "frame_000004.txt").read_text(encoding="utf-8"), "")
            self.assertEqual(
                (output_dir / "labels" / "frame_000000.txt").read_text(encoding="utf-8"),
                "2 0.375000 0.375000 0.250000 0.250000\n",
            )
            self.assertTrue((output_dir / "sampling_plan.json").exists())

    def _write_demo_video(self, video_path: Path) -> None:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened(), "Failed to create demo video")
        try:
            for idx in range(6):
                frame = np.full((48, 64, 3), idx * 30, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
