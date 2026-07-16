from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from utils.mot_pipeline.models import FinalTrack
from utils.mot_pipeline import pipeline


class MotPipelineRenderingTests(unittest.TestCase):
    def test_pipeline_can_skip_all_video_rendering(self) -> None:
        tracker = Mock(return_value=[])
        fusion = Mock(return_value=[])
        final_track = FinalTrack(
            track_id=1,
            class_id=0,
            frames={0: [1.0, 2.0, 10.0, 20.0]},
            video_frames={0: 0},
        )
        config = {
            "tracking": {"method": "test", "disable_kalman": True, "iou_match": 0.1, "max_missed": 1},
            "fusion": {"method": "test", "iou_fuse": 0.1, "min_track_len": 1, "smooth_window": 1},
            "clips": {
                "pad_frames": 0, "crop_margin": 1.0, "crop_min_size": 2,
                "overview_filename": "overview.mp4", "codec": "mp4v",
                "overview_box_thickness": 1, "overview_font_scale": 0.5,
            },
            "exports": {"export_yolo_from_tracking": False, "export_label_studio": False},
            "quality_control": {},
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(pipeline, "TRACKER_REGISTRY", {"test": tracker}),
                patch.object(pipeline, "FUSION_REGISTRY", {"test": fusion}),
                patch.object(pipeline, "get_video_metadata", return_value=(100, 80, 10, 25.0)),
                patch.object(pipeline, "load_annotations", return_value=([object()], 0, [])),
                patch.object(pipeline, "build_final_tracks", return_value=[final_track]),
                patch.object(pipeline, "write_tracking_outputs", return_value=(root / "tracking_results.json", root / "tracking_results.csv")),
                patch.object(pipeline, "prepare_track_clips") as prepare,
                patch.object(pipeline, "render_tracking_overview") as overview,
                patch.object(pipeline, "extract_track_clips") as clips,
            ):
                pipeline.run_pipeline(root / "video.mp4", root / "labels", root, config, render_videos=False)

            prepare.assert_not_called()
            overview.assert_not_called()
            clips.assert_not_called()


if __name__ == "__main__":
    unittest.main()
