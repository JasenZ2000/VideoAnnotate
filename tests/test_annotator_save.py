from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from utils.annotator import server
from utils.annotator.state import AnnotationState


class AnnotatorSaveTests(unittest.TestCase):
    def test_save_updates_results_loaded_on_next_open(self) -> None:
        state = AnnotationState()
        state.video_path = "sample.mp4"
        state.width = 640
        state.height = 480
        state.frame_count = 20
        state.fps = 25.0
        track_id = state.add_track(class_id=3)
        state.set_frame(track_id, 7, [10, 20, 100, 200])

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            anomaly_path = workspace / "area_anomalies.json"
            with (
                patch.object(server, "STATE", state),
                patch.object(server, "_workspace", workspace),
                patch.object(server, "_write_area_anomaly_json", return_value=anomaly_path),
            ):
                response = TestClient(server.app).post("/api/save")

            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue((workspace / "annotation_project.json").is_file())
            results_path = workspace / "tracking_results.json"
            self.assertTrue(results_path.is_file())
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["num_tracks"], 1)
            self.assertEqual(payload["tracks"][0]["frames"][0]["video_frame_idx"], 7)


if __name__ == "__main__":
    unittest.main()
