from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from utils.annotator.export_formats import write_jpeg, write_voc_xml
from utils.annotator import server as annotator_server


class AnnotatorExportFormatTests(unittest.TestCase):
    def tearDown(self) -> None:
        if annotator_server._cap is not None:
            annotator_server._cap.release()
        annotator_server._cap = None
        annotator_server._cap_pos = -1
        annotator_server._workspace = None
        annotator_server.STATE.clear_tracks()

    def test_write_jpeg_supports_unicode_windows_style_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "中文工作区" / "images" / "frame_000000.jpg"
            frame = np.full((24, 32, 3), 127, dtype=np.uint8)

            write_jpeg(path, frame)

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (24, 32))

    def test_write_voc_xml_uses_configured_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations" / "frame_000000.xml"
            write_voc_xml(
                path,
                "frame_000000.jpg",
                640,
                480,
                [(0, [10, 20, 110, 220]), (7, [0, 0, 640, 480])],
                {0: "person"},
            )

            root = ET.parse(path).getroot()
            self.assertEqual(root.findtext("filename"), "frame_000000.jpg")
            self.assertEqual([node.findtext("name") for node in root.findall("object")], ["person", "7"])
            self.assertEqual(root.findall("object")[0].findtext("bndbox/xmax"), "110")
            self.assertEqual(root.findall("object")[1].findtext("truncated"), "1")

    def test_export_endpoint_writes_matching_images_labels_and_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            video_path = workspace / "demo.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            try:
                writer.write(np.full((24, 32, 3), 20, dtype=np.uint8))
                writer.write(np.full((24, 32, 3), 80, dtype=np.uint8))
            finally:
                writer.release()

            client = TestClient(annotator_server.app)
            opened = client.post("/api/open-workspace", json={"path": str(workspace)})
            self.assertEqual(opened.status_code, 200, opened.text)
            saved = client.post(
                "/api/annotation",
                json={"track_id": 1, "frame_idx": 0, "bbox": [4, 5, 20, 18]},
            )
            self.assertEqual(saved.status_code, 200, saved.text)

            response = client.post("/api/export-yolo", json={"interval": 1})
            self.assertEqual(response.status_code, 200, response.text)
            output = workspace / "yoloset"
            self.assertEqual(len(list((output / "images").glob("*.jpg"))), 2)
            self.assertEqual(len(list((output / "labels").glob("*.txt"))), 2)
            self.assertEqual(len(list((output / "annotations").glob("*.xml"))), 2)
            expected_stems = {"demo_frame_000000", "demo_frame_000001"}
            self.assertEqual({path.stem for path in (output / "images").glob("*.jpg")}, expected_stems)
            self.assertEqual({path.stem for path in (output / "labels").glob("*.txt")}, expected_stems)
            self.assertEqual({path.stem for path in (output / "annotations").glob("*.xml")}, expected_stems)
            xml = ET.parse(output / "annotations" / "demo_frame_000000.xml").getroot()
            self.assertEqual(xml.findtext("filename"), "demo_frame_000000.jpg")
            self.assertEqual(xml.findtext("object/name"), "person")
            annotator_server._cap.release()
            annotator_server._cap = None


if __name__ == "__main__":
    unittest.main()
