from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi.testclient import TestClient

from gpu_services import locateanything, sam31
from gpu_services.server import app


class GpuServicesTests(unittest.TestCase):
    def test_unified_api_exposes_separate_service_namespaces(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertTrue({
            "/api/health",
            "/api/sam31/jobs",
            "/api/sam31/jobs/{job_id}",
            "/api/sam31/jobs/{job_id}/tracking-results",
            "/api/locateanything/jobs",
            "/api/locateanything/videos",
            "/api/locateanything/jobs/{job_id}",
            "/api/locateanything/jobs/{job_id}/yolo-zip",
        }.issubset(paths))

    def test_sam31_runner_is_co_located_with_the_service(self) -> None:
        self.assertTrue(Path(sam31.DEFAULT_RUNNER).is_file())

    def test_root_health_reports_both_runtimes(self) -> None:
        response = TestClient(app).get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "video-annotation-gpu")
        self.assertIn("sam31", payload)
        self.assertIn("locateanything", payload)

    def test_locateanything_external_root_must_contain_worker(self) -> None:
        original = locateanything.SETTINGS["external_root"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locateanything.SETTINGS["external_root"] = str(root)
            with self.assertRaisesRegex(RuntimeError, "locateanything_worker.py"):
                locateanything._external_root()
            (root / "locateanything_worker.py").write_text("# external worker\n", encoding="utf-8")
            self.assertEqual(locateanything._external_root(), root.resolve())
        locateanything.SETTINGS["external_root"] = original

    def test_locateanything_writes_pascal_voc_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "annotations" / "sample_1.xml"
            locateanything._write_voc_xml(
                output,
                "sample_1.jpg",
                1920,
                1080,
                [{"label": "person", "bbox_xyxy": [214.4, 375.2, 533.1, 893.0]}],
            )
            root = ET.parse(output).getroot()
            self.assertEqual(root.findtext("filename"), "sample_1.jpg")
            self.assertEqual(root.findtext("size/width"), "1920")
            self.assertEqual(root.findtext("object/name"), "person")
            self.assertEqual(root.findtext("object/bndbox/xmin"), "214")
            self.assertEqual(root.findtext("object/bndbox/ymax"), "893")

    def test_locateanything_writes_empty_pascal_voc_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "annotations" / "sample_2.xml"
            locateanything._write_voc_xml(output, "sample_2.jpg", 640, 480, [])
            root = ET.parse(output).getroot()
            self.assertEqual(root.findall("object"), [])


if __name__ == "__main__":
    unittest.main()
