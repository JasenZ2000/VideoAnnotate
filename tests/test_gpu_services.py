from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
