from __future__ import annotations

import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from utils.annotator.server import _gpu_api_url, _project_config_path, app


class AnnotatorGpuIntegrationTests(TestCase):
    def test_project_config_fallback_stays_at_repository_root(self) -> None:
        with patch.dict(os.environ, {"ANNOTATOR_CONFIG": ""}):
            self.assertEqual(
                _project_config_path(),
                Path(__file__).resolve().parents[1] / "config.json",
            )

    def test_workbench_only_builds_sam31_gpu_urls(self) -> None:
        self.assertEqual(
            _gpu_api_url("http://gpu-server:9010/", "sam31", "jobs/abc/tracking-results"),
            "http://gpu-server:9010/api/sam31/jobs/abc/tracking-results",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported GPU service"):
            _gpu_api_url("http://gpu-server:9010", "locateanything", "jobs")

    def test_workbench_does_not_expose_locateanything_routes(self) -> None:
        paths = TestClient(app).get("/openapi.json").json()["paths"]
        self.assertFalse(any(path.startswith("/api/locateanything") for path in paths))
