from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from local_workbench.server import app


class LocalWorkbenchTests(unittest.TestCase):
    def test_health_lists_both_local_apps(self) -> None:
        response = TestClient(app).get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["apps"], {
            "annotator": "/annotator/",
            "frame_sampler": "/sampler/",
        })

    def test_root_redirects_to_annotator(self) -> None:
        response = TestClient(app).get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/annotator/")

    def test_both_mounted_apps_serve_their_html(self) -> None:
        client = TestClient(app)
        self.assertIn("Video Annotator", client.get("/annotator/").text)
        self.assertIn("Training Frame Sampler", client.get("/sampler/").text)


if __name__ == "__main__":
    unittest.main()
