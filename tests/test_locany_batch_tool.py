from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from gpu_services import locateanything
from locany_batch_tool.server import (
    JOBS,
    BatchReq,
    _check_direct_capabilities,
    _remote_videos,
    _run_batch,
    _videos,
    app,
)


class LocateAnythingBatchToolTests(TestCase):
    def test_health(self) -> None:
        client = TestClient(app)
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "locateanything-batch-tool")
        self.assertIn("LocateAnything 批量预标注", client.get("/").text)

    def test_video_discovery_filters_extensions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.mp4").touch()
            (root / "b.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.mov").touch()
            self.assertEqual([path.name for path in _videos(str(root), False)], ["a.mp4"])
            self.assertEqual([path.name for path in _videos(str(root), True)], ["a.mp4", "c.mov"])

    def test_direct_mode_keeps_gpu_server_posix_paths_unchanged(self) -> None:
        remote_path = "/data2/DET_Group/ZZS/data/embedded_cosmos/videos/sample.mp4"
        with patch("locany_batch_tool.server._json_request", return_value={"videos": [remote_path]}) as request:
            videos = _remote_videos(
                "http://gpu-server:10114",
                "/data2/DET_Group/ZZS/data/embedded_cosmos/videos",
                False,
            )
        self.assertEqual(videos, [remote_path])
        self.assertIn("path=%2Fdata2%2FDET_Group", request.call_args.args[1])
        self.assertNotIn("D%3A", request.call_args.args[1])

    def test_direct_connection_rejects_old_gpu_service(self) -> None:
        with patch("locany_batch_tool.server._json_request", return_value={"paths": {}}):
            with self.assertRaisesRegex(RuntimeError, "版本过旧"):
                _check_direct_capabilities("http://gpu-server:10114", {})

    def test_direct_connection_requires_output_allowed_roots(self) -> None:
        openapi = {"paths": {"/api/locateanything/videos": {}}}
        with patch("locany_batch_tool.server._json_request", return_value=openapi):
            with self.assertRaisesRegex(RuntimeError, "LOCANY_OUTPUT_ALLOWED_ROOTS"):
                _check_direct_capabilities("http://gpu-server:10114", {})

    def test_direct_batch_accepts_remote_video_strings(self) -> None:
        remote_video = "/data2/videos/sample.mp4"
        request = BatchReq(
            server_url="http://gpu-server:10114",
            mode="direct",
            input_path="/data2/videos",
            output_path="/data2/labels",
        )
        job_id = "direct-test"
        JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued"}

        def fake_request(method, url, payload=None, timeout=30):
            if method == "POST":
                self.assertEqual(payload["video_path"], remote_video)
                return {"job_id": "remote-job"}
            return {"status": "done", "message": "Done", "direct_output_dir": "/data2/labels/sample"}

        with patch("locany_batch_tool.server._remote_videos", return_value=[remote_video]), patch(
            "locany_batch_tool.server._json_request", side_effect=fake_request
        ):
            _run_batch(job_id, request)

        self.assertEqual(JOBS[job_id]["status"], "done")
        self.assertEqual(JOBS[job_id]["items"][0]["output"], "/data2/labels/sample")

    def test_direct_output_must_be_inside_allowed_root(self) -> None:
        original = list(locateanything.SETTINGS.get("output_allowed_roots", []))
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                locateanything.SETTINGS["output_allowed_roots"] = [root]
                output = locateanything._resolve_output_dir(str(root / "job"))
                self.assertTrue(output.is_dir())
                with self.assertRaisesRegex(RuntimeError, "outside allowed roots"):
                    locateanything._resolve_output_dir(str(root.parent / "elsewhere"))
        finally:
            locateanything.SETTINGS["output_allowed_roots"] = original
