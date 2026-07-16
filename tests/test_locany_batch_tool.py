from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from gpu_services import locateanything
from locany_batch_tool.server import (
    JOBS,
    BatchReq,
    _check_direct_capabilities,
    _images,
    _selected_cuda_devices,
    parse_cuda_devices,
    _remote_videos,
    _run_batch,
    _videos,
    app,
)


class LocateAnythingBatchToolTests(TestCase):
    def test_cuda_device_list_parser_and_legacy_single_device(self) -> None:
        self.assertEqual(parse_cuda_devices("0, cuda:2，2,5"), [0, 2, 5])
        with self.assertRaisesRegex(ValueError, "无效"):
            parse_cuda_devices("0,x")
        legacy = BatchReq(
            server_url="http://gpu-server:10114", mode="direct",
            input_path="/videos", output_path="/labels", cuda_device=3,
        )
        self.assertEqual(_selected_cuda_devices(legacy), [3])

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

    def test_image_discovery_requires_directory_and_preserves_recursive_choice(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").touch()
            (root / "ignored.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.PNG").touch()
            self.assertEqual([path.name for path in _images(str(root), False)], ["a.jpg"])
            self.assertEqual([path.name for path in _images(str(root), True)], ["a.jpg", "b.PNG"])
            with self.assertRaisesRegex(RuntimeError, "must be a directory"):
                _images(str(root / "a.jpg"), False)

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
            with self.assertRaisesRegex(RuntimeError, "too old"):
                _check_direct_capabilities("http://gpu-server:10114", {})

    def test_direct_connection_requires_output_allowed_roots(self) -> None:
        openapi = {"paths": {"/api/locateanything/jobs": {}, "/api/locateanything/videos": {}}}
        with patch("locany_batch_tool.server._json_request", return_value=openapi):
            with self.assertRaisesRegex(RuntimeError, "LOCANY_OUTPUT_ALLOWED_ROOTS"):
                _check_direct_capabilities("http://gpu-server:10114", {})

    def test_direct_image_mode_requires_image_job_api(self) -> None:
        openapi = {"paths": {"/api/locateanything/jobs": {}, "/api/locateanything/videos": {}}}
        with patch("locany_batch_tool.server._json_request", return_value=openapi):
            with self.assertRaisesRegex(RuntimeError, "image-jobs"):
                _check_direct_capabilities("http://gpu-server:10114", {"output_allowed_roots": ["/data2"]}, "images")

    def test_direct_image_directory_batch_uses_separate_api(self) -> None:
        request = BatchReq(
            server_url="http://gpu-server:10114", mode="direct", task_kind="images",
            input_path="/data2/images", output_path="/data2/output",
            cuda_devices=[1], recursive=True, max_images=25,
        )
        job_id = "direct-image-test"
        JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued"}
        calls = []

        def fake_request(method, url, payload=None, timeout=30):
            calls.append((method, url, payload))
            if url.endswith("/api/locateanything/health"):
                return {"devices": ["cuda:0", "cuda:1"], "output_allowed_roots": ["/data2"]}
            if url.endswith("/openapi.json"):
                return {"paths": {"/api/locateanything/image-jobs": {}}}
            if method == "POST":
                self.assertEqual(payload["input_dir"], "/data2/images")
                self.assertEqual(payload["output_dir"], "/data2/output")
                self.assertEqual(payload["device"], "cuda:1")
                self.assertEqual(payload["max_images"], 25)
                return {"job_id": "image-job"}
            return {"status": "done", "message": "Done", "direct_output_dir": "/data2/output"}

        with patch("locany_batch_tool.server._json_request", side_effect=fake_request):
            _run_batch(job_id, request)

        self.assertEqual(JOBS[job_id]["status"], "done")
        self.assertEqual(JOBS[job_id]["items"][0]["output"], "/data2/output")
        self.assertTrue(any(url.endswith("/api/locateanything/image-jobs") for _, url, _ in calls))

    def test_image_directory_batch_rejects_multiple_gpus(self) -> None:
        request = BatchReq(
            server_url="http://gpu-server:10114", mode="direct", task_kind="images",
            input_path="/data2/images", output_path="/data2/output", cuda_devices=[0, 1],
        )
        job_id = "image-multi-gpu-test"
        JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued"}
        _run_batch(job_id, request)
        self.assertEqual(JOBS[job_id]["status"], "failed")
        self.assertIn("exactly one CUDA device", JOBS[job_id]["message"])

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
            if url.endswith("/api/locateanything/health"):
                return {"devices": ["cuda:0", "cuda:1"], "parallel_jobs": True}
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

    def test_direct_batch_runs_one_worker_per_selected_gpu(self) -> None:
        videos = [f"/data2/videos/v{index}.mp4" for index in range(4)]
        request = BatchReq(
            server_url="http://gpu-server:10114", mode="direct",
            input_path="/data2/videos", output_path="/data2/labels",
            cuda_devices=[0, 1],
        )
        job_id = "multi-gpu-test"
        JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued"}
        first_pair = threading.Barrier(2, timeout=2)
        payloads = []
        payload_lock = threading.Lock()

        def fake_request(method, url, payload=None, timeout=30):
            if url.endswith("/api/locateanything/health"):
                return {"devices": ["cuda:0", "cuda:1"], "parallel_jobs": True}
            if method == "POST":
                with payload_lock:
                    payloads.append(dict(payload))
                    call_number = len(payloads)
                if call_number <= 2:
                    first_pair.wait()
                return {"job_id": Path(payload["video_path"]).stem}
            remote_id = url.rsplit("/", 1)[-1]
            return {
                "status": "done", "message": "Done",
                "assigned_device": next(
                    item["device"] for item in payloads if Path(item["video_path"]).stem == remote_id
                ),
                "direct_output_dir": f"/data2/labels/{remote_id}",
            }

        with patch("locany_batch_tool.server._remote_videos", return_value=videos), patch(
            "locany_batch_tool.server._json_request", side_effect=fake_request
        ):
            _run_batch(job_id, request)

        job = JOBS[job_id]
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["completed"], 4)
        self.assertEqual(job["cuda_devices"], [0, 1])
        self.assertEqual({payload["device"] for payload in payloads}, {"cuda:0", "cuda:1"})
        self.assertTrue(all(item["status"] == "done" for item in job["items"]))

    def test_batch_rejects_devices_not_enabled_by_gpu_service(self) -> None:
        request = BatchReq(
            server_url="http://gpu-server:10114", mode="direct",
            input_path="/data2/videos", output_path="/data2/labels",
            cuda_devices=[0, 2],
        )
        job_id = "disabled-gpu-test"
        JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued"}
        with patch("locany_batch_tool.server._remote_videos", return_value=["/data2/videos/v.mp4"]), patch(
            "locany_batch_tool.server._json_request",
            return_value={"devices": ["cuda:0", "cuda:1"], "parallel_jobs": True},
        ):
            _run_batch(job_id, request)
        self.assertEqual(JOBS[job_id]["status"], "failed")
        self.assertIn("cuda:2", JOBS[job_id]["message"])

    def test_multi_gpu_batch_rejects_old_serial_gpu_service(self) -> None:
        request = BatchReq(
            server_url="http://gpu-server:10114", mode="direct",
            input_path="/data2/videos", output_path="/data2/labels",
            cuda_devices=[0, 1],
        )
        job_id = "old-serial-service-test"
        JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued"}
        with patch("locany_batch_tool.server._remote_videos", return_value=["/data2/videos/v.mp4"]), patch(
            "locany_batch_tool.server._json_request", return_value={"device": "cuda:0"}
        ):
            _run_batch(job_id, request)
        self.assertEqual(JOBS[job_id]["status"], "failed")
        self.assertIn("版本过旧", JOBS[job_id]["message"])
        self.assertIn("Waiting for previous", JOBS[job_id]["message"])

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
