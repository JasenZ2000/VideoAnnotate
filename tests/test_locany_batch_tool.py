from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import json
import xml.etree.ElementTree as ET
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import cv2

from fastapi.testclient import TestClient

from gpu_services import locateanything
from locany_batch_tool.frame_extract import extract_video_frames
from locany_batch_tool.workflow_defaults import build_workflow_defaults, discover_workflow_workspaces
from locany_batch_tool.server import (
    JOBS,
    AdaptiveSampleReq,
    BatchReq,
    MotFilterReq,
    _check_direct_capabilities,
    _images,
    _selected_cuda_devices,
    parse_cuda_devices,
    run_adaptive_sample,
    run_mot_filter,
    _remote_videos,
    _run_batch,
    _videos,
    app,
)


class LocateAnythingBatchToolTests(TestCase):
    def test_workflow_defaults_run_frame_mot_voc_and_sampling_without_path_edits(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory) / "sample"
            workspace.mkdir()
            video = workspace / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(3):
                writer.write(np.full((24, 32, 3), index * 50, dtype=np.uint8))
            writer.release()
            raw_labels = workspace / "sample"
            raw_labels.mkdir()
            for frame_id in range(1, 4):
                (raw_labels / f"sample_{frame_id}.txt").write_text(
                    "0 0.5 0.5 0.25 0.5 0.9\n", encoding="utf-8"
                )
            (workspace / "metadata.json").write_text(json.dumps({
                "width": 32, "height": 24, "class_map": {"person": 0},
            }), encoding="utf-8")

            self.assertEqual(discover_workflow_workspaces(workspace.parent), [workspace.resolve()])
            defaults = build_workflow_defaults(workspace)
            frames = extract_video_frames(
                defaults["video_path"], output_dir=defaults["frame_output_dir"]
            )
            mot = run_mot_filter(MotFilterReq(
                input_dir=defaults["mot_input_dir"],
                output_dir=defaults["mot_output_dir"],
                metadata_path=defaults["metadata_path"],
                voc_output_dir=defaults["mot_voc_output_dir"],
                output_voc=True,
                min_track_len=2,
            ))
            sampled = run_adaptive_sample(AdaptiveSampleReq(
                dataset_dir=defaults["sample_dataset_dir"],
                output_dir=defaults["sample_output_dir"],
                mode="uniform",
                intervals=(1, 1, 1, 1),
            ))

            self.assertEqual(frames["frames_written"], 3)
            self.assertEqual(mot["voc_files"], 3)
            self.assertEqual(sampled["metadata"]["selected_frames"], 3)
            self.assertEqual(sorted(path.name for path in (workspace / "dataset" / "images").glob("*.jpg")), ["sample_1.jpg", "sample_2.jpg", "sample_3.jpg"])
            self.assertTrue((workspace / "dataset" / "labels" / "sample_1.txt").is_file())
            self.assertTrue((workspace / "dataset" / "annotations" / "sample_1.xml").is_file())
            self.assertTrue((workspace / "dataset_sampled" / "images" / "sample_1.jpg").is_file())
            self.assertTrue((workspace / "dataset_sampled" / "labels" / "sample_1.txt").is_file())
            self.assertTrue((workspace / "dataset_sampled" / "annotations" / "sample_1.xml").is_file())

    def test_frame_extract_writes_interval_frames_beside_video(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(5):
                writer.write(np.full((24, 32, 3), index * 40, dtype=np.uint8))
            writer.release()

            result = extract_video_frames(video, frame_interval=2)

            self.assertEqual(result["frames_read"], 5)
            self.assertEqual(result["frames_written"], 3)
            self.assertEqual(Path(result["output_dir"]), root.resolve())
            self.assertEqual(
                sorted((path.name for path in root.glob("sample_*.jpg")), key=lambda name: int(Path(name).stem.rsplit("_", 1)[1])),
                ["sample_1.jpg", "sample_3.jpg", "sample_5.jpg"],
            )
            with self.assertRaisesRegex(RuntimeError, "already exist"):
                extract_video_frames(video, frame_interval=2)

    def test_adaptive_sampling_copies_selected_triplets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "sampled"
            for folder in ("images", "labels", "annotations"):
                (dataset / folder).mkdir(parents=True)
            for frame_id in range(1, 6):
                cv2.imwrite(str(dataset / "images" / f"sample_{frame_id}.jpg"), np.full((24, 32, 3), frame_id * 20, dtype=np.uint8))
                (dataset / "labels" / f"sample_{frame_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                (dataset / "annotations" / f"sample_{frame_id}.xml").write_text("<annotation/>", encoding="utf-8")

            result = run_adaptive_sample(AdaptiveSampleReq(
                dataset_dir=str(dataset), output_dir=str(output), mode="uniform", intervals=(2, 2, 2, 2)
            ))

            self.assertEqual(result["metadata"]["selected_frames"], 3)
            self.assertEqual(sorted(path.name for path in (output / "images").iterdir()), ["sample_1.jpg", "sample_3.jpg", "sample_5.jpg"])
            self.assertEqual(sorted(path.name for path in (output / "labels").iterdir()), ["sample_1.txt", "sample_3.txt", "sample_5.txt"])
            self.assertEqual(sorted(path.name for path in (output / "annotations").iterdir()), ["sample_1.xml", "sample_3.xml", "sample_5.xml"])
            self.assertTrue((output / "sampling_report.json").is_file())

    def test_adaptive_sampling_preview_does_not_create_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "images").mkdir(parents=True)
            cv2.imwrite(str(dataset / "images" / "sample_1.jpg"), np.zeros((24, 32, 3), dtype=np.uint8))
            output = root / "output"
            result = run_adaptive_sample(AdaptiveSampleReq(
                dataset_dir=str(dataset), output_dir=str(output), mode="uniform", dry_run=True
            ))
            self.assertTrue(result["dry_run"])
            self.assertFalse(output.exists())

    def test_mot_filter_keeps_long_tracks_and_removes_isolated_detections(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "labels"
            output = root / "filtered"
            source.mkdir()
            for frame_idx in range(12):
                lines = [f"0 {0.3 + frame_idx * 0.002:.6f} 0.500000 0.100000 0.200000 0.900000"]
                if frame_idx == 5:
                    lines.append("0 0.850000 0.200000 0.050000 0.050000 0.950000")
                # Match LocateAnything: one-based frame id without zero padding.
                (source / f"sample_{frame_idx + 1}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = run_mot_filter(MotFilterReq(input_dir=str(source), output_dir=str(output)))

            self.assertEqual(result["input_detections"], 13)
            self.assertEqual(result["output_detections"], 12)
            self.assertEqual(result["removed_detections"], 1)
            self.assertEqual(result["min_track_len"], 10)
            self.assertEqual(len((output / "sample_6.txt").read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len((output / "sample_6.txt").read_text(encoding="utf-8").split()[0:]), 6)
            self.assertEqual(
                (output / "sample_1.txt").read_text(encoding="utf-8"),
                (source / "sample_1.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                {path.name for path in output.glob("*.txt")},
                {path.name for path in source.glob("*.txt")},
            )

    def test_mot_filter_optionally_writes_matching_voc_xml_from_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "labels"
            output = root / "labels_filtered"
            source.mkdir()
            (root / "metadata.json").write_text(json.dumps({
                "width": 640,
                "height": 480,
                "class_map": {"person": 0},
            }), encoding="utf-8")
            for frame_id in range(1, 4):
                lines = [f"0 {0.4 + frame_id * 0.001:.6f} 0.5 0.2 0.4 0.9"]
                if frame_id == 2:
                    lines.append("0 0.9 0.1 0.05 0.05 0.8")
                (source / f"sample_{frame_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = run_mot_filter(MotFilterReq(
                input_dir=str(source),
                output_dir=str(output),
                min_track_len=2,
                output_voc=True,
            ))

            annotations = root / "annotations_filtered"
            self.assertEqual(result["voc_output_dir"], str(annotations.resolve()))
            self.assertEqual(result["voc_files"], 3)
            xml_path = annotations / "sample_2.xml"
            xml = ET.parse(xml_path).getroot()
            self.assertEqual(xml.findtext("filename"), "sample_2.jpg")
            self.assertEqual(xml.findtext("size/width"), "640")
            self.assertEqual(xml.findtext("size/height"), "480")
            self.assertEqual([item.findtext("name") for item in xml.findall("object")], ["person"])
            self.assertEqual(len((output / "sample_2.txt").read_text(encoding="utf-8").splitlines()), 1)

    def test_mot_filter_rejects_output_with_existing_txt_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "labels"
            output = root / "filtered"
            source.mkdir()
            output.mkdir()
            (source / "sample_0.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (output / "old.txt").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already contains TXT"):
                run_mot_filter(MotFilterReq(input_dir=str(source), output_dir=str(output), min_track_len=1))

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
        self.assertIn("MOT 轨迹过滤", client.get("/").text)
        self.assertIn("/api/mot-filter", {route.path for route in app.routes})
        self.assertIn("自适应筛帧", client.get("/").text)
        self.assertIn("/api/adaptive-sample", {route.path for route in app.routes})
        self.assertNotIn("/api/deduplicate-dataset", {route.path for route in app.routes})
        self.assertIn("本地视频抽帧", client.get("/").text)
        self.assertIn("/api/extract-frames", {route.path for route in app.routes})
        self.assertIn("GPU 预标注", client.get("/").text)
        self.assertIn("标注处理", client.get("/").text)
        self.assertIn("数据集工具", client.get("/").text)

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
                return {
                    "devices": ["cuda:0", "cuda:1"], "output_allowed_roots": ["/data2"],
                    "runtime": "batch", "generation_mode": "hybrid", "batch_size": 4,
                    "batch_runtime_supported": True, "batch_utils_available": True,
                    "kernel_utils_available": True,
                }
            if url.endswith("/openapi.json"):
                return {"paths": {"/api/locateanything/image-jobs": {}}}
            if method == "POST":
                self.assertEqual(payload["input_dir"], "/data2/images")
                self.assertEqual(payload["output_dir"], "/data2/output")
                self.assertEqual(payload["device"], "cuda:1")
                self.assertEqual(payload["max_images"], 25)
                self.assertNotIn("generation_mode", payload)
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
                return {
                    "devices": ["cuda:0", "cuda:1"], "parallel_jobs": True,
                    "runtime": "batch", "generation_mode": "hybrid", "batch_size": 4,
                    "batch_runtime_supported": True, "batch_utils_available": True,
                    "kernel_utils_available": True,
                }
            if method == "POST":
                self.assertEqual(payload["video_path"], remote_video)
                self.assertNotIn("generation_mode", payload)
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
                return {
                    "devices": ["cuda:0", "cuda:1"], "parallel_jobs": True,
                    "runtime": "batch", "generation_mode": "hybrid", "batch_size": 4,
                    "batch_runtime_supported": True, "batch_utils_available": True,
                    "kernel_utils_available": True,
                }
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
            return_value={
                "devices": ["cuda:0", "cuda:1"], "parallel_jobs": True,
                "runtime": "batch", "generation_mode": "hybrid", "batch_size": 4,
                "batch_runtime_supported": True, "batch_utils_available": True,
                "kernel_utils_available": True,
            },
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
