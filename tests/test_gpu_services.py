from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from gpu_services import locateanything, sam31
from gpu_services.device_pool import parse_devices
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
            "/api/locateanything/image-jobs",
            "/api/locateanything/image-directories",
            "/api/locateanything/image-jobs/{job_id}",
            "/api/locateanything/image-jobs/{job_id}/validate",
            "/api/locateanything/image-jobs/{job_id}/annotations-zip",
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
        self.assertTrue(payload["locateanything"]["parallel_jobs"])
        self.assertTrue(payload["locateanything"]["image_directory_jobs"])
        self.assertTrue(payload["locateanything"]["image_directory_discovery"])
        self.assertEqual(payload["locateanything"]["scheduler"], "per-device-v1")
        self.assertEqual(payload["locateanything"]["runtime"], "batch")
        self.assertEqual(payload["locateanything"]["generation_mode"], "hybrid")
        self.assertEqual(payload["locateanything"]["batch_size"], 4)

    def test_locateanything_request_defaults_to_hybrid_only(self) -> None:
        request = locateanything.LocateAnythingInferenceReq()
        self.assertEqual(request.generation_mode, "hybrid")
        with self.assertRaises(ValueError):
            locateanything.LocateAnythingInferenceReq(generation_mode="slow")

    def test_locateanything_worker_uses_batch_runtime_defaults(self) -> None:
        calls = []

        class FakeWorker:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

        original_workers = dict(locateanything.WORKERS)
        locateanything.WORKERS.clear()
        try:
            with patch.object(locateanything, "_worker_type", return_value=FakeWorker), patch.object(
                locateanything, "_parse_dtype", return_value="bf16"
            ):
                locateanything._ensure_worker(locateanything.LocateAnythingInferenceReq(), "cuda:0")
        finally:
            locateanything.WORKERS.clear()
            locateanything.WORKERS.update(original_workers)
        self.assertTrue(calls[0][1]["use_batch_runtime"])
        self.assertEqual(calls[0][1]["attn"], "la_flash")
        self.assertEqual(calls[0][1]["scheduler"], "pipeline")
        self.assertTrue(calls[0][1]["strict_attn"])

    def test_run_model_batch_uses_hybrid_and_preserves_batch_size(self) -> None:
        calls = []

        class FakeWorker:
            def predict_batch(self, pairs, **kwargs):
                calls.append((pairs, kwargs))
                return [{"answer": str(index)} for index in range(len(pairs))]

        images = [Image.new("RGB", (8, 8)) for _ in range(4)]
        request = locateanything.LocateAnythingInferenceReq(prompt="person</c>car")
        results = locateanything._run_model_batch(FakeWorker(), images, request)
        self.assertEqual(len(results), 4)
        self.assertEqual(calls[0][1]["generation_mode"], "hybrid")
        self.assertEqual(
            calls[0][0][0][1],
            "Locate all the instances that match the following description: person</c>car.",
        )

    def test_batch_oom_is_retried_as_smaller_batches(self) -> None:
        attempts = []

        def fake_batch(_worker, images, _request):
            attempts.append(len(images))
            if len(images) > 2:
                raise RuntimeError("CUDA out of memory")
            return [{"answer": "ok"} for _ in images]

        job = {}
        images = [Image.new("RGB", (8, 8)) for _ in range(4)]
        with patch.object(locateanything, "_run_model_batch", side_effect=fake_batch):
            results = locateanything._run_model_batch_resilient(
                object(), images, locateanything.LocateAnythingInferenceReq(),
                device="cpu", job=job,
            )
        self.assertEqual(attempts, [4, 2, 2])
        self.assertEqual(len(results), 4)
        self.assertEqual(job["oom_retries"][0]["failed_batch_size"], 4)

    def test_multiple_gpu_configuration_is_reported(self) -> None:
        original_locany = list(locateanything.SETTINGS["devices"])
        original_sam31 = list(sam31.SETTINGS["devices"])
        try:
            locateanything.SETTINGS["devices"] = parse_devices("cuda:0,cuda:1")
            sam31.SETTINGS["devices"] = parse_devices("cuda:0,cuda:1")
            payload = TestClient(app).get("/api/health").json()
            self.assertEqual(payload["locateanything"]["devices"], ["cuda:0", "cuda:1"])
            self.assertEqual(payload["sam31"]["devices"], ["cuda:0", "cuda:1"])
        finally:
            locateanything.SETTINGS["devices"] = original_locany
            sam31.SETTINGS["devices"] = original_sam31

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

    def test_locateanything_health_reports_worker_import_failure(self) -> None:
        original = locateanything.SETTINGS["external_root"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "locateanything_worker.py").write_text("# external worker\n", encoding="utf-8")
            locateanything.SETTINGS["external_root"] = str(root)
            try:
                with patch.object(locateanything, "_worker_type", side_effect=RuntimeError("missing dependency")):
                    payload = locateanything.health_payload()
                self.assertTrue(payload["worker_available"])
                self.assertFalse(payload["worker_importable"])
                self.assertIn("missing dependency", payload["worker_import_error"])
            finally:
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

    def test_locateanything_keeps_ref_label_for_all_following_boxes(self) -> None:
        answer = (
            "<ref>person</ref>"
            "<box><10><20><110><220></box><box><120><30><220><230></box>"
            "<ref>car</ref>"
            "<box><300><400><600><700></box><box><610><410><900><710></box>"
        )
        items = locateanything._extract_items(answer, 1000, 1000)
        self.assertEqual([item["label"] for item in items], ["person", "person", "car", "car"])

        request = locateanything.LocateAnythingInferenceReq(
            prompt="person</c>car",
            class_map={"person": 0, "car": 1},
        )
        class_map = locateanything._normalized_class_map(request)
        self.assertEqual(
            [locateanything._class_id_for_item(item, request, class_map) for item in items],
            [0, 0, 1, 1],
        )

    def test_locateanything_lists_image_directory_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "nested"
            nested.mkdir()
            Image.new("RGB", (32, 24)).save(root / "a.jpg")
            Image.new("RGB", (16, 12)).save(nested / "b.png")
            (root / "ignored.txt").write_text("not an image", encoding="utf-8")

            flat = locateanything._list_image_files(root, recursive=False)
            recursive = locateanything._list_image_files(root, recursive=True)

            self.assertEqual([relative.as_posix() for _, relative in flat], ["a.jpg"])
            self.assertEqual(
                [relative.as_posix() for _, relative in recursive],
                ["a.jpg", "nested/b.png"],
            )

    def test_locateanything_discovers_each_image_subdirectory_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = root / "set-a"
            nested = root / "group" / "set-b"
            empty = root / "empty"
            first.mkdir(parents=True)
            nested.mkdir(parents=True)
            empty.mkdir()
            Image.new("RGB", (8, 8)).save(first / "1.jpg")
            Image.new("RGB", (8, 8)).save(first / "2.png")
            Image.new("RGB", (8, 8)).save(nested / "3.webp")

            flat = locateanything._discover_image_directories(root, recursive=False, include_root=False)
            recursive = locateanything._discover_image_directories(root, recursive=True, include_root=False)

            self.assertEqual([(item["relative_path"], item["image_count"]) for item in flat], [("set-a", 2)])
            self.assertEqual(
                [(item["relative_path"], item["image_count"]) for item in recursive],
                [("group/set-b", 1), ("set-a", 2)],
            )

    def test_image_directory_discovery_api_plans_separate_result_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "input"
            output = root / "output"
            (source / "group" / "set-b").mkdir(parents=True)
            (source / "set-a").mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(source / "set-a" / "1.jpg")
            Image.new("RGB", (8, 8)).save(source / "group" / "set-b" / "2.png")
            with patch.object(locateanything, "_resolve_image_directory", return_value=source), patch.object(
                locateanything, "_resolve_output_candidate", return_value=output,
            ):
                response = TestClient(app).get("/api/locateanything/image-directories", params={
                    "path": "/remote/input", "recursive": "true", "output_root": "/remote/output",
                })

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["directory_count"], 2)
            planned = {item["relative_path"]: item["output_dir"] for item in payload["directories"]}
            self.assertTrue(planned["set-a"].endswith("set-a\\locany_result") or planned["set-a"].endswith("set-a/locany_result"))
            self.assertTrue(
                planned["group/set-b"].endswith("group\\set-b\\locany_result")
                or planned["group/set-b"].endswith("group/set-b/locany_result")
            )

    def test_image_output_cannot_overwrite_input_images_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            input_dir = dataset / "images"
            input_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                locateanything._validate_image_output_layout(input_dir, dataset, copy_images=True)

            locateanything._validate_image_output_layout(input_dir, dataset, copy_images=False)

    def test_image_job_overwrite_never_rejects_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            image = source / "sample.jpg"
            Image.new("RGB", (8, 8)).save(image)
            (output / "existing.txt").write_text("occupied", encoding="utf-8")
            with patch.object(locateanything, "_resolve_image_directory", return_value=source.resolve()), patch.object(
                locateanything, "_resolve_output_dir", return_value=output.resolve(),
            ), patch.object(locateanything.GPU_DEVICE_POOL, "validate"):
                response = TestClient(app).post("/api/locateanything/image-jobs", json={
                    "input_dir": "/remote/input",
                    "output_dir": "/remote/output",
                    "prompt": "person",
                    "class_map": {"person": 0},
                    "device": "cuda:1",
                    "overwrite": "never",
                })
            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("not empty", response.json()["detail"])

    def test_image_job_remote_validation_checks_all_output_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "labels" / "nested").mkdir(parents=True)
            (output / "annotations" / "nested").mkdir(parents=True)
            (output / "labels" / "a.txt").write_text("", encoding="utf-8")
            (output / "labels" / "nested" / "b.txt").write_text("", encoding="utf-8")
            (output / "annotations" / "a.xml").write_text("<annotation/>", encoding="utf-8")
            (output / "annotations" / "nested" / "b.xml").write_text("<annotation/>", encoding="utf-8")
            (output / "metadata.json").write_text(
                '{"total_images": 2, "copy_images": false}', encoding="utf-8"
            )
            (output / "raw_answers.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (output / "locateanything_images.zip").write_bytes(b"zip")
            job_id = "validate-image-job"
            locateanything.IMAGE_JOBS[job_id] = {
                "id": job_id,
                "status": "done",
                "total_images": 2,
                "processed_images": 2,
                "failed_images": 0,
                "direct_output_dir": str(output),
            }
            try:
                response = TestClient(app).get(f"/api/locateanything/image-jobs/{job_id}/validate")
            finally:
                locateanything.IMAGE_JOBS.pop(job_id, None)
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["labels"], 2)
            self.assertEqual(payload["annotations"], 2)
            self.assertEqual(payload["raw_answers"], 2)

    def test_locateanything_image_job_writes_images_yolo_voc_and_zip(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeTorch:
            cuda = FakeCuda()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            cache_dir = root / "cache"
            nested = input_dir / "nested"
            nested.mkdir(parents=True)
            Image.new("RGB", (100, 80), "white").save(input_dir / "first.jpg")
            Image.new("RGB", (50, 40), "black").save(nested / "second.png")
            items = locateanything._list_image_files(input_dir.resolve(), recursive=True)
            request = locateanything.LocateAnythingImageDirectoryReq(
                input_dir=str(input_dir),
                recursive=True,
                prompt="person</c>car",
                class_map={"person": 4, "car": 7},
                copy_images=True,
                device="cpu",
            )
            job_id = "image-job-test"
            locateanything.IMAGE_JOBS[job_id] = {"id": job_id, "status": "queued"}
            original_cache = locateanything.SETTINGS["cache_dir"]
            original_keep = locateanything.SETTINGS["keep_model_loaded"]
            locateanything.SETTINGS["cache_dir"] = cache_dir
            locateanything.SETTINGS["keep_model_loaded"] = True
            try:
                with (
                    patch.object(locateanything.GPU_DEVICE_POOL, "acquire", return_value="cpu"),
                    patch.object(locateanything.GPU_DEVICE_POOL, "release") as release,
                    patch.object(locateanything, "_ensure_worker", return_value=object()),
                    patch.object(locateanything, "_torch", return_value=FakeTorch()),
                    patch.object(
                        locateanything,
                        "_run_model_batch",
                        side_effect=lambda _worker, images, _request: [
                            {
                                "answer": (
                                    "<ref>person</ref><box><100><200><500><800></box>"
                                    "<ref>car</ref><box><550><300><900><700></box>"
                                )
                            }
                            for _ in images
                        ],
                    ),
                ):
                    locateanything._run_image_job_sync(job_id, request, input_dir.resolve(), items)
            finally:
                locateanything.SETTINGS["cache_dir"] = original_cache
                locateanything.SETTINGS["keep_model_loaded"] = original_keep

            job = locateanything.IMAGE_JOBS.pop(job_id)
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["processed_images"], 2)
            self.assertEqual(job["failed_images"], 0)
            release.assert_called_once_with("cpu")
            output = cache_dir / job_id
            self.assertTrue((output / "images" / "first.jpg").is_file())
            self.assertTrue((output / "images" / "nested" / "second.png").is_file())
            yolo_classes = [line.split()[0] for line in (output / "labels" / "first.txt").read_text().splitlines()]
            self.assertEqual(yolo_classes, ["4", "7"])
            self.assertEqual(
                ET.parse(output / "annotations" / "nested" / "second.xml").getroot().findtext("object/name"),
                "person",
            )
            with zipfile.ZipFile(output / "locateanything_images.zip") as archive:
                names = set(archive.namelist())
            self.assertIn("images/first.jpg", names)
            self.assertIn("labels/nested/second.txt", names)
            self.assertIn("annotations/nested/second.xml", names)


if __name__ == "__main__":
    unittest.main()
