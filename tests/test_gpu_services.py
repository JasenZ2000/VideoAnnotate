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
            "/api/locateanything/image-jobs/{job_id}",
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
        self.assertEqual(payload["locateanything"]["scheduler"], "per-device-v1")

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

    def test_image_output_cannot_overwrite_input_images_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            input_dir = dataset / "images"
            input_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                locateanything._validate_image_output_layout(input_dir, dataset, copy_images=True)

            locateanything._validate_image_output_layout(input_dir, dataset, copy_images=False)

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
                prompt="person",
                class_map={"person": 4},
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
                        "_run_model",
                        return_value={"answer": "<ref>person</ref><box><100><200><500><800></box>"},
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
            self.assertTrue((output / "labels" / "first.txt").read_text().startswith("4 "))
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
