from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gpu_services_isolated import locateanything
from gpu_services_isolated.server import app
from gpu_services_isolated.worker_process import (
    IsolatedWorkerManager,
    RemoteWorkerError,
    WorkerConfig,
)


def _config(device: str = "cuda:0", dtype: str = "bf16") -> WorkerConfig:
    return WorkerConfig(
        device=device,
        dtype=dtype,
        model="/models/locany",
        external_root="/runtime/embodied",
        batch_attn="la_flash",
        vision_attn="auto",
        scheduler="pipeline",
        group_size=0,
        strict_attn=True,
        dense_backend="sdpa",
    )


class FakeProxy:
    next_pid = 1000

    def __init__(self, config, *, startup_timeout, rpc_timeout):
        self.config = config
        self.startup_timeout = startup_timeout
        self.rpc_timeout = rpc_timeout
        self.alive = True
        self.pid = FakeProxy.next_pid
        FakeProxy.next_pid += 1
        self.closed = False
        self.ready_info = {
            "kind": "ready",
            "pid": self.pid,
            "device": config.device,
            "actual_device": config.device,
            "flash_smoke": {"ok": True},
        }

    def close(self):
        self.closed = True
        self.alive = False

    def status(self):
        return {"pid": self.pid, "actual_device": self.config.device}


class IsolatedGpuServiceTests(unittest.TestCase):
    def test_manager_creates_one_process_proxy_per_device(self) -> None:
        manager = IsolatedWorkerManager()
        try:
            with patch("gpu_services_isolated.worker_process.IsolatedWorkerProxy", FakeProxy):
                zero = manager.ensure(_config("cuda:0"))
                one = manager.ensure(_config("cuda:1"))
                zero_again = manager.ensure(_config("cuda:0"))
            self.assertIs(zero, zero_again)
            self.assertIsNot(zero, one)
            self.assertNotEqual(zero.pid, one.pid)
            self.assertEqual(
                [row["device"] for row in manager.snapshot()],
                ["cuda:0", "cuda:1"],
            )
        finally:
            manager.close_all()

    def test_dtype_change_replaces_only_that_devices_process(self) -> None:
        manager = IsolatedWorkerManager()
        try:
            with patch("gpu_services_isolated.worker_process.IsolatedWorkerProxy", FakeProxy):
                old = manager.ensure(_config("cuda:2", "bf16"))
                other = manager.ensure(_config("cuda:3", "bf16"))
                replacement = manager.ensure(_config("cuda:2", "fp16"))
            self.assertTrue(old.closed)
            self.assertFalse(other.closed)
            self.assertIsNot(old, replacement)
            self.assertEqual(replacement.config.dtype, "fp16")
        finally:
            manager.close_all()

    def test_remote_oom_remains_detectable_by_resilient_batch_splitter(self) -> None:
        error = RemoteWorkerError({
            "error_type": "OutOfMemoryError",
            "error": "CUDA out of memory",
            "traceback": "remote stack",
        })
        self.assertTrue(locateanything._compat._is_cuda_oom(error))

    def test_parent_pipeline_never_reports_cuda_available(self) -> None:
        torch_facade = locateanything._compat._torch()
        self.assertFalse(torch_facade.cuda.is_available())

    def test_worker_config_keeps_physical_device_identity(self) -> None:
        original = dict(locateanything.SETTINGS)
        try:
            locateanything.SETTINGS.update({
                "model": "/model",
                "external_root": "/runtime",
                "batch_attn": "la_flash",
                "vision_attn": "auto",
                "batch_scheduler": "pipeline",
                "batch_group_size": 0,
                "strict_attn": True,
                "dense_backend": "sdpa",
            })
            config = locateanything._worker_config("cuda:7", "bf16")
            self.assertEqual(config.device, "cuda:7")
            self.assertEqual(config.dtype, "bf16")
        finally:
            locateanything.SETTINGS.clear()
            locateanything.SETTINGS.update(original)

    def test_compatible_job_routes_and_worker_diagnostics_are_exposed(self) -> None:
        paths = TestClient(app).get("/openapi.json").json()["paths"]
        self.assertIn("/api/locateanything/jobs", paths)
        self.assertIn("/api/locateanything/image-jobs", paths)
        self.assertIn("/api/locateanything/workers", paths)
        self.assertIn("/api/locateanything/workers/preload", paths)

    def test_health_describes_process_isolation_and_distinct_children(self) -> None:
        workers = [
            {
                "kind": "ready", "pid": 2001, "device": "cuda:0",
                "actual_device": "cuda:0", "alive": True, "flash_smoke": {"ok": True},
            },
            {
                "kind": "ready", "pid": 2002, "device": "cuda:1",
                "actual_device": "cuda:1", "alive": True, "flash_smoke": {"ok": True},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            runtime = root / "runtime"
            (model / "batch_utils").mkdir(parents=True)
            (model / "kernel_utils").mkdir()
            runtime.mkdir()
            (runtime / "locateanything_worker.py").write_text(
                "class LocateAnythingWorker:\n    def __init__(self, use_batch_runtime=False): pass\n",
                encoding="utf-8",
            )
            original = dict(locateanything.SETTINGS)
            try:
                locateanything.SETTINGS.update({
                    "model": str(model), "external_root": str(runtime),
                    "devices": ["cuda:0", "cuda:1"],
                })
                with patch.object(locateanything.WORKER_MANAGER, "snapshot", return_value=workers), patch.object(
                    locateanything, "_dependency_preflight",
                    return_value={
                        "la_flash_available": True,
                        "worker_source_supports_batch_runtime": True,
                    },
                ):
                    payload = locateanything.health_payload()
            finally:
                locateanything.SETTINGS.clear()
                locateanything.SETTINGS.update(original)

        self.assertEqual(payload["architecture"], "one-process-per-gpu-v2")
        self.assertFalse(payload["parent_cuda_contexts"])
        self.assertEqual(payload["worker_count"], 2)
        self.assertEqual({row["pid"] for row in payload["loaded_workers"]}, {2001, 2002})
        self.assertEqual(payload["scheduler"], "per-device-process-v2")
        self.assertTrue(payload["la_flash_available"])


if __name__ == "__main__":
    unittest.main()
