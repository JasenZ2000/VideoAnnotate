from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from gpu_services import locateanything
from gpu_services.device_pool import DevicePool, normalize_device, parse_devices


class DevicePoolTests(unittest.TestCase):
    def test_device_parser_accepts_ids_and_deduplicates(self) -> None:
        self.assertEqual(parse_devices("0,cuda:1,cuda:0"), ["cuda:0", "cuda:1"])
        self.assertEqual(normalize_device("cuda"), "cuda:0")

    def test_auto_jobs_use_different_free_devices(self) -> None:
        pool = DevicePool()
        first = pool.acquire(["cuda:0", "cuda:1"])
        second = pool.acquire(["cuda:0", "cuda:1"])
        self.assertEqual({first, second}, {"cuda:0", "cuda:1"})
        self.assertEqual(pool.snapshot()["busy_devices"], ["cuda:0", "cuda:1"])
        pool.release(first)
        pool.release(second)

    def test_same_device_waits_until_current_job_releases_it(self) -> None:
        pool = DevicePool()
        pool.acquire(["cuda:0"], "cuda:0")
        acquired = threading.Event()

        def wait_for_device() -> None:
            device = pool.acquire(["cuda:0"], "cuda:0")
            acquired.set()
            pool.release(device)

        thread = threading.Thread(target=wait_for_device)
        thread.start()
        self.assertFalse(acquired.wait(0.05))
        pool.release("cuda:0")
        self.assertTrue(acquired.wait(1))
        thread.join(timeout=1)

    def test_requested_device_must_be_enabled(self) -> None:
        pool = DevicePool()
        with self.assertRaisesRegex(ValueError, "not enabled"):
            pool.validate(["cuda:0", "cuda:1"], "cuda:2")

    def test_locateanything_release_removes_worker_and_clears_cuda_cache(self) -> None:
        calls: list[object] = []

        class FakeDeviceContext:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def synchronize(device: str) -> None:
                calls.append(("synchronize", device))

            @staticmethod
            def device(device: str) -> FakeDeviceContext:
                calls.append(("device", device))
                return FakeDeviceContext()

            @staticmethod
            def empty_cache() -> None:
                calls.append("empty_cache")

            @staticmethod
            def ipc_collect() -> None:
                calls.append("ipc_collect")

        class FakeTorch:
            cuda = FakeCuda()

        key = ("cuda:0", "bf16")
        locateanything.WORKERS[key] = object()
        with patch.object(locateanything, "_torch", return_value=FakeTorch()):
            locateanything._release_worker(*key)
        self.assertNotIn(key, locateanything.WORKERS)
        self.assertIn("empty_cache", calls)
        self.assertIn("ipc_collect", calls)


if __name__ == "__main__":
    unittest.main()
