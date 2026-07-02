from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

import workflow_platform.server as platform
from workflow_platform.server import CreateTaskReq, health, parse_classes_text


class WorkflowPlatformTests(unittest.TestCase):
    def test_class_table_accepts_supported_notation(self) -> None:
        classes = parse_classes_text("0 person\ncar=1\n2: bicycle")
        self.assertEqual(
            classes,
            [
                {"id": 0, "name": "person"},
                {"id": 1, "name": "car"},
                {"id": 2, "name": "bicycle"},
            ],
        )

    def test_class_table_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(HTTPException):
            parse_classes_text("0 person\n0 car")

    def test_health_contract(self) -> None:
        payload = asyncio.run(health())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "annotation-platform")
        self.assertEqual(payload["database"]["quick_check"], "ok")

    def test_task_api_uses_sqlite_without_writing_task_json(self) -> None:
        old_settings = dict(platform.SETTINGS)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform.SETTINGS["tasks_dir"] = root
            platform.SETTINGS["database_path"] = str(root / "metadata.sqlite3")
            try:
                response = asyncio.run(platform.create_task(CreateTaskReq(
                    name="SQLite API 测试",
                    classes="0 person\n1 car",
                )))
                task_id = response["task"]["task_id"]
                detail = asyncio.run(platform.get_task(task_id))
                self.assertEqual(len(detail["classes"]), 2)
                self.assertEqual(detail["events"][0]["message"], "Task created")
                self.assertTrue((root / "metadata.sqlite3").exists())
                self.assertFalse((root / task_id / "task.json").exists())
            finally:
                platform.SETTINGS.clear()
                platform.SETTINGS.update(old_settings)


if __name__ == "__main__":
    unittest.main()
