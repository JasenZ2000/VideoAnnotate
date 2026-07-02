from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workflow_platform.database import PlatformDatabase


def sample_task(task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "name": "数据库测试任务",
        "assignee": "tester",
        "notes": "保留中文",
        "prelabel_source": "locateanything",
        "prompt": "person",
        "status": "segmented",
        "deleted": False,
        "current_video_id": "video-1",
        "created_at": "2026-07-02T10:00:00+08:00",
        "updated_at": "2026-07-02T10:01:00+08:00",
        "custom_field": {"preserved": True},
        "classes": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"},
        ],
        "stages": {
            "video": {"status": "done", "message": "a.mp4"},
            "tracking": {"status": "pending", "message": ""},
        },
        "videos": [
            {
                "video_id": "video-1",
                "name": "a.mp4",
                "path": "D:/tasks/a.mp4",
                "metadata": {"width": 1920, "height": 1080, "frame_count": 300, "fps": 25.0},
                "status": "uploaded",
                "input_label_dir": "D:/tasks/labels",
                "split": {
                    "status": "done",
                    "segment_length": 100,
                    "label_source": "input",
                    "segments": 1,
                },
                "segments": [
                    {
                        "segment_id": "seg_0000",
                        "start_frame": 0,
                        "end_frame": 99,
                        "frame_count": 100,
                        "video_path": "D:/tasks/seg.mp4",
                        "input_label_dir": "D:/tasks/seg-labels",
                        "locany_label_dir": "D:/tasks/locany",
                        "labels_copied": 98,
                        "status": "ready",
                        "locateanything": {"status": "done", "prompt": "person"},
                        "tracking": {"status": "done", "results": "D:/tasks/tracking.json"},
                    }
                ],
            }
        ],
    }


class PlatformDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db = PlatformDatabase(self.root / "platform.sqlite3")
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_nested_task_round_trip(self) -> None:
        self.db.save_task(sample_task())
        loaded = self.db.load_task("task-1")

        self.assertEqual(loaded["classes"][1], {"id": 1, "name": "car"})
        self.assertEqual(loaded["videos"][0]["metadata"]["frame_count"], 300)
        self.assertEqual(loaded["videos"][0]["segments"][0]["tracking"]["status"], "done")
        self.assertEqual(loaded["custom_field"], {"preserved": True})

    def test_events_are_thread_safe_and_ordered(self) -> None:
        self.db.save_task(sample_task())
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(
                lambda index: self.db.add_event("task-1", f"t{index:02d}", "info", f"event-{index}"),
                range(20),
            ))

        events = self.db.list_events("task-1", limit=100)
        self.assertEqual(len(events), 20)
        self.assertEqual(self.db.health()["quick_check"], "ok")

    def test_deleted_tasks_are_hidden_by_default(self) -> None:
        active = sample_task("active")
        deleted = sample_task("deleted")
        deleted["deleted"] = True
        deleted["status"] = "deleted"
        self.db.save_task(active)
        self.db.save_task(deleted)

        self.assertEqual([task["task_id"] for task in self.db.list_tasks()], ["active"])
        self.assertEqual(len(self.db.list_tasks(include_deleted=True)), 2)

    def test_legacy_json_and_events_are_imported_once(self) -> None:
        task_dir = self.root / "legacy-task"
        task_dir.mkdir()
        legacy = sample_task("legacy-task")
        (task_dir / "task.json").write_text(
            json.dumps(legacy, ensure_ascii=False),
            encoding="utf-8",
        )
        (task_dir / "events.jsonl").write_text(
            json.dumps({"time": "t1", "level": "info", "message": "旧事件"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        first = self.db.migrate_legacy_directory(self.root)
        second = self.db.migrate_legacy_directory(self.root)

        self.assertEqual(first, {"tasks": 1, "events": 1, "failed": 0})
        self.assertEqual(second, {"tasks": 0, "events": 0, "failed": 0})
        self.assertEqual(self.db.load_task("legacy-task")["name"], "数据库测试任务")
        self.assertEqual(self.db.list_events("legacy-task")[0]["message"], "旧事件")
        self.assertTrue((task_dir / "task.json").exists())


if __name__ == "__main__":
    unittest.main()
