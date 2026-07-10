from __future__ import annotations

import json
import sqlite3
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
        "task_type": "video_detection",
        "publisher": "publisher",
        "manager": "manager",
        "annotators": ["ann-a", "ann-b"],
        "instructions": "任务说明",
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
        self.assertEqual(loaded["annotators"], ["ann-a", "ann-b"])
        self.assertEqual(loaded["manager"], "manager")

    def test_part_claim_rework_and_time_accounting(self) -> None:
        self.db.save_task(sample_task())
        self.db.create_parts("task-1", 2, "批次", "统一要求", "2026-07-02T10:00:00+08:00")

        claimed = self.db.claim_next_part("task-1", "ann-a", "2026-07-02T10:01:00+08:00")
        self.assertEqual(claimed["part_index"], 1)
        submitted = self.db.submit_part(
            "task-1", claimed["part_id"], "ann-a", "初次完成", "2026-07-02T10:11:00+08:00"
        )
        self.assertEqual(submitted["work_seconds"], 600)

        rework = self.db.review_part(
            "task-1", claimed["part_id"], "manager", "rework", "需要修正", "2026-07-02T10:12:00+08:00"
        )
        self.assertEqual(rework["status"], "rework")
        self.db.start_part("task-1", claimed["part_id"], "ann-a", "2026-07-02T10:13:00+08:00")
        submitted_again = self.db.submit_part(
            "task-1", claimed["part_id"], "ann-a", "已修正", "2026-07-02T10:18:00+08:00"
        )
        self.assertEqual(submitted_again["work_seconds"], 900)
        completed = self.db.review_part(
            "task-1", claimed["part_id"], "publisher", "approve", "通过", "2026-07-02T10:20:00+08:00"
        )
        self.assertEqual(completed["status"], "completed")

    def test_concurrent_claims_receive_different_parts(self) -> None:
        self.db.save_task(sample_task())
        self.db.create_parts("task-1", 2, "Part", "", "2026-07-02T10:00:00+08:00")
        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(executor.map(
                lambda actor: self.db.claim_next_part(
                    "task-1", actor, "2026-07-02T10:01:00+08:00"
                ),
                ["ann-a", "ann-b"],
            ))
        self.assertEqual({part["part_id"] for part in claimed}, {1, 2})

    def test_non_annotator_cannot_claim_part(self) -> None:
        self.db.save_task(sample_task())
        self.db.create_parts("task-1", 1, "Part", "", "2026-07-02T10:00:00+08:00")
        with self.assertRaises(PermissionError):
            self.db.claim_next_part("task-1", "outsider", "2026-07-02T10:01:00+08:00")

    def test_issue_lifecycle(self) -> None:
        self.db.save_task(sample_task())
        issue = self.db.create_issue({
            "task_id": "task-1",
            "reported_by": "ann-a",
            "assigned_to": "manager",
            "severity": "high",
            "title": "说明不清楚",
            "description": "类别边界需要确认",
            "created_at": "2026-07-02T10:00:00+08:00",
            "updated_at": "2026-07-02T10:00:00+08:00",
        })
        resolved = self.db.resolve_issue(
            "task-1", issue["issue_id"], "manager", "已补充说明", "2026-07-02T10:10:00+08:00"
        )
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["resolution"], "已补充说明")

    def test_attachment_metadata(self) -> None:
        self.db.save_task(sample_task())
        record = self.db.add_attachment({
            "attachment_id": "file-1",
            "task_id": "task-1",
            "filename": "规范.pdf",
            "stored_path": "D:/tasks/规范.pdf",
            "media_type": "application/pdf",
            "size_bytes": 1024,
            "sha256": "a" * 64,
            "uploaded_by": "publisher",
            "created_at": "2026-07-02T10:00:00+08:00",
        })
        self.assertEqual(record["filename"], "规范.pdf")
        self.assertEqual(self.db.list_attachments("task-1")[0]["sha256"], "a" * 64)

    def test_user_and_session_lifecycle(self) -> None:
        created = self.db.create_user(
            "admin",
            "hashed-password",
            "admin",
            "Administrator",
            "2026-07-07T10:00:00+08:00",
            True,
        )
        self.assertEqual(created["role"], "admin")
        self.assertEqual(self.db.user_count(), 1)
        self.assertEqual(self.db.active_admin_count(), 1)

        self.db.create_session(
            "session-1",
            "admin",
            "token-hash",
            "2026-07-08T10:00:00+08:00",
            "2026-07-07T10:00:00+08:00",
        )
        session_user = self.db.get_session_user("token-hash", "2026-07-07T12:00:00+08:00")
        self.assertEqual(session_user["username"], "admin")

        self.db.update_user(
            "admin",
            role="user",
            display_name="Admin User",
            is_active=False,
            now="2026-07-07T12:05:00+08:00",
        )
        updated = self.db.get_user("admin")
        self.assertEqual(updated["role"], "user")
        self.assertFalse(updated["is_active"])
        self.assertIsNone(self.db.get_session_user("token-hash", "2026-07-07T12:06:00+08:00"))

    def test_version_one_database_is_upgraded_in_place(self) -> None:
        path = self.root / "version-one.sqlite3"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                assignee TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                prelabel_source TEXT NOT NULL DEFAULT 'none',
                prompt TEXT NOT NULL DEFAULT 'person',
                status TEXT NOT NULL DEFAULT 'created',
                deleted INTEGER NOT NULL DEFAULT 0,
                current_video_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                extra_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO tasks(
                task_id, name, assignee, created_at, updated_at
            ) VALUES ('old-task', '旧任务', '旧负责人', 't1', 't1');
            """
        )
        connection.commit()
        connection.close()

        upgraded = PlatformDatabase(path)
        upgraded.initialize()
        task = upgraded.load_task("old-task")

        self.assertEqual(task["manager"], "旧负责人")
        self.assertEqual(task["publisher"], "旧负责人")
        self.assertEqual(task["task_type"], "general")
        self.assertEqual(upgraded.health()["schema_version"], 3)

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
