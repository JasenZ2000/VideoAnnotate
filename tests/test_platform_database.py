from __future__ import annotations

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workflow_platform.database import PlatformDatabase


def task(task_id: str = "task-1") -> dict[str, str]:
    return {
        "task_id": task_id,
        "name": "行人检测 · 清洗相似图片",
        "status": "ready",
        "publisher": "publisher",
        "manager": "publisher",
        "product_tag": "BSD",
        "part_prefix": "BSD",
        "application_date": "2022/8/10",
        "applicant": "刘湛基",
        "project": "行人检测",
        "annotation_content": "清洗相似图片",
        "dataset_source": "客户视频",
        "hourly_capacity": "1000",
        "data_amount": "15074",
        "estimated_hours": "15",
        "data_path": r"\\server\data",
        "guide_path": r"\\server\guide.pdf",
        "created_at": "2026-07-15T10:00:00+08:00",
        "updated_at": "2026-07-15T10:00:00+08:00",
    }


class PlatformDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.db = PlatformDatabase(Path(self.temporary.name) / "platform.sqlite3")
        self.db.initialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_task_parts_timing_review_and_rework(self) -> None:
        created = self.db.create_task(task(), 2, "2026-07-15T10:00:00+08:00")
        self.assertEqual(created["product_tag"], "BSD")
        self.assertEqual([p["name"] for p in self.db.list_parts("task-1", "2026-07-15T10:00:00+08:00")],
                         ["BSD_part_001", "BSD_part_002"])

        claimed = self.db.claim_next_part("task-1", "worker", "2026-07-15T10:01:00+08:00")
        submitted = self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "发现两张模糊图", "2026-07-15T10:11:00+08:00"
        )
        self.assertEqual(submitted["work_seconds"], 600)
        rework = self.db.review_part(
            "task-1", claimed["part_id"], "publisher", "rework", "补充漏框", "2026-07-15T10:12:00+08:00"
        )
        self.assertEqual(rework["status"], "rework")
        self.db.start_rework("task-1", claimed["part_id"], "worker", "2026-07-15T10:13:00+08:00")
        self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "已按意见修改", "2026-07-15T10:18:00+08:00"
        )
        approved = self.db.review_part(
            "task-1", claimed["part_id"], "publisher", "approve", "通过", "2026-07-15T10:20:00+08:00"
        )
        self.assertEqual(approved["status"], "completed")
        self.assertEqual(approved["work_seconds"], 900)
        self.assertEqual(len(approved["comments"]), 4)
        stats = self.db.annotator_statistics("task-1", "2026-07-15T10:20:00+08:00")
        self.assertEqual(stats[0]["completed"], 1)
        self.assertEqual(stats[0]["work_seconds"], 900)

    def test_concurrent_claims_get_different_parts(self) -> None:
        self.db.create_task(task(), 2, "2026-07-15T10:00:00+08:00")
        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(executor.map(
                lambda actor: self.db.claim_next_part("task-1", actor, "2026-07-15T10:01:00+08:00"),
                ["worker-a", "worker-b"],
            ))
        self.assertEqual({part["part_id"] for part in claimed}, {1, 2})

    def test_only_publisher_can_add_and_review(self) -> None:
        self.db.create_task(task(), 1, "2026-07-15T10:00:00+08:00")
        with self.assertRaises(PermissionError):
            self.db.add_parts("task-1", 1, "worker", "2026-07-15T10:01:00+08:00")
        parts = self.db.add_parts("task-1", 2, "publisher", "2026-07-15T10:01:00+08:00")
        self.assertEqual(len(parts), 3)
        with self.assertRaises(PermissionError):
            self.db.claim_next_part("task-1", "publisher", "2026-07-15T10:02:00+08:00")

    def test_user_session_lifecycle(self) -> None:
        self.db.create_user("admin", "hash", "admin", "管理员", "2026-07-15T10:00:00+08:00")
        self.db.create_session("sid", "admin", "token", "2026-07-16T10:00:00+08:00",
                               "2026-07-15T10:00:00+08:00")
        self.assertEqual(
            self.db.get_session_user("token", "2026-07-15T11:00:00+08:00")["username"], "admin"
        )
        self.db.update_user("admin", password_hash="new", now="2026-07-15T12:00:00+08:00")
        self.assertIsNone(self.db.get_session_user("token", "2026-07-15T12:01:00+08:00"))

    def test_legacy_task_table_is_upgraded_in_place(self) -> None:
        path = Path(self.temporary.name) / "legacy.sqlite3"
        connection = sqlite3.connect(path)
        connection.executescript("""
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY, name TEXT NOT NULL, assignee TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'created', deleted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            INSERT INTO tasks(task_id,name,assignee,created_at,updated_at)
            VALUES('old','旧任务','old-publisher','t1','t1');
        """)
        connection.close()
        upgraded = PlatformDatabase(path)
        upgraded.initialize()
        loaded = upgraded.get_task("old", "2026-07-15T10:00:00+08:00")
        self.assertEqual(loaded["publisher"], "old-publisher")
        self.assertEqual(loaded["product_tag"], "")


if __name__ == "__main__":
    unittest.main()
