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

    def test_task_can_create_parts_from_work_directory_specs(self) -> None:
        self.db.create_task(
            task(), 0, "2026-07-15T10:00:00+08:00",
            part_specs=[
                {"name": "split_001", "work_path": r"\\server\data\split_001"},
                {"name": "group / split_002", "work_path": r"\\server\data\group\split_002"},
            ],
        )
        parts = self.db.list_parts("task-1", "2026-07-15T10:00:00+08:00")
        self.assertEqual([part["name"] for part in parts], ["split_001", "group / split_002"])
        self.assertEqual(parts[1]["work_path"], r"\\server\data\group\split_002")
        self.assertEqual(self.db.health()["schema_version"], 6)

    def test_part_can_pause_resume_and_return(self) -> None:
        self.db.create_task(task(), 1, "2026-07-15T10:00:00+08:00")
        claimed = self.db.claim_next_part(
            "task-1", "worker", "2026-07-15T10:01:00+08:00"
        )
        paused = self.db.pause_part(
            "task-1", claimed["part_id"], "worker", "2026-07-15T10:05:00+08:00"
        )
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["work_seconds"], 240)
        with self.assertRaisesRegex(ValueError, "active part"):
            self.db.claim_next_part("task-1", "worker", "2026-07-15T10:06:00+08:00")

        resumed = self.db.resume_part(
            "task-1", claimed["part_id"], "worker", "2026-07-15T10:10:00+08:00"
        )
        self.assertEqual(resumed["status"], "in_progress")
        returned = self.db.return_part(
            "task-1", claimed["part_id"], "worker", "cannot continue",
            "2026-07-15T10:12:00+08:00",
        )
        self.assertEqual(returned["status"], "pending")
        self.assertEqual(returned["annotator"], "")
        self.assertEqual(returned["work_seconds"], 0)
        self.assertEqual(returned["comments"][-1]["kind"], "return")
        returned_stats = self.db.annotator_statistics(
            "task-1", "2026-07-15T10:12:00+08:00"
        )
        self.assertEqual(returned_stats[0]["work_seconds"], 360)
        reassigned = self.db.claim_next_part(
            "task-1", "worker-2", "2026-07-15T10:13:00+08:00"
        )
        self.assertEqual(reassigned["annotator"], "worker-2")

    def test_expected_part_time_flags_deviation_and_publisher_reviews_it(self) -> None:
        self.db.create_task(task(), 1, "2026-07-15T10:00:00+08:00")
        updated = self.db.update_task(
            "task-1", "publisher", {"manager": "observer", "expected_part_seconds": 120},
            "2026-07-15T10:00:30+08:00",
        )
        self.assertEqual(updated["manager"], "observer")
        self.assertEqual(updated["expected_part_seconds"], 120)
        claimed = self.db.claim_next_part(
            "task-1", "worker", "2026-07-15T10:01:00+08:00"
        )
        submitted = self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "done", "2026-07-15T10:07:00+08:00"
        )
        self.assertTrue(submitted["has_time_deviation"])
        self.assertEqual(submitted["time_deviation_ratio"], 2.0)
        with self.assertRaises(PermissionError):
            self.db.review_part_time(
                "task-1", claimed["part_id"], "observer", "estimate_unreasonable", "",
                "2026-07-15T10:08:00+08:00",
            )
        reviewed = self.db.review_part_time(
            "task-1", claimed["part_id"], "publisher", "estimate_unreasonable",
            "estimate was too short", "2026-07-15T10:08:00+08:00",
        )
        self.assertEqual(reviewed["time_review_status"], "estimate_unreasonable")
        self.assertEqual(reviewed["time_review_actor"], "publisher")

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
        claimed = self.db.claim_next_part(
            "task-1", "publisher", "2026-07-15T10:02:00+08:00"
        )
        self.assertEqual(claimed["annotator"], "publisher")

    def test_admin_flag_can_review_another_publishers_part(self) -> None:
        self.db.create_task(task(), 1, "2026-07-15T10:00:00+08:00")
        claimed = self.db.claim_next_part(
            "task-1", "worker", "2026-07-15T10:01:00+08:00"
        )
        self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "已完成", "2026-07-15T10:02:00+08:00"
        )
        with self.assertRaisesRegex(PermissionError, "publisher or admin"):
            self.db.review_part(
                "task-1", claimed["part_id"], "other-user", "approve", "", "2026-07-15T10:03:00+08:00"
            )
        reviewed = self.db.review_part(
            "task-1", claimed["part_id"], "admin-user", "approve", "管理员通过",
            "2026-07-15T10:04:00+08:00", is_admin=True,
        )
        self.assertEqual(reviewed["status"], "completed")
        self.assertEqual(reviewed["comments"][-1]["actor"], "admin-user")

    def test_publisher_can_edit_and_delete_task_in_any_state(self) -> None:
        self.db.create_task(task(), 1, "2026-07-15T10:00:00+08:00")
        updated = self.db.update_task(
            "task-1", "publisher",
            {"product_tag": "AEB", "project": "车辆检测", "part_prefix": "NEW"},
            "2026-07-15T10:01:00+08:00",
        )
        self.assertEqual(updated["product_tag"], "AEB")
        self.assertEqual(updated["name"], "车辆检测 · 清洗相似图片")
        with self.assertRaises(PermissionError):
            self.db.update_task("task-1", "worker", {"project": "越权"},
                                "2026-07-15T10:02:00+08:00")

        self.db.claim_next_part("task-1", "publisher", "2026-07-15T10:03:00+08:00")
        with self.assertRaises(PermissionError):
            self.db.delete_task("task-1", "worker", "2026-07-15T10:04:00+08:00")
        self.db.delete_task("task-1", "publisher", "2026-07-15T10:04:00+08:00")
        with self.assertRaises(KeyError):
            self.db.get_task("task-1", "2026-07-15T10:05:00+08:00")

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
