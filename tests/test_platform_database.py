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
        self.db.update_part(
            "task-1", claimed["part_id"], "worker", "2026-07-15T10:02:00+08:00",
            progress_count=12,
        )
        submitted = self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "发现两张模糊图", "2026-07-15T10:11:00+08:00"
        )
        self.assertEqual(submitted["work_seconds"], 600)
        submitted_summary = self.db.part_summary("task-1")
        self.assertEqual(submitted_summary["annotated"], 1)
        self.assertEqual(submitted_summary["completed"], 0)
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
        self.assertEqual(self.db.part_summary("task-1")["annotated"], 1)
        stats = self.db.annotator_statistics("task-1", "2026-07-15T10:20:00+08:00")
        self.assertEqual(stats[0]["completed"], 1)
        self.assertEqual(stats[0]["work_seconds"], 900)
        self.assertEqual(stats[0]["image_count"], 12)
        self.assertEqual(stats[0]["images_per_hour"], 48.0)
        self.assertEqual(
            self.db.task_annotation_statistics("task-1", "2026-07-15T10:20:00+08:00"),
            {"work_seconds": 900.0, "image_count": 12, "images_per_hour": 48.0},
        )

    def test_submit_marks_all_part_images_annotated(self) -> None:
        self.db.create_task(
            task(), 0, "2026-07-15T10:00:00+08:00",
            part_specs=[{"name": "split_001", "image_count": 120}],
        )
        claimed = self.db.claim_next_part("task-1", "worker", "2026-07-15T10:01:00+08:00")
        self.assertEqual(claimed["progress_count"], 0)
        submitted = self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "完成", "2026-07-15T10:11:00+08:00",
        )
        self.assertEqual(submitted["image_count"], 120)
        self.assertEqual(submitted["progress_count"], 120)
        stats = self.db.annotator_statistics("task-1", "2026-07-15T10:11:00+08:00")
        self.assertEqual(stats[0]["image_count"], 120)

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
        self.assertEqual(self.db.health()["schema_version"], 7)

    def test_large_part_batch_can_be_created_and_reviewed_by_page(self) -> None:
        self.db.create_task(task(), 2000, "2026-07-15T10:00:00+08:00")
        self.assertEqual(self.db.part_summary("task-1")["total"], 2000)
        page = self.db.list_parts_page(
            "task-1", "2026-07-15T10:01:00+08:00", query="part_0", page=2,
            page_size=25,
        )
        self.assertEqual(page["page"], 2)
        self.assertEqual(page["page_size"], 25)
        self.assertEqual(len(page["items"]), 25)
        self.assertGreater(page["total"], 25)
        self.assertTrue(all("part_0" in item["name"] for item in page["items"]))

        claimed = self.db.claim_next_part(
            "task-1", "worker", "2026-07-15T10:02:00+08:00"
        )
        self.db.submit_part(
            "task-1", claimed["part_id"], "worker", "done",
            "2026-07-15T10:03:00+08:00",
        )
        submitted = self.db.list_parts_page(
            "task-1", "2026-07-15T10:03:00+08:00", status="submitted"
        )
        self.assertEqual(submitted["total"], 1)
        self.assertEqual(submitted["items"][0]["part_id"], claimed["part_id"])

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

    def test_admin_updates_task_rank_and_priority_with_audit_logs(self) -> None:
        first = self.db.create_task(task("task-1"), 1, "2026-07-15T10:00:00+08:00")
        second = self.db.create_task(task("task-2"), 1, "2026-07-15T10:01:00+08:00")
        self.assertGreater(second["rank"], first["rank"])
        with self.assertRaises(PermissionError):
            self.db.update_task_ordering(
                "task-1", "worker", rank=100, priority="urgent",
                now="2026-07-15T10:02:00+08:00",
            )
        updated = self.db.update_task_ordering(
            "task-1", "admin", rank=100, priority="urgent",
            now="2026-07-15T10:02:00+08:00", is_admin=True,
        )
        self.assertEqual(updated["rank"], 2)
        self.assertEqual(updated["priority"], "urgent")
        ordered = self.db.list_tasks("2026-07-15T10:03:00+08:00")
        self.assertEqual([item["task_id"] for item in ordered], ["task-2", "task-1"])
        self.assertEqual([item["rank"] for item in ordered], [1, 2])
        logs = self.db.list_task_audit_logs("task-1")
        self.assertEqual({log["field_name"] for log in logs}, {"rank", "priority"})
        self.assertTrue(all(log["actor"] == "admin" for log in logs))
        shifted_logs = self.db.list_task_audit_logs("task-2")
        self.assertEqual(shifted_logs[0]["field_name"], "rank")
        self.assertEqual(shifted_logs[0]["actor"], "admin")
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.db.update_task_ordering(
                "task-1", "admin", rank=0, priority=None,
                now="2026-07-15T10:04:00+08:00", is_admin=True,
            )

    def test_completed_tasks_are_below_ranked_tasks_and_cannot_be_ranked(self) -> None:
        self.db.create_task(task("task-low"), 1, "2026-07-15T10:00:00+08:00")
        self.db.create_task(task("task-high"), 1, "2026-07-15T10:01:00+08:00")
        self.db.create_task(task("task-completed"), 1, "2026-07-15T10:02:00+08:00")
        self.db.update_task_ordering(
            "task-low", "admin", rank=10, priority=None,
            now="2026-07-15T10:03:00+08:00", is_admin=True,
        )
        self.db.update_task_ordering(
            "task-high", "admin", rank=100, priority=None,
            now="2026-07-15T10:04:00+08:00", is_admin=True,
        )
        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE tasks SET status='completed',rank=1000 WHERE task_id='task-completed'"
            )
        created = self.db.create_task(
            task("task-new"), 1, "2026-07-15T10:04:30+08:00"
        )
        self.assertEqual(created["rank"], 3)
        with self.db.transaction() as connection:
            connection.execute("UPDATE tasks SET rank=10 WHERE task_id='task-low'")
            connection.execute("UPDATE tasks SET rank=20 WHERE task_id='task-high'")
            connection.execute("UPDATE tasks SET rank=30 WHERE task_id='task-new'")

        ordered = self.db.list_tasks("2026-07-15T10:05:00+08:00")
        self.assertEqual(
            [item["task_id"] for item in ordered],
            ["task-low", "task-high", "task-new", "task-completed"],
        )
        self.assertEqual([item["rank"] for item in ordered[:3]], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "does not participate"):
            self.db.update_task_ordering(
                "task-completed", "admin", rank=2000, priority=None,
                now="2026-07-15T10:06:00+08:00", is_admin=True,
            )

    def test_publisher_deletes_active_part_and_cascades_timing_session(self) -> None:
        self.db.create_task(task(), 2, "2026-07-15T10:00:00+08:00")
        claimed = self.db.claim_next_part(
            "task-1", "worker", "2026-07-15T10:01:00+08:00"
        )
        with self.assertRaises(PermissionError):
            self.db.delete_part(
                "task-1", claimed["part_id"], "worker", "2026-07-15T10:02:00+08:00"
            )
        deleted = self.db.delete_part(
            "task-1", claimed["part_id"], "publisher", "2026-07-15T10:02:00+08:00"
        )
        self.assertEqual(deleted["status"], "in_progress")
        remaining = self.db.list_parts("task-1", "2026-07-15T10:03:00+08:00")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["status"], "pending")
        with self.db.connection() as connection:
            sessions = connection.execute(
                "SELECT COUNT(*) FROM part_work_sessions WHERE part_id=?", (claimed["part_id"],)
            ).fetchone()[0]
        self.assertEqual(sessions, 0)
        log = self.db.list_task_audit_logs("task-1")[0]
        self.assertEqual(log["action"], "delete_part")
        self.assertEqual(log["actor"], "publisher")

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
        with self.assertRaisesRegex(PermissionError, "publisher, manager or admin"):
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

    def test_delete_user_transfers_tasks_and_returns_active_parts(self) -> None:
        self.db.create_user("admin", "admin-hash", "admin", "管理员", "2026-07-15T10:00:00+08:00")
        self.db.create_user("publisher", "user-hash", "user", "待删除", "2026-07-15T10:00:00+08:00")
        self.db.create_task(task(), 1, "2026-07-15T10:01:00+08:00")
        claimed = self.db.claim_next_part(
            "task-1", "publisher", "2026-07-15T10:02:00+08:00"
        )

        summary = self.db.delete_user(
            "publisher", "admin", "2026-07-15T10:03:00+08:00"
        )
        self.assertEqual(summary["transferred_tasks"], 1)
        self.assertEqual(summary["released_parts"], 1)
        with self.assertRaises(KeyError):
            self.db.get_user("publisher")
        transferred = self.db.get_task("task-1", "2026-07-15T10:04:00+08:00")
        self.assertEqual(transferred["publisher"], "admin")
        self.assertEqual(transferred["manager"], "admin")
        returned = self.db.list_parts("task-1", "2026-07-15T10:04:00+08:00")[0]
        self.assertEqual(returned["part_id"], claimed["part_id"])
        self.assertEqual(returned["status"], "pending")
        self.assertEqual(returned["annotator"], "")
        log = self.db.list_task_audit_logs("task-1")[0]
        self.assertEqual(log["action"], "delete_user")
        self.assertEqual(log["actor"], "admin")

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
        self.assertEqual(loaded["priority"], "medium")
        self.assertEqual(loaded["rank"], 1)


if __name__ == "__main__":
    unittest.main()
