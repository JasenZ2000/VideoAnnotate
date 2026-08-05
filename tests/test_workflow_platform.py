from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import workflow_platform.server as platform


TABLE = """申请日期\t申请人\t项目\t标注内容\t数据集溯源\t每小时可标\t数据量\t预计工时/单人\t数据路径\t标注说明书路径
2022/8/10\t刘湛基\t行人检测\t手工清洗相似度高的图片\t\t1000\t15074\t15\t\\\\server\\data\t\\\\server\\guide.pdf
2022/8/18\t刘湛基\t反光衣客诉\t清洗图片\t客户视频\t2000\t24000\t12\t\\\\server\\data2\t\\\\server\\guide2.pdf"""


class WorkflowPlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.old_settings = dict(platform.SETTINGS)
        platform.SETTINGS["data_dir"] = Path(self.temporary.name)
        platform.SETTINGS["database_path"] = str(Path(self.temporary.name) / "metadata.sqlite3")
        self.client = TestClient(platform.app)
        response = self.client.post("/api/auth/bootstrap-admin", json={
            "username": "publisher", "password": "publisher-pass", "display_name": "发布者"
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.client.post("/api/users", json={
            "username": "worker", "password": "worker-pass", "display_name": "标注员"
        })

    def tearDown(self) -> None:
        self.client.close()
        platform.SETTINGS.clear()
        platform.SETTINGS.update(self.old_settings)
        self.temporary.cleanup()

    def _publish(self, table: str = TABLE, count: int = 2) -> dict:
        response = self.client.post("/api/tasks", json={
            "clipboard_text": table, "product_tag": "BSD", "part_count": count,
            "part_prefix": "BSD",
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_table_parser_supports_header_and_headerless_rows(self) -> None:
        rows = platform.parse_spreadsheet_rows(TABLE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["project"], "行人检测")
        self.assertEqual(rows[0]["data_path"], r"\\server\data")
        headerless = platform.parse_spreadsheet_rows(TABLE.splitlines()[1])
        self.assertEqual(headerless[0]["applicant"], "刘湛基")

    def test_part_manifest_builds_nested_work_paths_and_rejects_duplicates(self) -> None:
        specs = platform.parse_part_manifest(
            "split_001\ncamera_a/split_002\n特殊任务\tcamera_b/split_003",
            r"\\server\dataset",
        )
        self.assertEqual([item["name"] for item in specs], [
            "split_001", "camera_a / split_002", "特殊任务",
        ])
        self.assertEqual(specs[1]["work_path"], r"\\server\dataset\camera_a\split_002")
        with self.assertRaisesRegex(ValueError, "重复工作目录"):
            platform.parse_part_manifest("split_001\nsplit_001", r"\\server\dataset")

    def test_publish_with_part_manifest_exposes_work_directory_to_worker(self) -> None:
        row = TABLE.splitlines()[1]
        response = self.client.post("/api/tasks", json={
            "clipboard_text": row,
            "product_tag": "BSD",
            "part_count": 0,
            "part_manifest": "split_001\ngroup_a/split_002",
        })
        self.assertEqual(response.status_code, 200, response.text)
        task_id = response.json()["tasks"][0]["task_id"]
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertEqual([part["name"] for part in detail["parts"]], [
            "split_001", "group_a / split_002",
        ])
        self.assertEqual(
            detail["parts"][1]["work_path"], r"\\server\data\group_a\split_002",
        )

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "worker", "password": "worker-pass"})
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next").json()["part"]
        self.assertEqual(claimed["work_path"], r"\\server\data\split_001")

    def test_manifest_publish_requires_one_task_row(self) -> None:
        response = self.client.post("/api/tasks/preview", json={
            "clipboard_text": TABLE,
            "product_tag": "BSD",
            "part_count": 0,
            "part_manifest": "split_001",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("一次只能发布一行任务", response.text)

    def test_publish_multiple_rows_and_publisher_detail(self) -> None:
        created = self._publish()
        self.assertEqual(created["count"], 2)
        task_id = created["tasks"][0]["task_id"]
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertTrue(detail["is_publisher"])
        self.assertEqual(detail["part_summary"]["total"], 2)
        self.assertEqual([part["name"] for part in detail["parts"]],
                         ["BSD_part_001", "BSD_part_002"])
        self.assertIn("statistics", detail)

    def test_worker_claim_submit_rework_resubmit_and_approve(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 2)["tasks"][0]["task_id"]
        self.client.post("/api/auth/logout")
        login = self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass"
        })
        self.assertEqual(login.status_code, 200)
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next")
        self.assertEqual(claimed.status_code, 200, claimed.text)
        part_id = claimed.json()["part"]["part_id"]
        submitted = self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/submit", json={"note": "有两张模糊图"}
        )
        self.assertEqual(submitted.json()["part"]["status"], "submitted")
        denied = self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/review",
            json={"action": "approve", "note": "越权审核"},
        )
        self.assertEqual(denied.status_code, 403)

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "publisher", "password": "publisher-pass"})
        reviewed = self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/review",
            json={"action": "rework", "note": "请补充漏标"},
        )
        self.assertEqual(reviewed.json()["part"]["status"], "rework")

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "worker", "password": "worker-pass"})
        self.assertEqual(
            self.client.post(f"/api/tasks/{task_id}/parts/{part_id}/start-rework").status_code, 200
        )
        self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/submit", json={"note": "已经修改"}
        )

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "publisher", "password": "publisher-pass"})
        approved = self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/review",
            json={"action": "approve", "note": "通过"},
        )
        self.assertEqual(approved.json()["part"]["status"], "completed")
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertEqual(detail["statistics"][0]["completed"], 1)

    def test_admin_can_review_parts_from_another_publishers_task(self) -> None:
        self.client.post("/api/users", json={
            "username": "reviewer", "password": "reviewer-pass",
            "display_name": "审核管理员", "role": "admin",
        })
        task_id = self._publish(TABLE.splitlines()[1], 2)["tasks"][0]["task_id"]

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "worker", "password": "worker-pass"})
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next").json()["part"]
        self.client.post(
            f"/api/tasks/{task_id}/parts/{claimed['part_id']}/submit",
            json={"note": "等待管理员审核"},
        )

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "reviewer", "password": "reviewer-pass",
        })
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertFalse(detail["is_publisher"])
        self.assertTrue(detail["can_review"])
        self.assertEqual(len(detail["parts"]), 2)
        self.assertNotIn("statistics", detail)

        reviewed = self.client.post(
            f"/api/tasks/{task_id}/parts/{claimed['part_id']}/review",
            json={"action": "approve", "note": "管理员检查通过"},
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        self.assertEqual(reviewed.json()["part"]["status"], "completed")
        self.assertEqual(reviewed.json()["part"]["comments"][-1]["actor"], "reviewer")

    def test_worker_cannot_see_statistics_or_add_parts(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 1)["tasks"][0]["task_id"]
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "worker", "password": "worker-pass"})
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertNotIn("statistics", detail)
        denied = self.client.post(f"/api/tasks/{task_id}/parts", json={"count": 2})
        self.assertEqual(denied.status_code, 403)

    def test_publisher_can_claim_edit_and_delete_own_task(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 2)["tasks"][0]["task_id"]
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next")
        self.assertEqual(claimed.status_code, 200, claimed.text)
        self.assertEqual(claimed.json()["part"]["annotator"], "publisher")

        edited = self.client.patch(f"/api/tasks/{task_id}", json={
            "product_tag": "AEB", "project": "车辆检测", "data_path": r"\\server\new"
        })
        self.assertEqual(edited.status_code, 200, edited.text)
        self.assertEqual(edited.json()["task"]["product_tag"], "AEB")
        self.assertEqual(edited.json()["task"]["project"], "车辆检测")

        deleted = self.client.delete(f"/api/tasks/{task_id}")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(self.client.get(f"/api/tasks/{task_id}").status_code, 404)

    def test_non_publisher_cannot_edit_or_delete_task(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 1)["tasks"][0]["task_id"]
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "worker", "password": "worker-pass"})
        self.assertEqual(
            self.client.patch(f"/api/tasks/{task_id}", json={"project": "越权"}).status_code,
            403,
        )
        self.assertEqual(self.client.delete(f"/api/tasks/{task_id}").status_code, 403)

    def test_publisher_can_assign_collaborative_viewer(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 2)["tasks"][0]["task_id"]
        updated = self.client.patch(f"/api/tasks/{task_id}", json={
            "manager": "worker", "expected_part_seconds": 600,
        })
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["task"]["manager"], "worker")
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass",
        })
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertTrue(detail["is_manager"])
        self.assertTrue(detail["can_view_all"])
        self.assertTrue(detail["can_review"])
        self.assertEqual(len(detail["parts"]), 2)
        self.assertIn("statistics", detail)
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next").json()["part"]
        submitted = self.client.post(
            f"/api/tasks/{task_id}/parts/{claimed['part_id']}/submit", json={"note": "完成"},
        )
        self.assertEqual(submitted.status_code, 200, submitted.text)
        task_list = self.client.get("/api/tasks").json()["tasks"]
        summary = next(item for item in task_list if item["task_id"] == task_id)["part_summary"]
        self.assertEqual(summary["annotated"], 1)
        self.assertEqual(summary["completed"], 0)
        reviewed = self.client.post(
            f"/api/tasks/{task_id}/parts/{claimed['part_id']}/review",
            json={"action": "approve", "note": "协同审核通过"},
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        self.assertEqual(reviewed.json()["part"]["status"], "completed")
        denied = self.client.patch(f"/api/tasks/{task_id}", json={"project": "denied"})
        self.assertEqual(denied.status_code, 403)

    def test_worker_can_pause_resume_and_return_part(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 1)["tasks"][0]["task_id"]
        self.assertEqual(self.client.patch(
            f"/api/tasks/{task_id}", json={"expected_part_seconds": 1},
        ).status_code, 200)
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass",
        })
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next").json()["part"]
        part_id = claimed["part_id"]
        paused = self.client.post(f"/api/tasks/{task_id}/parts/{part_id}/pause")
        self.assertEqual(paused.status_code, 200, paused.text)
        self.assertEqual(paused.json()["part"]["status"], "paused")
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "publisher", "password": "publisher-pass",
        })
        time_review = self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/time-review",
            json={"decision": "estimate_unreasonable", "note": "one second is too short"},
        )
        self.assertEqual(time_review.status_code, 200, time_review.text)
        self.assertEqual(time_review.json()["part"]["time_review_status"], "estimate_unreasonable")
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass",
        })
        resumed = self.client.post(f"/api/tasks/{task_id}/parts/{part_id}/resume")
        self.assertEqual(resumed.json()["part"]["status"], "in_progress")
        returned = self.client.post(
            f"/api/tasks/{task_id}/parts/{part_id}/return", json={"note": "switch worker"},
        )
        self.assertEqual(returned.status_code, 200, returned.text)
        self.assertEqual(returned.json()["part"]["status"], "pending")
        self.assertEqual(returned.json()["part"]["annotator"], "")

    def test_admin_orders_tasks_and_priority_change_is_audited(self) -> None:
        created = self._publish(TABLE, 1)["tasks"]
        other_id = created[0]["task_id"]
        target_id = created[1]["task_id"]
        moved_other = self.client.patch(
            f"/api/tasks/{other_id}/ordering", json={"rank": 10}
        )
        self.assertEqual(moved_other.status_code, 200, moved_other.text)
        updated = self.client.patch(f"/api/tasks/{target_id}/ordering", json={
            "rank": 1, "priority": "urgent",
        })
        self.assertEqual(updated.status_code, 200, updated.text)
        tasks = self.client.get("/api/tasks").json()["tasks"]
        self.assertEqual(tasks[0]["task_id"], target_id)
        self.assertEqual(tasks[0]["rank"], 1)
        self.assertEqual(tasks[0]["priority"], "urgent")
        detail = self.client.get(f"/api/tasks/{target_id}").json()
        self.assertEqual(
            {log["field_name"] for log in detail["audit_logs"]}, {"rank", "priority"}
        )
        self.assertTrue(all(log["actor"] == "publisher" for log in detail["audit_logs"]))

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass",
        })
        denied = self.client.patch(f"/api/tasks/{target_id}/ordering", json={"rank": 1000})
        self.assertEqual(denied.status_code, 403)

    def test_publisher_deletes_pending_and_active_parts(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 2)["tasks"][0]["task_id"]
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        pending_id = detail["parts"][0]["part_id"]
        deleted_pending = self.client.delete(f"/api/tasks/{task_id}/parts/{pending_id}")
        self.assertEqual(deleted_pending.status_code, 200, deleted_pending.text)
        self.assertEqual(deleted_pending.json()["summary"]["total"], 1)

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass",
        })
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next").json()["part"]
        denied = self.client.delete(f"/api/tasks/{task_id}/parts/{claimed['part_id']}")
        self.assertEqual(denied.status_code, 403)

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "publisher", "password": "publisher-pass",
        })
        deleted_active = self.client.delete(
            f"/api/tasks/{task_id}/parts/{claimed['part_id']}"
        )
        self.assertEqual(deleted_active.status_code, 200, deleted_active.text)
        self.assertEqual(deleted_active.json()["summary"]["total"], 0)
        final_detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertEqual(final_detail["parts"], [])
        self.assertEqual(
            [log["action"] for log in final_detail["audit_logs"]],
            ["delete_part", "delete_part"],
        )

    def test_health_and_authentication_contract(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.json()["api_schema_version"], 12)
        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.get("/api/tasks").status_code, 401)

    def test_admin_resets_and_deletes_user(self) -> None:
        task_id = self._publish(TABLE.splitlines()[1], 1)["tasks"][0]["task_id"]
        reset = self.client.patch("/api/users/worker", json={"password": "worker-new-pass"})
        self.assertEqual(reset.status_code, 200, reset.text)

        self.client.post("/api/auth/logout")
        old_login = self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-pass",
        })
        self.assertEqual(old_login.status_code, 401)
        new_login = self.client.post("/api/auth/login", json={
            "username": "worker", "password": "worker-new-pass",
        })
        self.assertEqual(new_login.status_code, 200, new_login.text)
        denied = self.client.delete("/api/users/publisher")
        self.assertEqual(denied.status_code, 403)
        claimed = self.client.post(f"/api/tasks/{task_id}/parts/claim-next")
        self.assertEqual(claimed.status_code, 200, claimed.text)

        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={
            "username": "publisher", "password": "publisher-pass",
        })
        self_delete = self.client.delete("/api/users/publisher")
        self.assertEqual(self_delete.status_code, 400)
        deleted = self.client.delete("/api/users/worker")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(deleted.json()["summary"]["released_parts"], 1)
        self.assertNotIn("worker", [u["username"] for u in self.client.get("/api/users").json()["users"]])
        detail = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertEqual(detail["parts"][0]["status"], "pending")

    def test_frontend_has_long_path_wrapping_and_clipboard_fallback(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("overflow-wrap:anywhere", page.text)
        self.assertIn("window.isSecureContext&&navigator.clipboard", page.text)
        self.assertIn("document.execCommand('copy')", page.text)
        self.assertIn("priorityName", page.text)
        self.assertIn("deletePart", page.text)
        self.assertIn("resetUserPassword", page.text)
        self.assertIn("deleteManagedUser", page.text)
        self.assertIn("completed=t.status==='completed'", page.text)
        self.assertIn("ac!==bc", page.text)
        self.assertIn("s.annotated", page.text)
        self.assertIn("协同审核", page.text)
        self.assertIn("任务操作日志", page.text)

    def test_main_enables_https_and_secure_cookie_with_pem_files(self) -> None:
        cert = Path(self.temporary.name) / "server.crt"
        key = Path(self.temporary.name) / "server.key"
        cert.write_text("test certificate", encoding="utf-8")
        key.write_text("test key", encoding="utf-8")
        with patch.object(platform, "database"), patch.object(platform.uvicorn, "run") as run:
            platform.main([
                "--host", "0.0.0.0", "--port", "8443",
                "--tasks-dir", self.temporary.name,
                "--database", str(Path(self.temporary.name) / "tls.sqlite3"),
                "--ssl-certfile", str(cert), "--ssl-keyfile", str(key),
            ])
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["ssl_certfile"], str(cert.resolve()))
        self.assertEqual(kwargs["ssl_keyfile"], str(key.resolve()))
        self.assertTrue(platform.SETTINGS["secure_cookie"])

    def test_main_auto_https_generates_certificate_in_tasks_directory(self) -> None:
        tasks_dir = Path(self.temporary.name) / "tasks"
        with (
            patch.object(platform, "database"),
            patch.object(platform, "discover_tls_hosts", return_value=["localhost", "127.0.0.1"]),
            patch.object(platform.uvicorn, "run") as run,
        ):
            platform.main([
                "--host", "0.0.0.0", "--port", "8443",
                "--tasks-dir", str(tasks_dir),
                "--database", str(tasks_dir / "platform.sqlite3"),
                "--auto-https", "--tls-hosts", "annotation-host,192.0.2.20",
            ])
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["ssl_certfile"], str((tasks_dir / "tls/selfsigned-cert.pem").resolve()))
        self.assertEqual(kwargs["ssl_keyfile"], str((tasks_dir / "tls/selfsigned-key.pem").resolve()))
        self.assertTrue(Path(kwargs["ssl_certfile"]).is_file())
        self.assertTrue(Path(kwargs["ssl_keyfile"]).is_file())
        self.assertTrue(platform.SETTINGS["secure_cookie"])


if __name__ == "__main__":
    unittest.main()
