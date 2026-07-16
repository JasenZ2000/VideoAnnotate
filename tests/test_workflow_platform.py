from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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

    def test_health_and_authentication_contract(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.json()["api_schema_version"], 8)
        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.get("/api/tasks").status_code, 401)


if __name__ == "__main__":
    unittest.main()
