from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

import workflow_platform.server as platform
from workflow_platform.server import (
    CreateTaskReq,
    LocateAnythingSettingsReq,
    health,
    locateanything_remote_video_path,
    parse_classes_text,
)


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
        self.assertEqual(payload["api_schema_version"], 4)
        self.assertEqual(payload["database"]["quick_check"], "ok")

    def test_task_creation_requires_responsible_people(self) -> None:
        with self.assertRaises(HTTPException) as context:
            asyncio.run(platform.create_task(CreateTaskReq(name="缺少负责人")))
        self.assertEqual(context.exception.status_code, 400)

    def test_orphaned_legacy_task_can_assign_responsible_people(self) -> None:
        old_settings = dict(platform.SETTINGS)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform.SETTINGS["tasks_dir"] = root
            platform.SETTINGS["database_path"] = str(root / "metadata.sqlite3")
            task_id = "legacy-orphan"
            (root / task_id).mkdir()
            try:
                platform.database().save_task({
                    "task_id": task_id,
                    "name": "旧版未指定人员任务",
                    "publisher": "",
                    "manager": "",
                    "annotators": [],
                    "created_at": "2026-07-02T10:00:00+08:00",
                    "updated_at": "2026-07-02T10:00:00+08:00",
                })
                response = asyncio.run(platform.update_task(
                    task_id,
                    platform.UpdateTaskReq(
                        actor="接管人",
                        publisher="发布人",
                        manager="负责人",
                        annotators="标注员甲, 标注员乙",
                    ),
                ))
                self.assertEqual(response["task"]["publisher"], "发布人")
                self.assertEqual(response["task"]["manager"], "负责人")
                self.assertEqual(response["task"]["annotators"], ["标注员甲", "标注员乙"])
            finally:
                platform.SETTINGS.clear()
                platform.SETTINGS.update(old_settings)

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
                    task_type="general",
                    publisher="publisher",
                    manager="manager",
                    annotators="ann-a, ann-b",
                    instructions="任务说明",
                    part_count=3,
                )))
                task_id = response["task"]["task_id"]
                detail = asyncio.run(platform.get_task(task_id))
                self.assertEqual(len(detail["classes"]), 2)
                self.assertEqual(detail["events"][0]["message"], "Task created")
                self.assertEqual(detail["publisher"], "publisher")
                self.assertEqual(detail["annotators"], ["ann-a", "ann-b"])
                self.assertEqual(detail["part_summary"]["total"], 3)
                claimed = asyncio.run(platform.claim_next_part(
                    task_id,
                    platform.ActorReq(actor="ann-a"),
                ))
                self.assertEqual(claimed["part"]["status"], "in_progress")
                submitted = asyncio.run(platform.submit_part(
                    task_id,
                    claimed["part"]["part_id"],
                    platform.SubmitPartReq(actor="ann-a", note="完成"),
                ))
                self.assertEqual(submitted["part"]["status"], "submitted")
                reviewed = asyncio.run(platform.review_part(
                    task_id,
                    claimed["part"]["part_id"],
                    platform.ReviewPartReq(actor="manager", action="approve", note="通过"),
                ))
                self.assertEqual(reviewed["part"]["status"], "completed")
                with self.assertRaises(HTTPException):
                    asyncio.run(platform.claim_next_part(
                        task_id,
                        platform.ActorReq(actor="outsider"),
                    ))
                self.assertTrue((root / "metadata.sqlite3").exists())
                self.assertFalse((root / task_id / "task.json").exists())
            finally:
                platform.SETTINGS.clear()
                platform.SETTINGS.update(old_settings)

    def test_video_task_locany_settings_are_saved_without_password(self) -> None:
        old_settings = dict(platform.SETTINGS)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform.SETTINGS["tasks_dir"] = root
            platform.SETTINGS["database_path"] = str(root / "metadata.sqlite3")
            try:
                created = asyncio.run(platform.create_task(CreateTaskReq(
                    name="Video Task",
                    task_type="video_detection",
                    publisher="publisher",
                    manager="manager",
                    prompt="person",
                    classes="0 person\n1 car",
                )))
                task_id = created["task"]["task_id"]
                response = asyncio.run(platform.save_task_locany_settings(
                    task_id,
                    LocateAnythingSettingsReq(
                        actor="manager",
                        prompt="person car",
                        classes="0 person\n1 car",
                        server_url="http://gpu-server:9010",
                        video_transfer="path",
                        local_path_prefix="D:/shared/videos",
                        remote_path_prefix="/data/shared/videos",
                        sftp_host="gpu-server",
                        sftp_port=22,
                        sftp_username="annotator",
                        sftp_password="secret",
                        sftp_password_env="LOCANY_SFTP_PASSWORD",
                        sftp_remote_dir="/data/cache/locany",
                        device="cuda:1",
                        dtype="bf16",
                    ),
                ))
                self.assertEqual(response["prompt"], "person car")
                self.assertEqual(response["settings"]["device"], "cuda:1")
                self.assertFalse("sftp_password" in response["settings"])
                config_path = root / task_id / "config.json"
                self.assertTrue(config_path.exists())
                payload = config_path.read_text(encoding="utf-8")
                self.assertIn('"server_url": "http://gpu-server:9010"', payload)
                self.assertNotIn("secret", payload)
            finally:
                platform.SETTINGS.clear()
                platform.SETTINGS.update(old_settings)

    def test_locany_path_mapping_uses_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_root = root / "shared" / "videos"
            video_root.mkdir(parents=True)
            video = video_root / "clip.mp4"
            video.write_bytes(b"demo")
            remote = locateanything_remote_video_path(
                video,
                {
                    "video_transfer": "path",
                    "local_path_prefix": str(video_root),
                    "remote_path_prefix": "/data/shared/videos",
                },
                root,
            )
            self.assertEqual(remote, "/data/shared/videos/clip.mp4")

    def test_auth_bootstrap_login_and_admin_user_creation(self) -> None:
        old_settings = dict(platform.SETTINGS)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform.SETTINGS["tasks_dir"] = root
            platform.SETTINGS["database_path"] = str(root / "metadata.sqlite3")
            try:
                client = TestClient(platform.app)
                me = client.get("/api/auth/me")
                self.assertEqual(me.status_code, 200)
                self.assertTrue(me.json()["bootstrap_required"])

                bootstrap = client.post("/api/auth/bootstrap-admin", json={
                    "username": "admin",
                    "password": "admin-pass-123",
                    "display_name": "Admin",
                })
                self.assertEqual(bootstrap.status_code, 200)
                self.assertEqual(bootstrap.json()["user"]["role"], "admin")

                users = client.get("/api/users")
                self.assertEqual(users.status_code, 200)
                self.assertEqual(len(users.json()["users"]), 1)

                created = client.post("/api/users", json={
                    "username": "worker",
                    "password": "worker-pass-123",
                    "role": "user",
                    "display_name": "Worker",
                })
                self.assertEqual(created.status_code, 200)
                self.assertEqual(created.json()["user"]["username"], "worker")

                client.post("/api/auth/logout")
                denied = client.get("/api/tasks")
                self.assertEqual(denied.status_code, 401)

                login = client.post("/api/auth/login", json={
                    "username": "worker",
                    "password": "worker-pass-123",
                })
                self.assertEqual(login.status_code, 200)
                me_again = client.get("/api/auth/me")
                self.assertEqual(me_again.status_code, 200)
                self.assertEqual(me_again.json()["user"]["username"], "worker")
            finally:
                platform.SETTINGS.clear()
                platform.SETTINGS.update(old_settings)


if __name__ == "__main__":
    unittest.main()
