from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import utils.annotator.server as annotator_server
from utils.annotator.server import (
    LocateAnythingRemoteReq,
    _locany_request_config,
    _project_config_path,
    _gpu_api_url,
    _public_locany_settings,
    _validate_locany_connection_config,
)


class AnnotatorLocateAnythingTests(unittest.TestCase):
    def test_project_config_fallback_stays_at_repository_root(self) -> None:
        with patch.dict(os.environ, {"ANNOTATOR_CONFIG": ""}):
            self.assertEqual(
                _project_config_path(),
                Path(__file__).resolve().parents[1] / "config.json",
            )

    def test_gpu_service_urls_keep_job_types_separate(self) -> None:
        self.assertEqual(
            _gpu_api_url("http://gpu-server:9010/", "sam31", "jobs/abc/tracking-results"),
            "http://gpu-server:9010/api/sam31/jobs/abc/tracking-results",
        )
        self.assertEqual(
            _gpu_api_url("http://gpu-server:9010", "locateanything", "/jobs/abc/yolo-zip"),
            "http://gpu-server:9010/api/locateanything/jobs/abc/yolo-zip",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported GPU service"):
            _gpu_api_url("http://gpu-server:9010", "unknown", "jobs")

    def test_request_connection_settings_override_workspace_config(self) -> None:
        config = {
            "locateanything": {
                "server_url": "http://old-server:9010",
                "video_transfer": "path",
                "local_path_prefix": "D:/old",
                "remote_path_prefix": "/old",
            }
        }
        req = LocateAnythingRemoteReq(
            prompt="person",
            server_url="http://gpu-server:9010/",
            video_transfer="sftp",
            sftp_host="gpu-server",
            sftp_port=2222,
            sftp_username="annotator",
            sftp_password="session-secret",
            sftp_remote_dir="/data/locany/videos",
            sftp_reuse_existing=False,
            cuda_device=1,
        )

        result = _locany_request_config(config, req)

        self.assertEqual(result["server_url"], "http://gpu-server:9010/")
        self.assertEqual(result["video_transfer"], "sftp")
        self.assertEqual(result["sftp_port"], 2222)
        self.assertEqual(result["sftp_password"], "session-secret")
        self.assertFalse(result["sftp_reuse_existing"])
        _validate_locany_connection_config(result)

    def test_public_settings_never_return_password(self) -> None:
        config = {
            "locateanything": {
                "server_url": "http://gpu-server:9010",
                "video_transfer": "sftp",
                "sftp_host": "gpu-server",
                "sftp_username": "annotator",
                "sftp_password": "must-not-leak",
                "sftp_password_env": "LOCANY_TEST_PASSWORD",
                "sftp_remote_dir": "/data/locany/videos",
            }
        }
        with patch.dict(os.environ, {"LOCANY_TEST_PASSWORD": "environment-secret"}):
            settings = _public_locany_settings(config)

        self.assertNotIn("sftp_password", settings)
        self.assertTrue(settings["sftp_password_configured"])
        self.assertEqual(settings["sftp_password_env"], "LOCANY_TEST_PASSWORD")

    def test_connection_validation_rejects_incomplete_sftp(self) -> None:
        with self.assertRaisesRegex(ValueError, "sftp_username"):
            _validate_locany_connection_config({
                "server_url": "http://gpu-server:9010",
                "video_transfer": "sftp",
                "sftp_host": "gpu-server",
                "sftp_remote_dir": "/data/locany/videos",
            })

    def test_connection_endpoint_checks_api_and_sftp(self) -> None:
        req = LocateAnythingRemoteReq(
            prompt="person",
            server_url="http://gpu-server:9010",
            video_transfer="sftp",
            sftp_host="gpu-server",
            sftp_username="annotator",
            sftp_password="session-secret",
            sftp_remote_dir="/data/locany/videos",
        )
        with (
            patch.object(annotator_server, "_json_http_request", return_value={"ok": True, "cache_dir": "/cache"}),
            patch.object(annotator_server, "_test_locany_sftp", return_value={"ok": True, "remote_dir_exists": True}),
        ):
            result = asyncio.run(annotator_server.test_locany_connection(req))

        self.assertTrue(result["ok"])
        self.assertEqual(result["api"]["cache_dir"], "/cache")
        self.assertTrue(result["sftp"]["ok"])


if __name__ == "__main__":
    unittest.main()
