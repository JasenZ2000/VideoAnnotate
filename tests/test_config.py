from __future__ import annotations

import json
import unittest
from pathlib import Path

from utils.mot_pipeline.config import DEFAULT_CONFIG, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_valid_and_safe(self) -> None:
        path = PROJECT_ROOT / "config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["sam31"]["server_url"], "")
        self.assertEqual(payload["locateanything"]["server_url"], "")
        self.assertEqual(payload["sam31"]["sftp_username"], "")
        self.assertEqual(payload["locateanything"]["sftp_username"], "")

    def test_external_config_is_merged_with_defaults(self) -> None:
        config = load_config(str(PROJECT_ROOT / "config.json"))
        self.assertEqual(config["tracking"]["method"], "sparse_track")
        self.assertIn("overview_filename", config["clips"])
        self.assertEqual(config["annotator"], DEFAULT_CONFIG["annotator"])
        self.assertEqual(config["locateanything"]["sftp_password_env"], "LOCANY_SFTP_PASSWORD")

    def test_example_configs_are_valid_json(self) -> None:
        for path in (PROJECT_ROOT / "configs").glob("*.json"):
            with self.subTest(path=path):
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
