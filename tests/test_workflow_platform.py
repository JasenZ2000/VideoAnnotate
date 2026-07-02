from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException

from workflow_platform.server import health, parse_classes_text


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


if __name__ == "__main__":
    unittest.main()
