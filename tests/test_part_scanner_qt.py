from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from workflow_platform.part_scanner_gui import PartDirectoryScannerWindow


class PartScannerQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_window_scans_and_prepares_copyable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "group" / "split_001" / "images").mkdir(parents=True)
            window = PartDirectoryScannerWindow()
            window.root_path.setText(str(root))
            window._scan()
            self.assertEqual(window.table.rowCount(), 1)
            self.assertEqual(window.preview.toPlainText(), "group/split_001")
            self.assertTrue(window.copy_button.isEnabled())
            window._copy()
            self.assertEqual(QApplication.clipboard().text(), "group/split_001")
            window.close()


if __name__ == "__main__":
    unittest.main()
