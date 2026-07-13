import os
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from locany_batch_tool.gui import MainWindow, TaskWorker


class LocateAnythingQtTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_background_worker_is_retained_until_completion(self) -> None:
        window = MainWindow()
        results = []
        loop = QEventLoop()
        worker = TaskWorker(lambda _: {"ok": True})
        window._launch(worker, results.append)
        self.assertIs(window.worker, worker)
        assert window.thread is not None
        window.thread.finished.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        self.assertEqual(results, [{"ok": True}])
        self.assertIsNone(window.worker)
        window.close()
