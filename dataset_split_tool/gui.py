from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dataset_split_tool.core import scan_samples, split_dataset


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


class Worker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable[[], object]) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.succeeded.emit(self.operation())
        except BaseException as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("标注数据汇总均分工具")
        self.resize(760, 520)
        self._worker: Worker | None = None

        base = application_dir()
        self.ori_edit = QLineEdit(str(base / "ori"))
        self.output_edit = QLineEdit(str(base / "split_output"))
        self.parts_spin = QSpinBox()
        self.parts_spin.setRange(1, 10000)
        self.parts_spin.setValue(2)
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        ori_row = QWidget()
        ori_layout = QHBoxLayout(ori_row)
        ori_layout.setContentsMargins(0, 0, 0, 0)
        ori_layout.addWidget(self.ori_edit)
        ori_button = QPushButton("选择")
        ori_button.clicked.connect(self._choose_ori)
        ori_layout.addWidget(ori_button)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit)
        output_button = QPushButton("选择")
        output_button.clicked.connect(self._choose_output)
        output_layout.addWidget(output_button)

        form = QFormLayout()
        form.addRow("源目录（默认同目录 ori）：", ori_row)
        form.addRow("新输出目录：", output_row)
        form.addRow("均分份数：", self.parts_spin)

        self.scan_button = QPushButton("扫描检查")
        self.scan_button.clicked.connect(self._scan)
        self.run_button = QPushButton("开始汇总并均分")
        self.run_button.clicked.connect(self._run)
        buttons = QHBoxLayout()
        buttons.addWidget(self.scan_button)
        buttons.addWidget(self.run_button)

        note = QLabel(
            "每个样本必须具有同名的图片、YOLO TXT 和 VOC XML。"
            "输出为 part_001、part_002…，每份均包含 images / labels / annotations。"
        )
        note.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.log, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _choose_ori(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 ori 目录", self.ori_edit.text())
        if path:
            self.ori_edit.setText(path)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录的上级目录",
            str(Path(self.output_edit.text()).parent),
        )
        if path:
            self.output_edit.setText(str(Path(path) / "split_output"))

    def _set_busy(self, busy: bool) -> None:
        self.scan_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)

    def _launch(self, operation: Callable[[], object], callback: Callable[[object], None]) -> None:
        if self._worker is not None:
            return
        self._set_busy(True)
        worker = Worker(operation)
        self._worker = worker
        worker.succeeded.connect(callback)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.start()

    def _finished(self) -> None:
        self._set_busy(False)
        self._worker = None

    def _failed(self, message: str) -> None:
        self.log.setPlainText(message)
        QMessageBox.critical(self, "处理失败", message.split("\n", 1)[0])

    def _scan(self) -> None:
        ori = self.ori_edit.text().strip()
        self.log.setPlainText("正在扫描……")
        self._launch(lambda: scan_samples(ori), self._show_scan)

    def _show_scan(self, result: object) -> None:
        scan = result
        lines = [
            f"扫描完成：找到 {len(scan.dataset_roots)} 个数据集目录",
            f"完整配对样本：{len(scan.samples)} 个",
        ]
        lines.extend(f"提示：{warning}" for warning in scan.warnings)
        self.log.setPlainText("\n".join(lines))

    def _run(self) -> None:
        ori = self.ori_edit.text().strip()
        output = self.output_edit.text().strip()
        parts = self.parts_spin.value()
        answer = QMessageBox.question(
            self,
            "确认执行",
            f"将 {ori} 中的完整样本汇总并均分为 {parts} 份，输出到：\n{output}\n\n继续吗？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.log.setPlainText("正在复制并均分……")
        self._launch(lambda: split_dataset(ori, output, parts), self._show_result)

    def _show_result(self, result: object) -> None:
        payload = result
        lines = [
            "处理完成",
            f"完整样本：{payload['samples']} 个",
            f"分份数量：{payload['parts']} 份",
            f"各份样本数：{', '.join(str(value) for value in payload['counts'])}",
            f"输出目录：{payload['output_dir']}",
        ]
        lines.extend(f"提示：{warning}" for warning in payload["warnings"])
        self.log.setPlainText("\n".join(lines))
        QMessageBox.information(self, "处理完成", f"已输出到：\n{payload['output_dir']}")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
