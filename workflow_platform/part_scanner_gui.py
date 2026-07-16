from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from workflow_platform.part_directory_scanner import ScanResult, scan_part_directories


class PartDirectoryScannerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.result: ScanResult | None = None
        self.setWindowTitle("标注 Part 工作目录扫描工具")
        self.resize(980, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        body = QWidget()
        root_layout = QVBoxLayout(body)
        root_layout.setContentsMargins(22, 18, 22, 20)
        root_layout.setSpacing(13)

        title = QLabel("Part 工作目录扫描工具")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        root_layout.addWidget(title)
        intro = QLabel(
            "选择数据集根目录后自动寻找真正的工作目录；结果可直接复制并粘贴到多人标注平台的“Part 工作目录清单”。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#667085")
        root_layout.addWidget(intro)

        settings = QGroupBox("1. 选择目录和识别规则")
        grid = QGridLayout(settings)
        self.root_path = QLineEdit()
        self.root_path.setPlaceholderText(r"例如 D:\dataset 或 \\server\share\dataset")
        browse = QPushButton("选择根目录…")
        browse.clicked.connect(self._choose_root)
        self.max_depth = QSpinBox()
        self.max_depth.setRange(0, 32)
        self.max_depth.setValue(4)
        self.markers = QLineEdit("images, labels, annotations")
        self.minimum = QSpinBox()
        self.minimum.setRange(1, 3)
        self.minimum.setValue(1)
        self.full_paths = QCheckBox("复制完整路径（通常不勾选，推荐复制相对路径）")
        self.full_paths.stateChanged.connect(self._refresh_preview)
        depth_help = QLabel(
            "深度说明：根目录是 0，其直接子目录是 1。发现某目录直接包含识别标志后，就把该目录作为 Part，并停止扫描其内部。"
        )
        depth_help.setWordWrap(True)
        depth_help.setStyleSheet("color:#667085")
        grid.addWidget(QLabel("数据集根目录"), 0, 0)
        grid.addWidget(self.root_path, 0, 1, 1, 3)
        grid.addWidget(browse, 0, 4)
        grid.addWidget(QLabel("最大深度"), 1, 0)
        grid.addWidget(self.max_depth, 1, 1)
        grid.addWidget(QLabel("识别标志目录"), 1, 2)
        grid.addWidget(self.markers, 1, 3, 1, 2)
        grid.addWidget(QLabel("最少命中几个标志"), 2, 0)
        grid.addWidget(self.minimum, 2, 1)
        grid.addWidget(self.full_paths, 2, 2, 1, 3)
        grid.addWidget(depth_help, 3, 0, 1, 5)
        root_layout.addWidget(settings)

        action_row = QHBoxLayout()
        self.scan_button = QPushButton("开始扫描")
        self.scan_button.setMinimumHeight(40)
        self.scan_button.clicked.connect(self._scan)
        self.copy_button = QPushButton("复制清单")
        self.copy_button.clicked.connect(self._copy)
        self.copy_button.setEnabled(False)
        self.save_button = QPushButton("另存为 TXT…")
        self.save_button.clicked.connect(self._save)
        self.save_button.setEnabled(False)
        action_row.addWidget(self.scan_button)
        action_row.addStretch()
        action_row.addWidget(self.copy_button)
        action_row.addWidget(self.save_button)
        root_layout.addLayout(action_row)

        result_group = QGroupBox("2. 扫描结果")
        result_layout = QVBoxLayout(result_group)
        self.summary = QLabel("尚未扫描")
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["清单内容", "深度", "命中的标志", "完整目录"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview = QPlainTextEdit()
        self.preview.setPlaceholderText("这里会显示最终复制到平台的清单，可在复制前手工修改。")
        self.preview.setMinimumHeight(120)
        result_layout.addWidget(self.summary)
        result_layout.addWidget(self.table, 1)
        result_layout.addWidget(QLabel("最终清单（允许手工修改）"))
        result_layout.addWidget(self.preview)
        root_layout.addWidget(result_group, 1)

        self.setCentralWidget(body)
        self.setStyleSheet(
            "QGroupBox{font-weight:600;border:1px solid #d0d5dd;border-radius:8px;margin-top:10px;padding:14px 10px 10px}"
            "QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px}"
            "QLineEdit,QSpinBox,QPlainTextEdit,QTableWidget{padding:6px;border:1px solid #b8c0cc;border-radius:5px}"
            "QPushButton{padding:7px 14px}"
        )

    def _choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集根目录", self.root_path.text())
        if path:
            self.root_path.setText(path)

    def _marker_values(self) -> list[str]:
        return [value.strip() for value in self.markers.text().replace("，", ",").split(",") if value.strip()]

    def _scan(self) -> None:
        try:
            markers = self._marker_values()
            self.minimum.setMaximum(max(1, len(markers)))
            result = scan_part_directories(
                self.root_path.text().strip(),
                max_depth=self.max_depth.value(),
                marker_directories=markers,
                minimum_marker_count=self.minimum.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "扫描失败", str(exc))
            return
        self.result = result
        self.table.setRowCount(len(result.items))
        for row, item in enumerate(result.items):
            values = [item.relative_path, str(item.depth), ", ".join(item.matched_markers), item.full_path]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, cell)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, min(360, max(180, self.table.columnWidth(0))))
        self._refresh_preview()
        warning = f"；{len(result.warnings)} 个目录无法读取" if result.warnings else ""
        self.summary.setText(f"找到 {len(result.items)} 个工作目录{warning}。请检查列表后复制到平台。")
        self.copy_button.setEnabled(bool(result.items))
        self.save_button.setEnabled(bool(result.items))
        if not result.items:
            QMessageBox.information(
                self, "没有找到工作目录",
                "请适当增加最大深度，或确认工作目录直接包含 images、labels、annotations 等识别标志。",
            )

    def _refresh_preview(self) -> None:
        if not self.result:
            return
        self.preview.setPlainText("\n".join(self.result.manifest_lines(self.full_paths.isChecked())))

    def _manifest_text(self) -> str:
        return self.preview.toPlainText().strip()

    def _copy(self) -> None:
        text = self._manifest_text()
        if not text:
            QMessageBox.warning(self, "清单为空", "没有可复制的目录清单。")
            return
        QApplication.clipboard().setText(text)
        self.summary.setText(f"已复制 {len(text.splitlines())} 行，可切换到多人标注平台直接粘贴。")

    def _save(self) -> None:
        text = self._manifest_text()
        if not text:
            QMessageBox.warning(self, "清单为空", "没有可保存的目录清单。")
            return
        default = str(Path(self.root_path.text()).parent / "part_directories.txt")
        path, _ = QFileDialog.getSaveFileName(self, "保存 Part 目录清单", default, "文本文件 (*.txt)")
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8-sig")
            self.summary.setText(f"清单已保存到：{path}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Part 工作目录扫描工具")
    window = PartDirectoryScannerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
