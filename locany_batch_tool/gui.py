from __future__ import annotations

import copy
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QSettings, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)

from locany_batch_tool.server import (
    JOBS, BatchReq, ConnectionReq, _check_direct_capabilities, _connect_sftp,
    _json_request, _run_batch,
)
from locany_batch_tool.postprocess import organize_prelabels


class TaskWorker(QObject):
    progress = Signal(dict)
    succeeded = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, action: Callable[["TaskWorker"], dict[str, Any]]) -> None:
        super().__init__()
        self.action = action

    def run(self) -> None:
        try:
            self.succeeded.emit(self.action(self))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("VideoAnnotate", "LocateAnythingBatchTool")
        self.thread: QThread | None = None
        self.worker: TaskWorker | None = None
        self.setWindowTitle("LocateAnything 批量预标注工具")
        self.resize(920, 820)
        self._build_ui()
        self._load_settings()
        self._mode_changed()

    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(22, 18, 22, 22)
        root.setSpacing(14)
        title = QLabel("LocateAnything 批量预标注")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        root.addWidget(title)
        subtitle = QLabel("选择视频，测试连接，然后批量生成 YOLO TXT 与 Pascal VOC XML 标注。")
        subtitle.setStyleSheet("color:#667085")
        root.addWidget(subtitle)

        connection = QGroupBox("1. GPU 服务与传输方式")
        grid = QGridLayout(connection)
        self.server_url = QLineEdit("http://192.168.21.226:10114")
        self.mode = QComboBox(); self.mode.addItem("SFTP 上传并下载结果", "sftp"); self.mode.addItem("共享文件系统直连", "direct")
        self.cuda = QSpinBox(); self.cuda.setRange(0, 64)
        grid.addWidget(QLabel("GPU Services 地址"), 0, 0); grid.addWidget(self.server_url, 0, 1, 1, 3)
        grid.addWidget(QLabel("工作模式"), 1, 0); grid.addWidget(self.mode, 1, 1)
        grid.addWidget(QLabel("CUDA 设备号"), 1, 2); grid.addWidget(self.cuda, 1, 3)
        self.sftp_frame = QFrame(); sftp = QGridLayout(self.sftp_frame); sftp.setContentsMargins(0, 8, 0, 0)
        self.sftp_host = QLineEdit("192.168.21.226"); self.sftp_port = QSpinBox(); self.sftp_port.setRange(1,65535); self.sftp_port.setValue(22)
        self.sftp_user = QLineEdit("hx"); self.sftp_password = QLineEdit(); self.sftp_password.setEchoMode(QLineEdit.EchoMode.Password); self.sftp_password.setPlaceholderText("留空时读取 LOCANY_SFTP_PASSWORD")
        self.sftp_key = QLineEdit(); key_button = QPushButton("选择私钥…"); key_button.clicked.connect(self._choose_key)
        self.remote_dir = QLineEdit("/data2/DET_Group/ZZS/locateAnything/eagle/Embodied/fast_tmp")
        sftp.addWidget(QLabel("SFTP 主机"),0,0);sftp.addWidget(self.sftp_host,0,1);sftp.addWidget(QLabel("端口"),0,2);sftp.addWidget(self.sftp_port,0,3)
        sftp.addWidget(QLabel("用户名"),1,0);sftp.addWidget(self.sftp_user,1,1);sftp.addWidget(QLabel("密码"),1,2);sftp.addWidget(self.sftp_password,1,3)
        sftp.addWidget(QLabel("SSH 私钥"),2,0);sftp.addWidget(self.sftp_key,2,1,1,2);sftp.addWidget(key_button,2,3)
        sftp.addWidget(QLabel("远端缓存目录"),3,0);sftp.addWidget(self.remote_dir,3,1,1,3)
        grid.addWidget(self.sftp_frame,2,0,1,4)
        test_row = QHBoxLayout(); self.test_button=QPushButton("测试连接");self.test_button.clicked.connect(self._test_connection);self.test_result=QLabel("尚未测试")
        test_row.addWidget(self.test_button);test_row.addWidget(self.test_result,1);grid.addLayout(test_row,3,0,1,4)
        root.addWidget(connection)

        paths = QGroupBox("2. 视频与结果位置")
        form = QGridLayout(paths)
        self.input_path=QLineEdit(); self.input_path.setPlaceholderText("选择一个视频，或包含多个视频的目录")
        self.file_button=QPushButton("选择视频…"); self.file_button.clicked.connect(self._choose_video)
        self.input_dir_button=QPushButton("选择目录…"); self.input_dir_button.clicked.connect(self._choose_input_dir)
        self.output_label=QLabel("本地结果目录");self.output_path=QLineEdit();self.output_button=QPushButton("选择目录…");self.output_button.clicked.connect(self._choose_output_dir)
        form.addWidget(QLabel("输入"),0,0);form.addWidget(self.input_path,0,1);form.addWidget(self.file_button,0,2);form.addWidget(self.input_dir_button,0,3)
        form.addWidget(self.output_label,1,0);form.addWidget(self.output_path,1,1,1,2);form.addWidget(self.output_button,1,3)
        self.recursive=QCheckBox("递归查找子目录");self.reuse=QCheckBox("复用远端已有的同一视频");self.reuse.setChecked(True)
        form.addWidget(self.recursive,2,1);form.addWidget(self.reuse,2,2,1,2)
        root.addWidget(paths)

        inference = QGroupBox("3. 推理与类别")
        grid2=QGridLayout(inference)
        self.prompt=QLineEdit("person");self.task=QComboBox();self.task.addItems(["ground_multi","detect","ground_single"])
        self.dtype=QComboBox();self.dtype.addItems(["bf16","fp16","fp32"])
        self.frame_step=QSpinBox();self.frame_step.setRange(1,10000);self.frame_step.setValue(1)
        self.max_frames=QSpinBox();self.max_frames.setRange(0,100000000);self.max_frames.setSpecialValueText("全部")
        self.classes=QPlainTextEdit("0 person");self.classes.setMaximumHeight(100);self.classes.setPlaceholderText("每行填写：编号 类别名\n例如：0 person")
        grid2.addWidget(QLabel("Prompt"),0,0);grid2.addWidget(self.prompt,0,1);grid2.addWidget(QLabel("任务类型"),0,2);grid2.addWidget(self.task,0,3)
        grid2.addWidget(QLabel("精度"),1,0);grid2.addWidget(self.dtype,1,1);grid2.addWidget(QLabel("帧间隔"),1,2);grid2.addWidget(self.frame_step,1,3)
        grid2.addWidget(QLabel("最多处理帧数"),2,0);grid2.addWidget(self.max_frames,2,1)
        grid2.addWidget(QLabel("类别映射"),3,0);grid2.addWidget(self.classes,3,1,1,3)
        root.addWidget(inference)

        actions=QHBoxLayout();self.run_button=QPushButton("开始批量预标注");self.run_button.setMinimumHeight(42);self.run_button.clicked.connect(self._start_batch)
        actions.addStretch();actions.addWidget(self.run_button);root.addLayout(actions)
        progress_box=QGroupBox("任务进度");progress_layout=QVBoxLayout(progress_box);self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0)
        self.status=QLabel("等待任务");self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMinimumHeight(150)
        progress_layout.addWidget(self.progress);progress_layout.addWidget(self.status);progress_layout.addWidget(self.log);root.addWidget(progress_box)

        postprocess = QGroupBox("Windows 本地预标注目录后处理（可选）")
        post_layout = QGridLayout(postprocess)
        help_text = QLabel(
            "按视频名匹配预标注子目录，将视频复制进去，并把其中的 labels 目录改成视频同名目录。"
            "建议先预览；重复执行不会覆盖已有的不同文件。"
        )
        help_text.setWordWrap(True);help_text.setStyleSheet("color:#667085")
        self.post_video_dir = QLineEdit(r"D:\cosmos_vid")
        self.post_prelabel_dir = QLineEdit(r"D:\test")
        self.post_video_button = QPushButton("选择目录…");self.post_video_button.clicked.connect(self._choose_post_video_dir)
        self.post_prelabel_button = QPushButton("选择目录…");self.post_prelabel_button.clicked.connect(self._choose_post_prelabel_dir)
        self.post_preview_button = QPushButton("预览变更");self.post_preview_button.clicked.connect(lambda: self._run_postprocess(True))
        self.post_run_button = QPushButton("执行整理");self.post_run_button.clicked.connect(lambda: self._run_postprocess(False))
        self.post_log = QPlainTextEdit();self.post_log.setReadOnly(True);self.post_log.setMinimumHeight(150)
        post_layout.addWidget(help_text,0,0,1,4)
        post_layout.addWidget(QLabel("视频目录"),1,0);post_layout.addWidget(self.post_video_dir,1,1,1,2);post_layout.addWidget(self.post_video_button,1,3)
        post_layout.addWidget(QLabel("预标注目录"),2,0);post_layout.addWidget(self.post_prelabel_dir,2,1,1,2);post_layout.addWidget(self.post_prelabel_button,2,3)
        post_actions=QHBoxLayout();post_actions.addStretch();post_actions.addWidget(self.post_preview_button);post_actions.addWidget(self.post_run_button)
        post_layout.addLayout(post_actions,3,0,1,4);post_layout.addWidget(self.post_log,4,0,1,4)
        root.addWidget(postprocess)
        root.addStretch()
        scroll.setWidget(body);self.setCentralWidget(scroll)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.setStyleSheet("QGroupBox{font-weight:600;border:1px solid #d0d5dd;border-radius:8px;margin-top:10px;padding:14px 10px 10px}QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px}QLineEdit,QComboBox,QSpinBox,QPlainTextEdit{padding:6px;border:1px solid #b8c0cc;border-radius:5px}QPushButton{padding:7px 14px}QProgressBar{height:20px;text-align:center}")

    def _choose_video(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"选择视频",self.input_path.text(),"视频 (*.mp4 *.avi *.mkv *.mov *.webm)")
        if path:self.input_path.setText(path)
    def _choose_input_dir(self) -> None:
        path=QFileDialog.getExistingDirectory(self,"选择视频目录",self.input_path.text());
        if path:self.input_path.setText(path)
    def _choose_output_dir(self) -> None:
        path=QFileDialog.getExistingDirectory(self,"选择结果目录",self.output_path.text());
        if path:self.output_path.setText(path)
    def _choose_key(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"选择 SSH 私钥",self.sftp_key.text(),"所有文件 (*)");
        if path:self.sftp_key.setText(path)
    def _choose_post_video_dir(self) -> None:
        path=QFileDialog.getExistingDirectory(self,"选择本地视频目录",self.post_video_dir.text());
        if path:self.post_video_dir.setText(path)
    def _choose_post_prelabel_dir(self) -> None:
        path=QFileDialog.getExistingDirectory(self,"选择本地预标注目录",self.post_prelabel_dir.text());
        if path:self.post_prelabel_dir.setText(path)

    def _mode_changed(self) -> None:
        sftp=self.mode.currentData()=="sftp";self.sftp_frame.setVisible(sftp);self.reuse.setVisible(sftp)
        self.output_label.setText("本地标注 ZIP 目录" if sftp else "GPU 服务器输出目录")
        self.file_button.setVisible(sftp);self.input_dir_button.setVisible(sftp);self.output_button.setVisible(sftp)
        if sftp:
            self.input_path.setPlaceholderText("选择本机视频，或包含多个视频的本机目录")
            self.output_path.setPlaceholderText("选择下载 YOLO ZIP 的本机目录")
        else:
            self.input_path.setPlaceholderText("GPU 服务器上的 Linux 路径，例如 /data2/videos")
            self.output_path.setPlaceholderText("GPU 服务器上的 Linux 输出路径，例如 /data2/labels")

    def _connection(self) -> ConnectionReq:
        return ConnectionReq(server_url=self.server_url.text().strip(),mode=str(self.mode.currentData()),sftp_host=self.sftp_host.text().strip(),sftp_port=self.sftp_port.value(),sftp_username=self.sftp_user.text().strip(),sftp_password=self.sftp_password.text(),sftp_key_path=self.sftp_key.text().strip(),sftp_remote_dir=self.remote_dir.text().strip())

    def _mapping(self) -> tuple[list[str],dict[str,int]]:
        categories=[];mapping={}
        for raw in self.classes.toPlainText().splitlines():
            line=raw.strip()
            if not line:continue
            parts=line.split(maxsplit=1)
            if len(parts)!=2 or not parts[0].isdigit():raise ValueError(f"类别格式错误：{line}。应填写为“0 person”")
            mapping[parts[1].strip()]=int(parts[0]);categories.append(parts[1].strip())
        if not categories:raise ValueError("请至少填写一个类别映射")
        return categories,mapping

    def _batch_request(self) -> BatchReq:
        categories,mapping=self._mapping();connection=self._connection()
        if not self.input_path.text().strip():raise ValueError("请选择视频或视频目录")
        if not self.output_path.text().strip():raise ValueError("请选择或填写输出目录")
        return BatchReq(**connection.model_dump(),input_path=self.input_path.text().strip(),output_path=self.output_path.text().strip(),cuda_device=self.cuda.value(),dtype=self.dtype.currentText(),prompt=self.prompt.text().strip() or "person",categories=categories,class_map=mapping,task=self.task.currentText(),recursive=self.recursive.isChecked(),reuse_uploads=self.reuse.isChecked(),frame_step=self.frame_step.value(),max_frames=self.max_frames.value())

    def _launch(self, worker: TaskWorker, on_success: Callable[[dict],None]) -> None:
        if self.thread is not None and self.thread.isRunning():
            self._failed("已有任务正在运行，请等待其完成");return
        self.test_button.setEnabled(False);self.run_button.setEnabled(False);self.post_preview_button.setEnabled(False);self.post_run_button.setEnabled(False)
        self.thread=QThread(self);self.worker=worker;worker.moveToThread(self.thread);self.thread.started.connect(worker.run);worker.progress.connect(self._show_progress);worker.succeeded.connect(on_success);worker.failed.connect(self._failed);worker.finished.connect(self.thread.quit);worker.finished.connect(worker.deleteLater);self.thread.finished.connect(self._thread_done);self.thread.start()

    def _test_connection(self) -> None:
        try:req=self._connection()
        except Exception as exc:self._failed(str(exc));return
        self.test_button.setEnabled(False);self.test_result.setText("正在测试…")
        def action(_:TaskWorker)->dict:
            gpu=_json_request("GET",f"{req.server_url.rstrip('/')}/api/locateanything/health")
            result={"gpu":gpu}
            if req.mode=="sftp":
                client=_connect_sftp(req)
                try:
                    sftp=client.open_sftp()
                    try:sftp.stat(req.sftp_remote_dir);result["sftp"]={"ok":True}
                    finally:sftp.close()
                finally:client.close()
            else:
                _check_direct_capabilities(req.server_url,gpu);result["direct"]={"ok":True}
            return result
        self._launch(TaskWorker(action),self._connection_ok)

    def _start_batch(self) -> None:
        try:req=self._batch_request();self._save_settings()
        except Exception as exc:self._failed(str(exc));return
        self.run_button.setEnabled(False);self.log.clear();self.progress.setRange(0,0);self.status.setText("正在准备任务…")
        def action(worker:TaskWorker)->dict:
            job_id=uuid.uuid4().hex;JOBS[job_id]={"id":job_id,"status":"queued","message":"Queued","completed":0,"total":0,"items":[]}
            process=threading.Thread(target=_run_batch,args=(job_id,req),daemon=True);process.start()
            while process.is_alive():worker.progress.emit(copy.deepcopy(JOBS[job_id]));time.sleep(.5)
            process.join();worker.progress.emit(copy.deepcopy(JOBS[job_id]));job=JOBS[job_id]
            if job["status"]!="done":raise RuntimeError(job["message"])
            return copy.deepcopy(job)
        self._launch(TaskWorker(action),self._batch_ok)

    def _run_postprocess(self, dry_run: bool) -> None:
        video_dir=self.post_video_dir.text().strip();prelabel_dir=self.post_prelabel_dir.text().strip()
        if not video_dir or not prelabel_dir:
            self._failed("请填写视频目录和预标注目录");return
        if not dry_run:
            answer=QMessageBox.question(self,"确认执行","将复制视频并重命名/合并 labels 目录。是否继续？")
            if answer!=QMessageBox.StandardButton.Yes:return
        self._save_settings();self.post_preview_button.setEnabled(False);self.post_run_button.setEnabled(False)
        self.post_log.setPlainText("正在预览…" if dry_run else "正在整理…")
        self._launch(TaskWorker(lambda _: organize_prelabels(video_dir,prelabel_dir,dry_run=dry_run)),self._postprocess_ok)

    def _postprocess_ok(self,result:dict) -> None:
        counts=result["counts"]
        lines=[
            ("预览完成" if result["dry_run"] else "整理完成")+
            f"：视频 {result['video_count']}，匹配 {result['matched_count']}，"
            f"待处理/完成 {counts['ready']+counts['done']}，跳过 {counts['skipped']}，错误 {counts['error']}"
        ]
        for item in result["items"]:
            lines.append(f"[{item['status']}] {item['name']}")
            for action in item.get("actions",[]):lines.append(f"  - {action}")
            if item.get("error"):lines.append(f"  ! {item['error']}")
        self.post_log.setPlainText("\n".join(lines))
        if not result["dry_run"]:
            QMessageBox.information(self,"整理完成",lines[0])

    def _show_progress(self,job:dict) -> None:
        total=int(job.get("total",0));done=int(job.get("completed",0));self.progress.setRange(0,max(1,total));self.progress.setValue(done);self.status.setText(str(job.get("message","")))
        lines=[]
        for item in job.get("items",[]):lines.append(f"[{item.get('status','')}] {str(item.get('video','')).replace(chr(92),'/').rsplit('/',1)[-1]}\n  {item.get('message','')}{chr(10)+'  → '+item['output'] if item.get('output') else ''}")
        self.log.setPlainText("\n".join(lines))

    def _connection_ok(self,result:dict) -> None:
        gpu=result["gpu"];extra="，SFTP 正常" if "sftp" in result else "，直连接口与输出目录正常";self.test_result.setText(f"连接成功：{gpu.get('device')}/{gpu.get('dtype')}{extra}");self.test_result.setStyleSheet("color:#16803c")
    def _batch_ok(self,result:dict) -> None:
        self._show_progress(result);QMessageBox.information(self,"任务完成",f"已完成 {result.get('completed',0)} 个视频。")
    def _failed(self,message:str) -> None:
        self.status.setText(message);self.test_result.setText("失败："+message);self.test_result.setStyleSheet("color:#b42318");QMessageBox.critical(self,"操作失败",message)
    def _thread_done(self) -> None:
        self.test_button.setEnabled(True);self.run_button.setEnabled(True);self.post_preview_button.setEnabled(True);self.post_run_button.setEnabled(True);self.worker=None;self.thread=None

    def _save_settings(self) -> None:
        values={"server_url":self.server_url.text(),"mode":self.mode.currentData(),"cuda":self.cuda.value(),"sftp_host":self.sftp_host.text(),"sftp_port":self.sftp_port.value(),"sftp_user":self.sftp_user.text(),"sftp_key":self.sftp_key.text(),"remote_dir":self.remote_dir.text(),"input_path":self.input_path.text(),"output_path":self.output_path.text(),"prompt":self.prompt.text(),"classes":self.classes.toPlainText(),"task":self.task.currentText(),"dtype":self.dtype.currentText(),"frame_step":self.frame_step.value(),"max_frames":self.max_frames.value(),"recursive":self.recursive.isChecked(),"reuse":self.reuse.isChecked(),"post_video_dir":self.post_video_dir.text(),"post_prelabel_dir":self.post_prelabel_dir.text()}
        for key,value in values.items():self.settings.setValue(key,value)
    def _load_settings(self) -> None:
        text_fields={"server_url":self.server_url,"sftp_host":self.sftp_host,"sftp_user":self.sftp_user,"sftp_key":self.sftp_key,"remote_dir":self.remote_dir,"input_path":self.input_path,"output_path":self.output_path,"prompt":self.prompt,"post_video_dir":self.post_video_dir,"post_prelabel_dir":self.post_prelabel_dir}
        for key,widget in text_fields.items():
            value=self.settings.value(key)
            if value is not None:widget.setText(str(value))
        classes=self.settings.value("classes")
        if classes is not None:self.classes.setPlainText(str(classes))
        for key,widget in (("cuda",self.cuda),("sftp_port",self.sftp_port),("frame_step",self.frame_step),("max_frames",self.max_frames)):
            value=self.settings.value(key)
            if value is not None:widget.setValue(int(value))
        for key,widget in (("mode",self.mode),("task",self.task),("dtype",self.dtype)):
            value=self.settings.value(key)
            if value is not None:
                index=widget.findData(value) if key=="mode" else widget.findText(str(value))
                if index>=0:widget.setCurrentIndex(index)
        self.recursive.setChecked(self.settings.value("recursive",False,type=bool));self.reuse.setChecked(self.settings.value("reuse",True,type=bool))


def main() -> None:
    app=QApplication(sys.argv);app.setApplicationName("LocateAnything Batch Tool");window=MainWindow();window.show();sys.exit(app.exec())
