# 数据与目录约定

## 平台任务目录

```text
<tasks-dir>/
  platform.sqlite3       # 任务、类别、阶段、视频、分段和事件元数据
  platform.sqlite3-wal   # SQLite 运行期文件
  platform.sqlite3-shm   # SQLite 运行期文件
  <task-id>/
    attachments/        # 任务或 Part 资料附件
    videos/
      <video-id>/
        raw/
        input_labels/
        locany_labels/
        segments/
          seg_0000/
            raw/
            input_labels/
            locany_labels/
            tracking/
            package/
            reviewed/
            exports/
    tracking/    # 临时保留的整段视频兼容目录
    package/     # 临时保留的整段视频兼容目录
    reviewed/    # 临时保留的整段视频兼容目录
    exports/     # 临时保留的整段视频兼容目录
```

各阶段产物必须分别保存，不应覆盖用户上传的源标注或已经审核的结果。

数据库只保存元数据、角色、Part、工时、附件索引、问题状态、文件路径和处理状态，不保存附件内容、视频二进制或逐帧目标框。旧任务目录中的 `task.json` 与 `events.jsonl` 会被自动导入并原样保留；导出的标注包仍包含一份由数据库即时生成的 `task.json` 快照，供离线查看。

## YOLO 输入与输出

每个 TXT 文件对应一帧视频。支持以下两种行格式：

```text
class_id x_center y_center width height
class_id x_center y_center width height score
```

坐标和宽高均为归一化值。LocateAnything 输出包含 `score`。类别 ID 必须是非负整数，并与任务类别表一致。

## 跟踪结果

`tracking_results.json` 是 MOT 流水线、Annotator 和 SAM3.1 之间的交换格式。它包含视频元数据，以及带有稳定 `track_id`、`class_id` 和逐帧像素坐标框的轨迹。该文件是可编辑轨迹的权威数据源；人工审核后再从它派生 YOLO 文件。

## ZIP 跨平台约定

ZIP 内部文件路径必须使用相对路径和正斜杠。客户端不得依赖压缩包中的 Windows 盘符或 Linux 绝对路径。解压代码会拒绝任何试图逃逸目标目录的路径。

## 帧编号

源标注通常从第 1 帧开始编号。分段标注会从配置的帧偏移量重新编号。接入新的数据来源时，必须检查分段后的第一帧和最后一帧，尤其要注意上游模型是否使用从 0 开始的文件名。
