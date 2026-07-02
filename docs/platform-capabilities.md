# 平台现有功能与处理逻辑

本文档描述当前代码已经实现的能力，不把路线图中的计划功能算作现成功能。

## 功能总表

| 功能域 | 当前支持情况 | 说明与边界 |
| --- | --- | --- |
| 任务登记 | 已支持 | 创建任务、填写负责人、备注、提示词和类别表 |
| 任务查询 | 已支持 | 列表、详情、阶段状态、最近 100 条事件 |
| 任务修改 | 已支持 | 修改名称、负责人、备注、提示词和类别映射 |
| 任务删除 | 已支持 | 软删除，只隐藏记录，不删除磁盘文件 |
| 多视频任务 | 基础支持 | 一个任务可上传多个视频；界面主要操作当前视频，尚无完整的视频切换工作流 |
| 视频上传 | 已支持 | 支持 MP4、AVI、MKV、MOV、WEBM |
| YOLO 预标注上传 | 已支持 | 上传 ZIP，安全解压并归档当前视频的 TXT 标注 |
| 长视频分段 | 已支持 | 按固定帧数拆分视频，并同步重编号已有 YOLO 标注 |
| LocateAnything 整段推理 | 已支持 | 异步提交远端任务，轮询并下载 YOLO ZIP |
| LocateAnything 分段推理 | 已支持 | 按当前视频的全部分段顺序推理，记录每段状态 |
| 多类别 LocateAnything | 已支持 | 根据任务类别表发送 `categories` 和 `class_map` |
| 整段轨迹生成 | 已支持 | 读取上传或 LocateAnything 标注，执行双向跟踪与融合 |
| 分段轨迹生成 | 已支持 | 对有标注的分段逐段执行跟踪，对无标注分段标记为跳过 |
| 短漏检补全 | 已支持 | 跟踪后按配置帧数插值轨迹内部的短缺口 |
| 标注包生成 | 仅整段视频 | 打包视频、`tracking_results.json`、预览和任务快照 |
| 人工审核结果上传 | 仅整段视频 | 接收审核后的 `tracking_results.json` |
| YOLO 最终导出 | 仅整段视频 | 优先使用审核结果，否则使用整段跟踪结果 |
| SAM3.1 辅助跟踪 | Annotator 支持 | 由员工本地 Annotator 调用，不在公共平台任务页内执行 |
| 元数据持久化 | 已支持 | SQLite 保存任务、类别、阶段、视频、分段和事件 |
| 旧数据迁移 | 已支持 | 首次启动自动导入旧 `task.json` 和 `events.jsonl`，原文件保留 |
| GPU 任务队列 | 基础支持 | 平台进程内串行锁；重启恢复、取消、优先级和持久队列尚未实现 |
| 用户与权限 | 未支持 | 当前无登录、角色和任务级权限，只适合可信内网 |

## 端到端作业逻辑

```mermaid
flowchart TD
    A["创建任务<br/>负责人、备注、类别表"] --> B["上传一个或多个完整视频"]
    B --> C{"是否已有模型预标注？"}
    C -->|"有"| D["上传整段 YOLO ZIP"]
    C -->|"没有"| E{"是否使用 LocateAnything？"}
    C -->|"不需要"| F["无预标注，后续人工处理"]

    D --> G{"视频是否需要分段？"}
    E -->|"是"| G
    E -->|"否"| F
    G -->|"是"| H["按帧数拆分视频和已有标注"]
    G -->|"否"| I["处理整段视频"]

    H --> J{"预标注来源"}
    J -->|"上传的 YOLO"| K["使用分段重编号标注"]
    J -->|"LocateAnything"| L["逐分段远端推理"]
    J -->|"无"| M["分段无初始标注"]
    K --> N["逐分段跟踪与轨迹融合"]
    L --> N
    M --> O["本地人工补充标注"]

    I --> P{"整段预标注来源"}
    P -->|"上传的 YOLO"| Q["整段跟踪与轨迹融合"]
    P -->|"LocateAnything"| R["整段远端推理"]
    P -->|"无"| S["本地人工标注"]
    R --> Q
    Q --> T["插值少量短漏检"]
    T --> U["生成并下载整段标注包"]
    U --> V["员工本地 Annotator 清理"]
    V --> W{"是否需要 SAM3.1？"}
    W -->|"是"| X["远端目标框提示跟踪并合并"]
    W -->|"否"| Y["完成人工审核"]
    X --> Y
    Y --> Z["上传审核结果并导出 YOLO"]

    N -.->|"分段打包、回传和最终合并待完善"| U
```

## 系统调用逻辑

```mermaid
flowchart LR
    Browser["部门用户浏览器"] -->|"HTTP"| Platform["Windows 作业流程平台"]
    Platform --> DB[("SQLite<br/>任务元数据与事件")]
    Platform --> Files["任务文件目录<br/>视频、标注、轨迹、压缩包"]
    Platform -->|"提交、轮询、下载"| Locate["Linux LocateAnything 服务"]
    Platform -->|"SFTP 或共享路径"| GPUFiles["GPU 服务器视频目录"]

    Browser -->|"下载标注包"| Local["员工本地 Annotator"]
    Local -->|"目标框任务"| SAM["Linux SAM3.1 服务"]
    Local -->|"SFTP 或共享路径"| GPUFiles
    Local -->|"上传审核 JSON"| Platform

    Locate --> GPUFiles
    SAM --> GPUFiles
```

## SQLite 数据关系

视频、模型权重和生成文件不写入数据库，只在数据库中保存路径和状态。

```mermaid
erDiagram
    TASKS ||--o{ TASK_CLASSES : "定义类别"
    TASKS ||--o{ TASK_STAGES : "记录阶段"
    TASKS ||--o{ VIDEOS : "包含视频"
    TASKS ||--o{ EVENTS : "产生事件"
    VIDEOS ||--o{ SEGMENTS : "拆分为"

    TASKS {
        text task_id PK
        text name
        text assignee
        text status
        boolean deleted
        text current_video_id
        datetime created_at
        datetime updated_at
    }
    TASK_CLASSES {
        text task_id FK
        int class_id
        text name
    }
    TASK_STAGES {
        text task_id FK
        text stage
        text status
        text message
    }
    VIDEOS {
        text task_id FK
        text video_id
        text path
        int frame_count
        real fps
        text split_status
    }
    SEGMENTS {
        text task_id FK
        text video_id FK
        text segment_id
        int start_frame
        int end_frame
        text locate_status
        text tracking_status
    }
    EVENTS {
        int event_id PK
        text task_id FK
        datetime time
        text level
        text message
    }
```

## 状态记录逻辑

任务顶层 `status` 用于列表快速显示，`task_stages` 则分别记录视频、预标注、LocateAnything、跟踪、打包、审核和导出阶段。长任务在执行过程中会持续更新对应阶段：

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> queued: 用户提交操作
    queued --> running: 后台线程取得执行权
    running --> done: 结果落盘并登记
    running --> failed: 捕获异常并记录错误
    failed --> queued: 用户重新提交
```

SQLite 使用 WAL 模式和每次操作独立连接，平台内写入再通过进程内可重入锁串行化。该设计支持当前后台线程并发读写，但尚不能替代跨进程任务队列。
