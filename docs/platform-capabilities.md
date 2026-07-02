# 平台现有功能与处理逻辑

平台以通用标注作业为主，视频目标检测是带专用预处理步骤的一种任务类型。本文只描述当前代码已经实现的能力。

## 部门作业主流程

```mermaid
flowchart TD
    A["发布人创建任务"] --> B["填写说明并上传文档、脚本或软件"]
    B --> C["指定负责人和标注员"]
    C --> D["发布人或负责人规划 Part"]
    D --> E["待领取 Part 池"]
    E --> F["标注员领取下一个 Part<br/>自动开始计时"]
    F --> G["执行标注并补充资料"]
    G --> H{"是否遇到问题？"}
    H -->|"是"| I["提交任务或 Part 问题单"]
    I --> J["负责人排查并关闭问题"]
    J --> G
    H -->|"否"| K["提交 Part<br/>停止本次计时"]
    K --> L{"负责人验收"}
    L -->|"通过"| M["Part 完成"]
    L -->|"返工"| N["退回原标注员并记录要求"]
    N --> O["开始返工<br/>新增计时会话"]
    O --> K
    M --> P{"还有待领取 Part？"}
    P -->|"有"| E
    P -->|"无"| Q["任务收尾"]
```

## 角色职责

| 角色 | 当前支持的操作 |
| --- | --- |
| 发布人 | 创建任务、填写说明、上传资料、指定负责人和标注员、增加 Part、查看进度、验收 Part、解决问题 |
| 负责人 | 维护任务与人员、增加 Part、查看每个 Part 累计耗时、验收或退回、排查问题；视频检测任务中负责预处理 |
| 标注员 | 从公共 Part 池领取下一份工作、自动计时、提交或释放 Part、处理返工、上传资料、报告问题 |

浏览器保存“当前操作人”，后端据此进行业务角色校验。当前尚无登录和身份认证，因此只适用于可信内部网络。

## 功能总表

| 功能域 | 当前支持情况 | 说明与边界 |
| --- | --- | --- |
| 任务登记 | 已支持 | 通用任务与视频目标检测任务 |
| 任务说明 | 已支持 | 发布人提供长文本要求，并可持续维护 |
| 人员角色 | 已支持 | 发布人、负责人和多个标注员 |
| 资料附件 | 已支持 | 上传文档、脚本、软件或其他文件；平台只保存和下载，不执行附件 |
| Part 规划 | 已支持 | 创建任务时或执行中批量增加 Part |
| 动态领取 | 已支持 | 标注员原子领取下一个待处理 Part，避免重复领取 |
| Part 工时 | 已支持 | 领取或返工开始时计时，提交或释放时停止，多次会话累计 |
| Part 验收 | 已支持 | 发布人或负责人通过，或退回原标注员返工 |
| 问题排查 | 已支持 | 任务级或 Part 级问题单，支持严重程度和负责人关闭 |
| 任务查询与修改 | 已支持 | 列表、详情、阶段、事件、角色、说明和人员维护 |
| 任务删除 | 已支持 | 软删除，只隐藏记录，不删除磁盘文件 |
| 多视频任务 | 基础支持 | 一个任务可上传多个视频；完整视频切换工作流尚待完善 |
| YOLO 预标注上传 | 已支持 | 安全解压 ZIP 并归档当前视频的 TXT 标注 |
| 长视频分段 | 已支持 | 按帧数拆分视频并同步重编号已有 YOLO 标注 |
| LocateAnything | 已支持 | 整段或逐分段远端推理，支持多类别映射 |
| 跟踪与轨迹融合 | 已支持 | 整段或逐分段执行，并插值轨迹内部短缺口 |
| 标注包、审核、YOLO 导出 | 仅整段视频 | 分段级完整闭环仍待完善 |
| SAM3.1 辅助跟踪 | Annotator 支持 | 由员工本地 Annotator 调用 |
| SQLite 持久化 | 已支持 | 保存角色、Part、工时、附件索引、问题、视频和事件 |
| 旧数据迁移 | 已支持 | 自动导入旧 `task.json` 和 `events.jsonl` |
| GPU 任务队列 | 基础支持 | 进程内串行锁；取消、优先级和持久队列尚未实现 |
| 用户认证 | 未支持 | 当前操作人不是安全登录身份 |

## 视频检测预处理责任链

该流程只在任务类型为“视频目标检测”时显示，并由发布人或负责人操作。

```mermaid
flowchart LR
    P["发布人上传视频"] --> Y{"是否提供 YOLO 预标注？"}
    Y -->|"已提供"| T["负责人执行跟踪与轨迹融合"]
    Y -->|"未提供"| L["负责人执行 LocateAnything"]
    L --> T
    T --> F["负责人设置短缺漏补全帧数"]
    F --> R["生成初始轨迹结果"]
    R --> S["负责人规划 Part 并开放领取"]
```

## 视频目标检测完整流程

```mermaid
flowchart TD
    A["创建视频检测任务<br/>说明、人员、类别表"] --> B["上传完整视频"]
    B --> C{"发布人是否提供 YOLO？"}
    C -->|"是"| D["上传 YOLO ZIP"]
    C -->|"否"| E["负责人运行 LocateAnything"]
    D --> F{"是否拆分长视频？"}
    E --> F
    F -->|"是"| G["拆分视频和标注"]
    F -->|"否"| H["处理整段视频"]
    G --> I["分段跟踪、轨迹融合、缺漏补全"]
    H --> J["整段跟踪、轨迹融合、缺漏补全"]
    I -.->|"分段打包和最终合并待完善"| K["本地 Annotator 清理"]
    J --> L["生成并下载标注包"]
    L --> K
    K --> M{"是否需要 SAM3.1？"}
    M -->|"是"| N["远端目标框提示跟踪"]
    M -->|"否"| O["完成审核"]
    N --> O
    O --> P["上传审核结果并导出 YOLO"]
```

## 系统调用逻辑

```mermaid
flowchart LR
    Browser["部门用户浏览器"] --> Platform["Windows 作业流程平台"]
    Platform --> DB[("SQLite<br/>任务、Part、工时、问题")]
    Platform --> Files["任务文件目录<br/>附件、视频、标注、结果"]
    Platform --> Locate["Linux LocateAnything 服务"]
    Browser --> Local["员工本地 Annotator"]
    Local --> SAM["Linux SAM3.1 服务"]
    Local --> Platform
    Locate --> GPUFiles["GPU 视频目录"]
    SAM --> GPUFiles
```

## SQLite 数据关系

```mermaid
erDiagram
    TASKS ||--o{ TASK_ANNOTATORS : "指定"
    TASKS ||--o{ PARTS : "拆分"
    PARTS ||--o{ PART_WORK_SESSIONS : "累计工时"
    TASKS ||--o{ ATTACHMENTS : "包含"
    PARTS ||--o{ ATTACHMENTS : "关联"
    TASKS ||--o{ ISSUES : "产生"
    PARTS ||--o{ ISSUES : "关联"
    TASKS ||--o{ TASK_CLASSES : "定义类别"
    TASKS ||--o{ TASK_STAGES : "记录阶段"
    TASKS ||--o{ VIDEOS : "包含视频"
    VIDEOS ||--o{ SEGMENTS : "拆分为"
    TASKS ||--o{ EVENTS : "产生事件"

    TASKS { text task_id PK text task_type text publisher text manager text status }
    TASK_ANNOTATORS { text task_id FK text username }
    PARTS { int part_id PK text task_id FK int part_index text status text annotator real work_seconds }
    PART_WORK_SESSIONS { int session_id PK int part_id FK datetime started_at datetime ended_at }
    ATTACHMENTS { text attachment_id PK text task_id FK int part_id FK text stored_path text sha256 }
    ISSUES { int issue_id PK text task_id FK int part_id FK text severity text status }
    TASK_CLASSES { text task_id FK int class_id text name }
    TASK_STAGES { text task_id FK text stage text status }
    VIDEOS { text task_id FK text video_id text path int frame_count }
    SEGMENTS { text video_id FK text segment_id int start_frame int end_frame }
    EVENTS { int event_id PK text task_id FK datetime time text message }
```

## Part 状态与计时

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> in_progress: 标注员领取并开始计时
    in_progress --> pending: 标注员释放并停止计时
    in_progress --> submitted: 标注员提交并停止计时
    submitted --> completed: 发布人或负责人验收通过
    submitted --> rework: 发布人或负责人退回
    rework --> in_progress: 原标注员开始返工并新增计时
```
