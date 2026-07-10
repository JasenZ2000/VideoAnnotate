# 系统架构

## 部署拓扑

系统按照交互延迟和硬件职责拆分为三类节点：

- **Windows 公共机**负责保存任务状态和共享文件，向团队提供作业流程平台，并执行视频分段、跟踪、打包和导出等 CPU 任务。
- **Linux GPU 服务器**运行一个统一 HTTP 服务。LocateAnything 根据文本提示生成 YOLO 预标注；SAM3.1 根据人工给定的目标框，在视频片段中延伸目标轨迹。
- 每台**员工 Windows 电脑**都在本地运行一个工作台服务，其中挂载 Annotator 和帧采样器。视频帧浏览、标注编辑与采样计划在本机完成，只有明确触发的 GPU 任务才会发送到统一 GPU 服务。

统一服务按 `/api/locateanything` 与 `/api/sam31` 分隔接口。LocateAnything 在服务进程环境中运行；SAM3.1 通过 `SAM31_PYTHON` 调起原有 ComfyUI 环境的子进程，因此两套 CUDA 依赖不需要安装在同一个环境中。

## 组件职责边界

### 作业流程平台

`workflow_platform/server.py` 是任务编排器和文件系统接口，负责：

- 创建任务、分配人员、记录备注和软删除；
- 管理发布人、负责人、标注员、Part 队列、工时、附件和问题单；
- 接收视频和 YOLO ZIP；
- 对视频及其标注进行分段；
- 异步提交 LocateAnything 任务并下载结果；
- 执行 MOT 跟踪流水线；
- 生成标注包、接收审核结果并导出 YOLO。

任务、人员、Part、工时、附件索引、问题单、类别、阶段、视频、分段和审计事件保存在 SQLite 中；附件、视频与标注等大文件仍保存在任务文件目录。旧版 `task.json` 和 `events.jsonl` 会在首次启动时自动导入数据库。

详细功能边界和处理逻辑图见[平台现有功能与处理逻辑](platform-capabilities.md)。

### 本地标注器

`local_workbench/server.py` 将 `utils/annotator/` 和 `utils/frame_sampler/` 挂载到一个本地 Web 服务。Annotator 负责交互式轨迹编辑、插值、质量检查、本地保存与导出，以及远端 SAM3.1 调用；帧采样器负责按稠密/稀疏计划导出训练帧。工作台不应被当作中央任务数据库使用。

### MOT 流水线

`utils/mot_pipeline/` 存放本地工作台和平台共用的领域逻辑。它读取逐帧 YOLO 检测框，分别执行正向和反向跟踪，融合轨迹片段，平滑轨迹，并输出 `tracking_results.json`、YOLO 标注和可视化结果。

### GPU 服务

统一 GPU API 的两类任务都采用异步模式：

1. 客户端通过 `POST /api/sam31/jobs` 或 `POST /api/locateanything/jobs` 提交任务。
2. 服务立即返回 `job_id`。
3. 客户端在对应命名空间轮询 `GET /api/{service}/jobs/{job_id}`。
4. 状态变为 `done` 后，客户端下载结果。

这种方式可以避免长时间推理导致 HTTP 读取超时。当前任务状态保存在服务进程内存中，结果文件分别写入 SAM3.1 和 LocateAnything 的缓存目录。

## 视频传输方式

客户端支持两种传输方式：

- `path`：Windows 和 Linux 通过不同路径前缀访问同一份共享存储。大视频优先使用这种方式。
- `sftp`：客户端先把视频上传到 GPU 服务器允许访问的目录。密码从指定环境变量读取，不写入任务元数据。

GPU 服务会根据 `*_ALLOWED_ROOTS` 校验 `video_path`。多人共享部署时必须配置允许访问的根目录。

## 安全边界

当前 API 尚未实现用户认证、权限控制和 TLS。服务只能部署在可信内部网络中，应使用主机或网络防火墙限制端口，使用专用 SFTP 账号，并严格限制服务可访问的文件系统根目录。若要暴露到可信局域网之外，必须在前方增加带认证和 HTTPS 的反向代理。
