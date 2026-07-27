# HTTP API 概览

每个 FastAPI 服务都在 `/docs` 路径提供可交互的 OpenAPI 文档。

## 多人标注协作平台

- `GET /api/health`
- `GET /api/auth/me`
- `POST /api/auth/bootstrap-admin|login|logout|change-password`
- `GET|POST /api/users`、`PATCH /api/users/{username}`
- `GET /api/user-options`：获取可指定为协同查看人的启用用户
- `POST /api/tasks/preview`：解析粘贴的表格行
- `GET|POST /api/tasks`
- `GET /api/tasks/{task_id}`
- `POST /api/tasks/{task_id}/parts`：发布者追加 Part
- `PATCH /api/tasks/{task_id}`：发布者编辑任务信息
- `PATCH /api/tasks/{task_id}/ordering`：管理员修改任务排名和优先级
- `DELETE /api/tasks/{task_id}`：发布者删除任意状态的自有任务
- `POST /api/tasks/{task_id}/parts/claim-next`：领取下一个 Part，发布者也可领取自己的任务
- `POST /api/tasks/{task_id}/parts/{part_id}/start-rework`
- `POST /api/tasks/{task_id}/parts/{part_id}/pause`
- `POST /api/tasks/{task_id}/parts/{part_id}/resume`
- `POST /api/tasks/{task_id}/parts/{part_id}/return`
- `DELETE /api/tasks/{task_id}/parts/{part_id}`：发布者删除任意状态的 Part
- `POST /api/tasks/{task_id}/parts/{part_id}/submit`
- `POST /api/tasks/{task_id}/parts/{part_id}/comments`
- `POST /api/tasks/{task_id}/parts/{part_id}/review`
- `POST /api/tasks/{task_id}/parts/{part_id}/time-review`：发布者确认耗时偏差是实际异常还是预估不合理

平台 API 不再包含视频上传、LocateAnything、跟踪、分段或标注导出功能。

## 统一 GPU 服务

- `GET /api/health`
- `GET /api/sam31/health`
- `POST /api/sam31/jobs`
- `GET /api/sam31/jobs/{job_id}`
- `GET /api/sam31/jobs/{job_id}/tracking-results`
- `GET /api/locateanything/health`
- `POST /api/locateanything/jobs`
- `GET /api/locateanything/jobs/{job_id}`
- `GET /api/locateanything/jobs/{job_id}/yolo-zip`
- `POST /api/locateanything/image-jobs`：提交服务器图片目录批量标注任务
- `GET /api/locateanything/image-jobs/{job_id}`：查询图片任务状态
- `GET /api/locateanything/image-jobs/{job_id}/annotations-zip`：下载图片、YOLO TXT 与 VOC XML 结果

两类任务服务会返回 `queued`、`running`、`done` 或 `failed` 状态。开始运行后，任务信息中的 `assigned_device` 表示实际分配的设备。客户端应按合理的固定间隔轮询，仅在状态变为 `done` 后获取结果文件。

提交任务时省略 `device` 或设置为 `auto` 可使用服务端多 GPU 自动调度；设置为 `cuda:N` 则固定到指定设备排队。
