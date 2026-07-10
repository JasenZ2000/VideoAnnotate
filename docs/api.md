# HTTP API 概览

每个 FastAPI 服务都在 `/docs` 路径提供可交互的 OpenAPI 文档。

## 作业流程平台

- `GET /api/health`
- `GET|POST /api/tasks`
- `GET|PATCH|DELETE /api/tasks/{task_id}`
- `POST /api/tasks/{task_id}/video`
- `POST /api/tasks/{task_id}/labels-zip`
- `POST /api/tasks/{task_id}/split-video`
- `POST /api/tasks/{task_id}/run-locateanything`
- `POST /api/tasks/{task_id}/run-segment-locateanything`
- `POST /api/tasks/{task_id}/run-tracking`
- `POST /api/tasks/{task_id}/run-segment-tracking`
- `POST /api/tasks/{task_id}/parts`
- `POST /api/tasks/{task_id}/parts/claim-next`
- `POST /api/tasks/{task_id}/parts/{part_id}/start|submit|release|review`
- `POST /api/tasks/{task_id}/attachments`
- `GET /api/tasks/{task_id}/attachments/{attachment_id}/download`
- `POST /api/tasks/{task_id}/issues`
- `POST /api/tasks/{task_id}/issues/{issue_id}/resolve`
- 同一任务路径下还提供标注包生成、审核结果上传和 YOLO 导出接口

健康检查结果包含 SQLite 路径、数据结构版本、`quick_check` 结果、任务和事件数量，以及旧 JSON 迁移的成功与失败计数。

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

两类任务服务会返回 `queued`、`running`、`done` 或 `failed` 状态。客户端应按合理的固定间隔轮询，仅在状态变为 `done` 后获取结果文件。
