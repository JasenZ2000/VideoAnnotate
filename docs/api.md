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
- 同一任务路径下还提供标注包生成、审核结果上传和 YOLO 导出接口

## LocateAnything

- `GET /api/health`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/yolo-zip`

## SAM3.1

- `GET /api/health`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/tracking-results`

任务服务会返回 `queued`、`running`、`done` 或 `failed` 状态。客户端应按合理的固定间隔轮询，仅在状态变为 `done` 后获取结果文件。
