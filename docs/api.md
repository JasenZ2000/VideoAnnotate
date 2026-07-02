# HTTP API Summary

Interactive OpenAPI documentation is available at `/docs` on every FastAPI service.

## Workflow Platform

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
- package, reviewed upload and YOLO export endpoints under the same task path

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

The job services return `queued`, `running`, `done` or `failed`. Clients should poll with a bounded interval and fetch artifacts only after `done`.
