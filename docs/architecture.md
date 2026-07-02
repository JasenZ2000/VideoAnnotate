# Architecture

## Deployment Topology

The system is intentionally split by latency and hardware responsibility:

- The **Windows public machine** owns task state and shared files. It exposes the workflow platform to the team and runs CPU-side segmentation, tracking, packaging and export work.
- The **Linux GPU server** hosts two independent services. LocateAnything produces prompt-based YOLO prelabels; SAM3.1 extends a manually supplied box through a video segment.
- Each **employee Windows PC** runs Annotator locally. Video frames and interaction stay local, while only explicit GPU jobs are sent to SAM3.1.

The services are independent Python processes and should use separate environments. LocateAnything and SAM3.1 have different CUDA and dependency constraints.

## Component Boundaries

### Workflow Platform

`workflow_platform/server.py` is the task orchestrator and filesystem API. It owns:

- task creation, assignment, notes and soft deletion;
- video and YOLO ZIP ingestion;
- video/label segmentation;
- asynchronous LocateAnything requests and result download;
- MOT pipeline execution;
- package creation, reviewed-result upload and YOLO export.

Task metadata is stored in `task.json`; the audit stream is `events.jsonl`. There is no database in the current MVP.

### Annotator

`annotator/server.py` and `annotator/static/` form a local web application. It owns interactive track editing, interpolation, quality checks, local save/export and remote SAM3.1 calls. It must not be treated as the central task database.

### MOT Pipeline

`mot_pipeline/` is shared domain logic. It reads frame-level YOLO detections, runs a selected tracker in both directions, fuses tracklets, smooths trajectories and writes `tracking_results.json`, YOLO and visualization outputs.

### GPU Services

Both GPU APIs use asynchronous jobs:

1. Client submits `POST /api/jobs`.
2. Service returns a `job_id` immediately.
3. Client polls `GET /api/jobs/{job_id}`.
4. Client downloads the result after status becomes `done`.

This avoids HTTP read timeouts during long inference. Job state is currently held in process memory; result artifacts are written under each service cache directory.

## Video Transfer Modes

Two transfer modes are supported by clients:

- `path`: Windows and Linux see the same storage through different path prefixes. This is preferred for large videos.
- `sftp`: the client uploads the video to an allowed directory on the GPU server. Passwords are read from named environment variables, not stored in task metadata.

The GPU services validate `video_path` against `*_ALLOWED_ROOTS`. Always configure this in shared deployments.

## Trust Boundary

The current APIs do not implement user authentication, authorization or TLS. Deploy them only on a trusted internal network, restrict ports with host/network firewalls, use dedicated SFTP accounts, and limit allowed filesystem roots. An authenticated reverse proxy is required before exposure outside the trusted LAN.
