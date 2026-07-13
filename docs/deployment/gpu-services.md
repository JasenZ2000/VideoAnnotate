# Linux 统一 GPU 服务部署

## 运行模型

只启动一个 HTTP 进程：`gpu_services.server`，默认监听 `9010`。它通过两个命名空间暴露能力：

- `/api/sam31/...`：目标框提示的视频跟踪；
- `/api/locateanything/...`：视频逐帧预标注和 YOLO ZIP 导出。

HTTP 服务不要求把两套模型运行时合并到一个 Python 环境。它运行在能加载 LocateAnything 的环境中；SAM3.1 作业由 `SAM31_PYTHON` 指向原有的 ComfyUI 环境后，以子进程方式运行。

## 先决条件

- 服务进程环境：Python 3.10+、CUDA 匹配的 PyTorch、LocateAnything 的 Python 依赖；
- 外部 LocateAnything 目录，其中必须存在 `locateanything_worker.py`；
- 已可用的 ComfyUI 与 SAM3.1 checkpoint；
- GPU 服务只能绑定在可信内部网络，且两个模型的可读视频目录必须各自受到限制。

仓库不再包含或安装 LocateAnything 上游源码。将其部署在服务器独立目录后，通过 `LOCATEANYTHING_ROOT` 引用即可。

## 配置与启动

从 [环境变量模板](../../configs/gpu-services.env.example) 复制本机配置，不要提交真实路径、账号或模型权重路径。

```bash
cd /srv/video-annotation-workflow
source /path/to/locateanything-env/bin/activate
pip install -r requirements/gpu-services.txt

export GPU_SERVICE_HOST=0.0.0.0
export GPU_SERVICE_PORT=9010

export LOCATEANYTHING_ROOT=/opt/LocateAnything
export LOCANY_MODEL=nvidia/LocateAnything-3B
export LOCANY_CACHE_DIR=/data/annotation-cache/locateanything
export LOCANY_ALLOWED_ROOTS=/data/annotation-transfer/locateanything/videos
export LOCANY_OUTPUT_ALLOWED_ROOTS=/data/annotation-output/locateanything
export LOCANY_DEVICE=cuda:1
export LOCANY_DTYPE=bf16

export SAM31_COMFY_ROOT=/opt/ComfyUI
export SAM31_CHECKPOINT=/models/sam3.1_multiplex_fp16.safetensors
export SAM31_PYTHON=/opt/venvs/sam31-comfy/bin/python
export SAM31_RUNNER=/srv/video-annotation-workflow/gpu_services/sam31_track.py
export SAM31_CACHE_DIR=/data/annotation-cache/sam31
export SAM31_ALLOWED_ROOTS=/data/annotation-transfer/sam31/videos
export SAM31_DEVICE=cuda:0
export SAM31_DTYPE=fp16

./gpu_services/run_gpu_service.sh
```

`LOCANY_CACHE_DIR` 和 `SAM31_CACHE_DIR` 是服务自己的结果缓存目录。标注器和平台的 SFTP 上传目录不是缓存目录，且必须分别落在对应的 `*_ALLOWED_ROOTS` 内。

每次请求可以选择 `cuda:N`。LocateAnything 会按 `device + dtype` 缓存模型实例，所以启用多张卡会在每张卡上各加载一份模型；其推理任务仍在单个服务进程内串行执行。

## 验证

```bash
curl http://127.0.0.1:9010/api/health
curl http://127.0.0.1:9010/api/sam31/health
curl http://127.0.0.1:9010/api/locateanything/health
```

根健康检查会同时报告两个运行时的配置。LocateAnything 健康结果中的 `worker_available` 应为 `true`；它为 `false` 时，优先检查 `LOCATEANYTHING_ROOT` 是否指向包含 `locateanything_worker.py` 的目录。

## 进程守护与更新

正式部署建议使用 systemd、Supervisor 或其他进程管理器，工作目录设为仓库根目录并以受限账号运行。缓存清理由独立任务执行，不要在服务进程内部删除运行中的任务目录。

升级时停止这一项统一服务，更新仓库和外部 LocateAnything 运行时，检查 `/api/health`，再分别提交一段已知视频的 LocateAnything 与 SAM3.1 作业。任何 `gpu_services/` 适配层变更都需要重启统一服务。
