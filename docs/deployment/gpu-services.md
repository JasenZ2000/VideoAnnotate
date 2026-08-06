# Linux 统一 GPU 服务部署

## 运行模型

只启动一个 HTTP 进程：`gpu_services.server`，默认监听 `9010`。它通过两个命名空间暴露能力：

- `/api/sam31/...`：目标框提示的视频跟踪；
- `/api/locateanything/jobs...`：视频逐帧预标注；
- `/api/locateanything/image-jobs...`：服务器图片目录批量预标注。
- `/api/locateanything/image-directories`：发现远端根目录中直接包含图片的目录，供单卡顺序批处理客户端规划任务。

LocateAnything 结果 ZIP 中，YOLO 标注位于 `labels/`，Pascal VOC 标注位于 `annotations/`。两个目录按相同的视频名前缀和帧号逐帧对应。

图片目录任务支持 JPG、JPEG、PNG、BMP、WebP 和 TIFF。结果保持输入目录的相对子目录结构，标注分别写入 `labels/` 与 `annotations/`；默认不重复复制原图，设置 `copy_images=true` 时才会复制到 `images/`。图片输入目录必须位于 `LOCANY_ALLOWED_ROOTS` 内，直接输出目录必须位于 `LOCANY_OUTPUT_ALLOWED_ROOTS` 内。服务会拒绝任何可能覆盖或嵌套进输入目录的输出布局。

图片任务支持 `overwrite=never|replace`。Agent 批处理默认使用 `never`，并通过 `/api/locateanything/image-jobs/{job_id}/validate` 校验每个结果目录的标签、XML、元数据、原始回答与 ZIP。

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
export LOCANY_DEVICES=cuda:0,cuda:1
export LOCANY_DTYPE=bf16
export LOCANY_KEEP_MODEL_LOADED=0

export SAM31_COMFY_ROOT=/opt/ComfyUI
export SAM31_CHECKPOINT=/models/sam3.1_multiplex_fp16.safetensors
export SAM31_PYTHON=/opt/venvs/sam31-comfy/bin/python
export SAM31_RUNNER=/srv/video-annotation-workflow/gpu_services/sam31_track.py
export SAM31_CACHE_DIR=/data/annotation-cache/sam31
export SAM31_ALLOWED_ROOTS=/data/annotation-transfer/sam31/videos
export SAM31_DEVICES=cuda:0,cuda:1
export SAM31_DTYPE=fp16

./gpu_services/run_gpu_service.sh
```

`LOCANY_CACHE_DIR` 和 `SAM31_CACHE_DIR` 是服务自己的结果缓存目录。本地 Workbench 或 LocateAnything Qt 工具使用的 SFTP 上传目录不是缓存目录，且必须分别落在对应的 `*_ALLOWED_ROOTS` 内。协作平台不连接 GPU Services。

`LOCANY_DEVICES` 和 `SAM31_DEVICES` 使用逗号分隔设备，例如 `cuda:0,cuda:1`。两个运行时共享同一个设备池：同一张 GPU 同时只运行一个任务，不同 GPU 上的任务可以并行。请求不传 `device`（或传 `auto`）时会自动领取空闲 GPU；传 `cuda:N` 时会固定在该卡排队。请求指定的设备必须存在于对应的 `*_DEVICES` 列表中。

LocateAnything 默认设置 `LOCANY_KEEP_MODEL_LOADED=0`，任务结束后会删除模型 worker、执行 Python GC，并清理对应设备的 CUDA 缓存，从而释放模型占用的显存。CUDA 上下文本身仍可能在 `nvidia-smi` 中保留少量显存，这是 PyTorch 进程级上下文，不是模型泄漏。若更看重连续任务的启动速度，可设为 `1`，代价是每张已使用 GPU 上会常驻一份模型。

SAM3.1 每个任务使用独立子进程，子进程结束时由操作系统释放该任务的全部 CUDA 资源；设备池用于避免多个 SAM3.1 或 LocateAnything 任务同时挤占同一张卡。

## 验证

```bash
curl http://127.0.0.1:9010/api/health
curl http://127.0.0.1:9010/api/sam31/health
curl http://127.0.0.1:9010/api/locateanything/health
```

提交一个图片目录任务：

```bash
curl -X POST http://127.0.0.1:9010/api/locateanything/image-jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/data/images/task01",
    "recursive": true,
    "prompt": "person",
    "categories": ["person"],
    "class_map": {"person": 0},
    "task": "ground_multi",
    "copy_images": true,
    "device": "cuda:0"
  }'
```

轮询返回的 `job_id`：

```bash
curl http://127.0.0.1:9010/api/locateanything/image-jobs/JOB_ID
curl -OJ http://127.0.0.1:9010/api/locateanything/image-jobs/JOB_ID/annotations-zip
```

根健康检查会同时报告两个运行时的配置。LocateAnything 健康结果中的 `worker_available` 和 `worker_importable` 均应为 `true`；前者为 `false` 时检查 `LOCATEANYTHING_ROOT` 是否包含 `locateanything_worker.py`，后者为 `false` 时根据 `worker_import_error` 修复运行环境依赖。PyTorch/SymPy 环境至少需要 `mpmath>=1.3`。

## 进程守护与更新

正式部署建议使用 systemd、Supervisor 或其他进程管理器，工作目录设为仓库根目录并以受限账号运行。缓存清理由独立任务执行，不要在服务进程内部删除运行中的任务目录。

升级时停止这一项统一服务，更新仓库和外部 LocateAnything 运行时，检查 `/api/health`，再分别提交一段已知视频的 LocateAnything 与 SAM3.1 作业。任何 `gpu_services/` 适配层变更都需要重启统一服务。
