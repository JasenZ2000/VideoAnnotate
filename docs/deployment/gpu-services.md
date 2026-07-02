# Linux GPU Service Deployment

## General Rules

Run SAM3.1 and LocateAnything in separate Python environments. Install a PyTorch build that matches the host driver/CUDA runtime before installing model dependencies. Bind service ports to the trusted network only.

Copy `configs/gpu-services.env.example` to a server-local environment file or export the variables from your process supervisor. Do not commit real model paths, hostnames or credentials.

## SAM3.1

SAM3.1 depends on the existing ComfyUI installation and checkpoint:

```bash
conda activate sam31-comfy
cd /srv/video-annotation-workflow
pip install -r requirements/gpu-sam31.txt

export SAM31_COMFY_ROOT=/opt/ComfyUI
export SAM31_CHECKPOINT=/models/sam3.1_multiplex_fp16.safetensors
export SAM31_CACHE_DIR=/data/annotation-cache/sam31
export SAM31_ALLOWED_ROOTS=/data/annotation-transfer/sam31/videos
export SAM31_DEVICE=cuda:0
./scripts/linux/run-sam31-server.sh
```

Verify:

```bash
curl http://127.0.0.1:9001/api/health
```

## LocateAnything

Use Python 3.10 or newer. Python 3.9 cannot evaluate type-union annotations used by the Hugging Face remote processor code.

```bash
conda activate locateanything
cd /srv/video-annotation-workflow
# Install the CUDA-matched torch build first.
pip install -r requirements/gpu-locateanything.txt

export LOCANY_MODEL=nvidia/LocateAnything-3B
export LOCANY_CACHE_DIR=/data/annotation-cache/locateanything
export LOCANY_ALLOWED_ROOTS=/data/annotation-transfer/locateanything/videos
export LOCANY_DEVICE=cuda:1
export LOCANY_DTYPE=bf16
./scripts/linux/run-locateanything-server.sh
```

Verify:

```bash
curl http://127.0.0.1:9011/api/health
```

The service serializes inference jobs around one model worker. Lowering frame resolution and splitting videos are the primary controls for memory and turnaround time. `max_memory` during model loading does not cap temporary KV-cache/attention allocations during `generate()`.

## Process Supervision

Use systemd, Supervisor or another process manager in production. Set the working directory to the repository root, load the service-specific environment, and restart on failure. Keep cache cleanup outside the running process and retain failed-job logs long enough for diagnosis.

## Updating

1. Stop the target service.
2. Pull a reviewed Git revision.
3. Update only that service's environment.
4. Start it and verify `/api/health`.
5. Submit a short known-good video job before reopening team access.

Restart LocateAnything after changes to `locateanything_video_server.py`, including class mapping behavior.
