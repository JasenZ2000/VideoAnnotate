# Linux GPU 服务部署

## 通用要求

SAM3.1 和 LocateAnything 必须运行在独立的 Python 环境中。安装模型依赖前，应先安装与主机驱动和 CUDA 运行时匹配的 PyTorch。服务端口只能绑定到可信内部网络。

把 `configs/gpu-services.env.example` 复制为服务器本地环境文件，或通过进程管理器导出其中变量。真实模型路径、主机名和凭据不得提交到 Git。

## SAM3.1

SAM3.1 依赖服务器上已有的 ComfyUI 和模型权重：

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

验证服务：

```bash
curl http://127.0.0.1:9001/api/health
```

## LocateAnything

必须使用 Python 3.10 以上版本。Python 3.9 无法解析 Hugging Face 远端处理器代码使用的联合类型注解。

```bash
conda activate locateanything
cd /srv/video-annotation-workflow
# 先安装与 CUDA 匹配的 PyTorch
pip install -r requirements/gpu-locateanything.txt

export LOCANY_MODEL=nvidia/LocateAnything-3B
export LOCANY_CACHE_DIR=/data/annotation-cache/locateanything
export LOCANY_ALLOWED_ROOTS=/data/annotation-transfer/locateanything/videos
export LOCANY_DEVICE=cuda:1
export LOCANY_DTYPE=bf16
./scripts/linux/run-locateanything-server.sh
```

验证服务：

```bash
curl http://127.0.0.1:9011/api/health
```

服务会围绕单个模型工作进程串行执行推理任务。降低输入帧分辨率并提前拆分视频，是控制显存和任务周转时间的主要手段。模型加载阶段设置的 `max_memory` 无法限制 `generate()` 过程中 KV 缓存或注意力计算产生的临时显存峰值。

## 进程守护

正式部署应使用 systemd、Supervisor 或其他进程管理器。工作目录应设置为仓库根目录，加载对应服务的环境变量，并在异常退出时自动重启。缓存清理由独立任务执行，不要在服务进程内部清理；失败任务日志应保留足够时间以便排查。

## 更新流程

1. 停止目标服务。
2. 拉取经过审核的 Git 版本。
3. 只更新该服务对应的 Python 环境。
4. 启动服务并检查 `/api/health`。
5. 用一个已知结果的短视频测试成功后，再恢复团队访问。

修改 `locateanything_video_server.py` 后必须重启 LocateAnything 服务，其中也包括类别映射逻辑的修改。
