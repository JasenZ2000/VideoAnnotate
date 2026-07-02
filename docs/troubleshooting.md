# 常见问题排查

## 本地等待远端 GPU 任务时超时

HTTP 超时只应作用于提交、轮询和下载等单次请求。先确认客户端已经获得 `job_id`，并且正在轮询 `/api/jobs/{id}`。不要把一次推理实现成持续数小时的单个 HTTP 请求。只有结果 ZIP 下载确实超时时，才需要增加 `download_timeout`。

## LocateAnything 在 `generate()` 阶段显存不足

模型分片或 `max_memory` 只控制模型权重的放置，无法限制生成阶段的峰值临时显存。应拆分长视频、减小 `resize_long_edge`、降低 `max_new_tokens`、使用正确的数据类型，并避免同一模型并发执行多个任务。调整显存分配器之前，先检查 GPU 上是否存在无关进程。

## LocateAnything 在 `str | Image.Image` 处报错

当前环境使用的是 Python 3.9 或更早版本。请使用 Python 3.10 以上版本重新创建环境。

## CUDA 或 PyTorch 版本错误

安装项目依赖前，先从 PyTorch 官方软件源安装与服务器驱动和 CUDA 运行时匹配的版本。根目录依赖文件有意不安装 PyTorch。可使用以下命令验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## GPU 服务拒绝视频路径

视频路径必须在 Linux 上真实存在，并位于 `SAM31_ALLOWED_ROOTS` 或 `LOCANY_ALLOWED_ROOTS` 之下。使用共享存储时检查本地与远端路径前缀映射；使用 SFTP 时检查上传目录、Linux 账号权限，并确认 API 允许访问该上传目录。

## SFTP 要求登录

SFTP 始终需要身份认证。必须配置用户名，并提供 SSH 密钥，或者通过 `sftp_password_env` 指定的环境变量提供密码。平台不会自动推断服务器账号和密码。

## 任务列表出现重复任务

SQLite 使用 `task_id` 主键，从数据层阻止重复任务。如果界面看起来仍有重复记录，应先确认浏览器是否连接了多个平台进程，并检查这些进程使用的 `ANNOTATION_PLATFORM_TASKS_DIR` 和 `ANNOTATION_PLATFORM_DB` 是否一致。旧 JSON 导入只会导入数据库中尚不存在的任务。

## 浏览器中已选文件名消失

浏览器文件输入框可能在界面重新渲染后被清空。应确认选中的文件已经保存在 JavaScript 状态中，并检查实际网络请求。上传成功后，文件会显示在任务详情中，并出现在所选视频的 `raw/` 目录下。

## 推荐的首轮检查

先使用已部署服务地址运行 `python scripts/check-services.py`，然后依次检查各服务控制台、平台任务详情中的事件列表和 SQLite 健康状态。远端任务状态响应会在条件允许时包含最终错误信息，以及经过长度限制的标准输出和标准错误。
