# 配置说明

## 协作平台

协作平台不再读取 GPU 或视频处理配置。可用参数和环境变量：

| 命令行 | 环境变量 | 含义 |
| --- | --- | --- |
| `--host` | `ANNOTATION_PLATFORM_HOST` | 监听地址 |
| `--port` | `ANNOTATION_PLATFORM_PORT` | 监听端口 |
| `--tasks-dir` | `ANNOTATION_PLATFORM_TASKS_DIR` | 默认数据库目录 |
| `--database` | `ANNOTATION_PLATFORM_DB` | SQLite 文件路径 |
| - | `ANNOTATION_PLATFORM_SESSION_DAYS` | 登录会话有效天数 |
| - | `ANNOTATION_PLATFORM_SECURE_COOKIE=1` | HTTPS 部署时启用 Secure Cookie |

SQLite 应放在服务器本地可靠磁盘，不建议放在文件锁支持不完整的网络共享目录。

## 本地 Workbench

Annotator 优先使用工作区内的 `config.json`，项目级后备配置可由 `ANNOTATOR_CONFIG` 指定。其中包括 SAM3.1、LocateAnything、SFTP/共享路径和导出类别名称等设置。

## GPU Services

SAM3.1 使用 `SAM31_*` 环境变量，LocateAnything 使用 `LOCANY_*` 环境变量。完整变量见 `configs/gpu-services.env.example` 和 GPU 部署文档。
