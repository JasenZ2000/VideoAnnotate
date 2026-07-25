# 配置说明

## 协作平台

协作平台不再读取 GPU 或视频处理配置。可用参数和环境变量：

| 命令行 | 环境变量 | 含义 |
| --- | --- | --- |
| `--host` | `ANNOTATION_PLATFORM_HOST` | 监听地址 |
| `--port` | `ANNOTATION_PLATFORM_PORT` | 监听端口 |
| `--tasks-dir` | `ANNOTATION_PLATFORM_TASKS_DIR` | 默认数据库目录 |
| `--database` | `ANNOTATION_PLATFORM_DB` | SQLite 文件路径 |
| `--ssl-certfile` | `ANNOTATION_PLATFORM_SSL_CERTFILE` | HTTPS 使用的 PEM 证书链 |
| `--ssl-keyfile` | `ANNOTATION_PLATFORM_SSL_KEYFILE` | HTTPS 使用的 PEM 私钥 |
| `--auto-https` | `ANNOTATION_PLATFORM_AUTO_HTTPS=1` | 自动生成并复用自签名证书 |
| `--tls-hosts` | `ANNOTATION_PLATFORM_TLS_HOSTS` | 自动证书额外包含的逗号分隔域名/IP |
| `--tls-cert-dir` | `ANNOTATION_PLATFORM_TLS_CERT_DIR` | 自动证书保存目录，默认是任务目录下的 `tls` |
| - | `ANNOTATION_PLATFORM_SESSION_DAYS` | 登录会话有效天数 |
| - | `ANNOTATION_PLATFORM_SECURE_COOKIE=1` | TLS 在反向代理终止时手动启用 Secure Cookie；直接配置证书时会自动启用 |

证书和私钥必须同时配置。直接由平台提供 HTTPS 时，会话 Cookie 自动增加 `Secure` 属性。显式 PEM 证书优先于自动证书，因此后续替换受信任证书不需要删除原来的自签名文件。

SQLite 应放在服务器本地可靠磁盘，不建议放在文件锁支持不完整的网络共享目录。

## 本地 Workbench

Annotator 优先使用工作区内的 `config.json`，项目级后备配置可由 `ANNOTATOR_CONFIG` 指定。其中包括 SAM3.1、SFTP/共享路径和导出类别名称等设置。LocateAnything 预标注使用独立的 Qt 批量工具，不再由 Workbench 调用。

## GPU Services

SAM3.1 使用 `SAM31_*` 环境变量，LocateAnything 使用 `LOCANY_*` 环境变量。完整变量见 `configs/gpu-services.env.example` 和 GPU 部署文档。
