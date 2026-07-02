# 配置说明

配置文件是一个 JSON 覆盖层，会合并到 `mot_pipeline.config.DEFAULT_CONFIG`。没有显式填写的字段继续使用代码中的默认值。

## 配置文件与环境变量

- 根目录的 `config.json` 是安全的本地默认配置，不包含真实服务地址和凭据。
- `configs/platform.example.json` 是 Windows 公共机平台配置模板。
- `configs/annotator.example.json` 是员工标注工作区配置模板。
- `configs/gpu-services.env.example` 列出了 GPU 服务使用的环境变量。
- `*.local.json` 已被 Git 忽略，适合保存本机配置。

平台按以下优先级选择配置：命令行 `--config`、环境变量 `ANNOTATION_PLATFORM_CONFIG`、根目录 `config.json`。Annotator 优先使用工作区内的 `config.json`；其项目级后备配置可通过 `ANNOTATOR_CONFIG` 覆盖。

## 平台数据库

平台默认把 SQLite 数据库保存为 `<tasks-dir>/platform.sqlite3`。可通过命令行 `--database` 或环境变量 `ANNOTATION_PLATFORM_DB` 指定其他位置。数据库应放在 Windows 公共机的本地可靠磁盘上，不建议直接放在不完整支持文件锁的网络共享目录中。

## 远端传输字段

| 字段 | 含义 |
| --- | --- |
| `server_url` | GPU 服务基础地址 |
| `video_transfer` | `path` 或 `sftp` |
| `local_path_prefix` | Windows 客户端可见的路径前缀 |
| `remote_path_prefix` | Linux 端对应的路径前缀 |
| `sftp_host`、`sftp_port`、`sftp_username` | SFTP 连接信息 |
| `sftp_password_env` | 保存密码的环境变量名称 |
| `sftp_key_path` | 可选的私钥路径 |
| `sftp_remote_dir` | GPU 服务器上传目录，该目录也必须被 API 允许访问 |
| `sftp_reuse_existing` | 条件允许时复用远端同名文件 |
| `request_timeout` | 提交和轮询等短请求的超时时间，不是完整推理时限 |
| `poll_interval` | 两次任务状态查询之间的秒数 |

## GPU 服务环境变量

SAM3.1 使用 `SAM31_*` 环境变量，LocateAnything 使用 `LOCANY_*` 环境变量。关键配置包括服务端口、缓存目录、模型或权重路径、CUDA 设备、数据类型和以逗号分隔的允许访问根目录。完整变量名称见环境变量模板。

## 类别映射

平台把任务类别保存为有序的 ID 与名称对。LocateAnything 会接收类别名称组成的 `categories`，以及如下形式的 `class_map`：

```json
{
  "categories": ["person", "car"],
  "class_map": {"person": 0, "car": 1}
}
```

类别名称应简单、明确，并尽量使用模型会原样返回的词。当前无法识别的标签会回退到请求中的默认类别 ID，因此在开始大规模推理前，必须先抽查多类别样例。
