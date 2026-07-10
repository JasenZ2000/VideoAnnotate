# Windows 公共平台部署

## 环境要求

- Windows 10、Windows 11 或 Windows Server
- Python 3.10 以上版本，推荐 Python 3.11
- FFmpeg 或 OpenCV 能够读取的视频编码
- 足够保存源视频、分段副本和导出标注包的数据盘空间
- 能够访问 Linux GPU 服务端口；使用 SFTP 时还需访问 22 端口

## 安装

```powershell
git clone <repository-url> D:\video-annotation-workflow
cd D:\video-annotation-workflow
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\platform.txt
Copy-Item configs\platform.example.json configs\platform.local.json
```

编辑 `configs\platform.local.json`，填写 LocateAnything 服务地址和文件传输配置。不要提交该本机配置文件。

使用密码方式连接 SFTP 时，只把密码写入平台服务账号的环境变量：

```powershell
[Environment]::SetEnvironmentVariable(
  "LOCANY_SFTP_PASSWORD",
  "<password>",
  "User"
)
```

无人值守运行时，优先使用 SSH 密钥和权限受限的专用 Linux 账号。

## 启动

```powershell
$env:ANNOTATION_PLATFORM_TASKS_DIR="D:\annotation_tasks"
$env:ANNOTATION_PLATFORM_DB="D:\annotation_tasks\platform.sqlite3"
$env:ANNOTATION_PLATFORM_CONFIG="$PWD\configs\platform.local.json"
.\scripts\windows\run-platform.ps1
```

平台已经在运行时，可关闭占用同一端口的旧平台进程并在当前窗口重启：

```powershell
.\scripts\windows\run-platform.ps1 -Restart
```

脚本只会自动停止监听目标端口的 `python` 或 `pythonw` 进程；如果端口由其他程序占用，会拒绝终止并报告进程信息。平台在当前窗口前台运行，按 `Ctrl+C` 停止。

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8088/api/health
```

只允许部门内部网络访问 TCP 8088 端口。长期运行时，可通过 Windows 服务包装工具或任务计划程序执行启动命令，并使用对任务数据盘具有读写权限的专用服务账号。

## 备份

备份必须同时包含 SQLite 数据库和完整任务文件目录。最稳妥的方式是先停止平台服务，再复制整个任务根目录；在线备份应使用 SQLite 备份 API 或卷快照，不能只复制主数据库而漏掉运行中的 WAL 文件。大文件上传或 ZIP 解压正在写入时不要直接复制任务目录。
