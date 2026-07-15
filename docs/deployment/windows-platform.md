# Windows 多人标注协作平台部署

## 安装

```powershell
git clone <repository-url> D:\VideoAnnotate
cd D:\VideoAnnotate
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\platform.txt
```

## 启动

```powershell
.\scripts\windows\run-platform.ps1 `
  -HostName 0.0.0.0 `
  -Port 8088 `
  -TasksDir D:\annotation_platform
```

浏览器访问 `http://<服务器IP>:8088`。首次打开时创建第一个管理员账号，之后由管理员创建其他用户。

默认数据库为任务目录下的 `platform.sqlite3`，也可通过 `-Database` 或 `ANNOTATION_PLATFORM_DB` 指定。数据库包含账号、任务、Part、计时和备注；平台不再保存视频或标注文件。

端口占用时可显式替换已有平台进程：

```powershell
.\scripts\windows\run-platform.ps1 -Port 8088 -Restart
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8088/api/health
```

## 备份

停止服务后备份 `platform.sqlite3`。在线备份应使用 SQLite backup API 或支持 WAL 的卷快照，不能只复制正在写入的主数据库文件。
