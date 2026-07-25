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

浏览器访问 `http://<服务器IP>:8088`。首次打开时创建第一个管理员账号，之后由管理员创建其他用户。正式多人使用建议按下一节启用 HTTPS。

默认数据库为任务目录下的 `platform.sqlite3`，也可通过 `-Database` 或 `ANNOTATION_PLATFORM_DB` 指定。数据库包含账号、任务、Part、计时和备注；平台不再保存视频或标注文件。

## HTTPS

### 自动生成临时证书

内部临时使用可让平台首次启动时自动生成自签名证书：

```powershell
.\workflow_platform\run.ps1 `
  -HostName 0.0.0.0 `
  -Port 8443 `
  -TasksDir D:\annotation_platform `
  -AutoHttps `
  -TlsHosts "annotation-server,192.168.1.20"
```

把 `annotation-server` 和 `192.168.1.20` 换成浏览器实际访问平台时使用的机器名和 IP。即使省略 `-TlsHosts`，平台也会自动加入 `localhost`、本机名称和能够发现的本机 IP。

证书和私钥默认保存在 `D:\annotation_platform\tls\selfsigned-cert.pem` 和 `selfsigned-key.pem`。后续启动会复用同一证书；增加 `TlsHosts` 中的地址或证书临近过期时会自动重新生成。浏览器会对自签名证书显示警告，内部使用时可以手动继续访问；路径复制功能即使 Clipboard API 被浏览器禁用，也会自动使用兼容方式复制。

也可通过环境变量启用：

```powershell
$env:ANNOTATION_PLATFORM_AUTO_HTTPS = "1"
$env:ANNOTATION_PLATFORM_TLS_HOSTS = "annotation-server,192.168.1.20"
.\workflow_platform\run.ps1
```

### 替换受信任证书

以后准备好公司内部 CA 或正式 CA 签发的 PEM 证书链和私钥后，直接执行：

```powershell
.\workflow_platform\run.ps1 `
  -HostName 0.0.0.0 `
  -Port 8443 `
  -TasksDir D:\annotation_platform `
  -SslCertFile D:\certs\platform-fullchain.pem `
  -SslKeyFile D:\certs\platform-private-key.pem
```

显式证书优先于自动证书。访问 `https://<证书中的服务器域名>:8443`，平台会自动启用 Secure Cookie。也可以使用环境变量：

```powershell
$env:ANNOTATION_PLATFORM_SSL_CERTFILE = "D:\certs\platform-fullchain.pem"
$env:ANNOTATION_PLATFORM_SSL_KEYFILE = "D:\certs\platform-private-key.pem"
.\workflow_platform\run.ps1
```

证书和私钥必须同时配置；私钥文件应只授予平台服务账号读取权限。

端口占用时可显式替换已有平台进程：

```powershell
.\scripts\windows\run-platform.ps1 -Port 8088 -Restart
```

健康检查：

```powershell
Invoke-RestMethod https://<证书中的服务器域名>:8443/api/health
```

## 备份

停止服务后备份 `platform.sqlite3`。在线备份应使用 SQLite backup API 或支持 WAL 的卷快照，不能只复制正在写入的主数据库文件。
