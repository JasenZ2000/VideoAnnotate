# 员工端 Annotator 部署

Annotator 设计为在每位员工的 Windows 电脑上本地运行，使视频帧浏览和目标框编辑不受网络延迟影响。

## 从源码安装

```powershell
cd D:\video-annotation-workflow
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\annotator.txt
.\scripts\windows\run-annotator.bat
```

打开 `http://127.0.0.1:7860`。除非确实需要其他电脑访问该员工实例，否则监听地址应保持为 `127.0.0.1`。

## 工作区配置

标注包或工作区包含视频、`tracking_results.json` 和 `config.json`。以 `configs/annotator.example.json` 为模板，并设置：

- `sam31.server_url`：GPU API 地址；
- `video_transfer`：共享存储使用 `path`，上传文件使用 `sftp`；
- 共享存储的路径前缀，或 SFTP 主机、用户和远端目录；
- `exports.class_labels`：任务类别表。

使用密码认证时：

```powershell
$env:SAM31_SFTP_PASSWORD="<session-password>"
.\scripts\windows\run-annotator.bat
```

程序从 `sftp_password_env` 指定的环境变量读取密码。密码不得写入 JSON，也不得随审核结果上传到平台。部门安全策略允许时，优先使用 SSH 密钥。

## 使用 SAM3.1

打开工作区，选中或创建目标轨迹，在当前帧保存一个目标框，然后启动 `SAM31 Track Box`。本地服务会上传视频或转换共享路径、提交异步任务、轮询状态，并把返回的目标框合并到所选轨迹中。

必须人工检查生成的后续轨迹。SAM3.1 只是辅助编辑工具，不能代替人工审核结论。
