# Windows 本地标注工作台部署

每位员工的 Windows 电脑运行一个本地工作台服务：Annotator 与视频采样器共享同一个进程和端口。视频帧浏览、目标框编辑与采样计划都在本机完成，不受网络延迟影响。

## 从源码安装

```powershell
cd D:\video-annotation-workflow
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\annotator.txt
.\scripts\windows\run-local-workbench.bat
```

打开：

- `http://127.0.0.1:7860/annotator/`：视频标注；
- `http://127.0.0.1:7860/sampler/`：帧采样与 YOLO 训练集导出。

除非确实需要其他电脑访问该员工实例，否则监听地址应保持为 `127.0.0.1`。

## 打包 Windows 可执行文件

```powershell
pip install -r requirements\windows-build.txt
.\scripts\windows\build-local-workbench.ps1
```

生成文件为 `dist\VideoAnnotationWorkbench.exe`。该文件会启动同一个本地服务；默认保留控制台便于查看启动和错误日志。传入 `-Windowed` 可生成无控制台版本。

## 工作区配置

标注包或工作区包含视频、`tracking_results.json` 和 `config.json`。以 `configs/annotator.example.json` 为模板，并设置：

- `sam31.server_url` 和 `locateanything.server_url`：同一个统一 GPU API 地址；
- `video_transfer`：共享存储使用 `path`，上传文件使用 `sftp`；
- 共享存储的路径前缀，或 SFTP 主机、用户和远端目录；
- `exports.class_labels`：任务类别表。

使用密码认证时：

```powershell
$env:SAM31_SFTP_PASSWORD="<session-password>"
.\scripts\windows\run-local-workbench.bat
```

程序从 `sftp_password_env` 指定的环境变量读取密码。密码不得写入 JSON，也不得随审核结果上传到平台。部门安全策略允许时，优先使用 SSH 密钥。

## 使用 LocateAnything

Annotator 主界面的 **LocateAnything YOLO** 区域提供连接设置入口，可在不修改 JSON 的情况下调整：

- LocateAnything `server_url`；
- `path` 或 `sftp` 视频传输方式；
- SFTP 主机、端口、用户名、密码和私钥；
- 远端上传/缓存目录；
- 共享路径模式下的本地与远端路径前缀。

Prompt、最大推理帧数和 GPU 编号直接在 LocateAnything 执行区域填写。非敏感连接设置保存在当前浏览器；界面输入的 SFTP 密码只保留在当前页面内存中，关闭或刷新页面后消失，不会写入工作区配置和标注结果。远端上传目录必须位于 GPU 服务 `LOCANY_ALLOWED_ROOTS` 允许的目录内。

## 使用 SAM3.1

打开工作区，选中或创建目标轨迹，在当前帧保存一个目标框，然后启动 `SAM31 Track Box`。本地服务会上传视频或转换共享路径、提交异步任务、轮询状态，并把返回的目标框合并到所选轨迹中。

必须人工检查生成的后续轨迹。SAM3.1 只是辅助编辑工具，不能代替人工审核结论。

若要把独立生成的 `tracking_results.json` 作为新轨迹加入当前工作区，可在 Annotator 点击 **Append Tracking JSON**。工作台会自动重编号新增轨迹，避免与现有 `track_id` 冲突，并写回当前工作区的 `tracking_results.json`。
