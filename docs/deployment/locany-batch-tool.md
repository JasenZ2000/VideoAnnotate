# LocateAnything 批量预标注工具

该工具独立于 workflow platform，是一个 PySide6 Qt 桌面程序，可完成视频批量预标注或图片目录预标注。

界面顶部的“推理对象”用于明确选择任务类型。视频与图片模式分别保存自己的输入、输出路径；切换到图片模式后，选择单个视频、帧间隔和视频目录后处理等控件会隐藏，避免两类任务混用。

## 启动

```powershell
.\locany_batch_tool\run.ps1
```

启动后会直接显示桌面窗口，不需要浏览器。常用设置会保存在当前 Windows 用户的 Qt 设置中，密码不会保存。

## 构建 Windows exe

```powershell
pip install -r requirements\locany-tool-windows.txt
.\scripts\windows\build-locany-batch-tool.ps1
```

生成文件为 `dist\LocateAnythingBatchTool.exe`，可直接双击运行。

## SFTP 模式

填写本地视频或目录、本地 ZIP 输出目录、GPU Services URL 和 SFTP 设置。工具会上传视频、提交 LocateAnything 作业，并把每个视频的标注 ZIP 下载为 `<视频名>_yolo.zip`。ZIP 内的 YOLO TXT 位于 `labels/`，Pascal VOC XML 位于 `annotations/`。

密码只随当前页面请求发送，不会写入浏览器存储。也可在项目根目录的 `.env.local` 中设置 `LOCANY_SFTP_PASSWORD`；该文件已被 Git 忽略。

图片模式只接受一个本地图片目录。工具保持相对子目录结构上传支持的图片，并把整个目录作为一个 GPU 任务提交；完成后下载 `<图片目录名>_images.zip`，其中 YOLO TXT 位于 `labels/`，Pascal VOC XML 位于 `annotations/`。

## 直连模式

直连模式表示视频与输出目录已经位于 GPU 服务器，不要求运行 Qt 工具的电脑挂载或访问这些文件。输入路径和输出路径均原样作为 Linux 路径发送给 GPU Services；目录枚举、视频读取和结果写入全部在服务器端完成，不执行 SFTP 或结果下载。

例如：

```text
输入：/data2/DET_Group/ZZS/data/embedded_cosmos/videos
输出：/data2/DET_Group/ZZS/data/embedded_cosmos/labels
```

GPU Services 必须配置视频与输出允许根目录：

```bash
export LOCANY_ALLOWED_ROOTS=/data/videos
export LOCANY_OUTPUT_ALLOWED_ROOTS=/data/labels
```

服务端会拒绝允许根目录之外的视频和输出位置。每个视频的 `labels/`、`annotations/`、元数据、原始回答和 ZIP 会写入输出目录下以视频名命名的子目录。

图片模式下，输入必须是 GPU 服务器上的 Linux 图片目录，输出是该次目录任务的服务器端结果目录；路径不会在 Windows 本地解析或添加盘符。图片模式通过独立的 `/api/locateanything/image-jobs` 接口提交，连接测试也会提前确认该接口是否存在。

## 多 GPU 并行

“CUDA 设备号”支持使用逗号填写多张卡，例如：

```text
0,1,3
```

工具为每张卡启动一个任务线程，同时处理多个视频；某张卡完成一个视频后会继续领取下一个等待中的视频。单个视频仍只使用一张 GPU。GPU Services 端必须在 `LOCANY_DEVICES` 中启用相同设备，例如 `LOCANY_DEVICES=cuda:0,cuda:1,cuda:3`。

图片目录当前作为一个整体任务执行，因此图片模式只能填写一张 GPU；界面会在提交前阻止多 GPU 配置。视频模式的多 GPU 并行行为不受影响。

SFTP 模式下上传与推理任务也会并行；直连模式不涉及上传。进度列表会显示每个视频请求和实际分配的 CUDA 设备。只填写一个编号时行为与旧版本一致。

## 类别映射

每行填写 `编号 类别名`：

```text
0 person
1 car
2 bicycle
```

Prompt 可独立填写；类别映射会作为 LocateAnything 的 `categories` 与 `class_map` 提交。

## Windows 本地预标注目录后处理

Qt 工具底部提供独立的 Windows 本地后处理区。填写视频目录和预标注目录后，工具按视频文件名（不含扩展名）匹配预标注子目录。例如：

```text
D:\videos\0001.mp4
D:\prelabels\0001\labels\
```

执行后整理为：

```text
D:\prelabels\0001\0001.mp4
D:\prelabels\0001\0001\
```

建议先点击“预览变更”。操作可以重复执行：已复制且大小一致的视频会被复用，已经改名的目录会被跳过；如果 `labels` 与同名目录同时存在，则只在没有内容冲突时合并。目标视频大小不同或同名标注文件内容不同时会报错，不会覆盖原文件。
