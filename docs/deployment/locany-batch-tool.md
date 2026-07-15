# LocateAnything 批量预标注工具

该工具独立于 workflow platform，是一个 PySide6 Qt 桌面程序，专门完成视频批量预标注。

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

## 多 GPU 并行

“CUDA 设备号”支持使用逗号填写多张卡，例如：

```text
0,1,3
```

工具为每张卡启动一个任务线程，同时处理多个视频；某张卡完成一个视频后会继续领取下一个等待中的视频。单个视频仍只使用一张 GPU。GPU Services 端必须在 `LOCANY_DEVICES` 中启用相同设备，例如 `LOCANY_DEVICES=cuda:0,cuda:1,cuda:3`。

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
