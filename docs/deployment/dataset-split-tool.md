# 标注数据汇总均分工具

`AnnotationDatasetSplitter.exe` 是独立的 Windows Qt 工具。默认读取与 EXE 同目录的 `ori`，也可在界面中选择其他源目录。

工具递归查找同时含有以下三个目录的数据集根目录：

```text
ori/
└── 任意子目录/
    ├── images/
    ├── labels/
    └── annotations/
```

每张图片按文件名 stem 匹配同名 YOLO TXT 和 VOC XML。完整样本按界面设置的份数均衡分配，输出结构为：

```text
split_output/
├── part_001/
│   ├── images/
│   ├── labels/
│   └── annotations/
└── part_002/
    ├── images/
    ├── labels/
    └── annotations/
```

为了防止误覆盖，输出目录已存在时工具会停止。缺少配对文件、重复 stem 或份数大于样本数时也不会开始复制。

构建命令：

```powershell
.\scripts\windows\build-dataset-split-tool.ps1 -Python .venv\Scripts\python.exe
```
