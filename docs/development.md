# 开发说明

## 环境准备

```bash
python -m venv .venv
# 根据当前终端类型激活虚拟环境
pip install -r requirements/dev.txt
```

GPU 模型依赖被有意拆分到独立环境中。绝大多数平台、Annotator 和 MOT 代码开发不需要 GPU 环境。

## 基础检查

```bash
python -m pytest
python -m compileall annotator workflow_platform mot_pipeline sam31
python scripts/check-services.py --platform http://127.0.0.1:8088
```

集成测试应使用短小的合成视频和标注。不得提交真实生产视频、任务目录、凭据、模型权重或生成的压缩包。

本地数据集和探索性模型代码统一放在 `examples/local-data/`，避免污染仓库根目录或被意外提交到 Git。

## 代码职责边界

- 任务编排逻辑放在 `workflow_platform/`。
- 交互式标注编辑逻辑放在 `annotator/`。
- 可复用的跟踪与格式转换逻辑放在 `mot_pipeline/`。
- GPU HTTP 接口分别放在 `sam31/server.py` 和 `locateAnything/locateanything_video_server.py`。
- 修改 LocateAnything 时应优先调整适配层，尽量不要改动引入的上游模型内部代码。

远端 API 发生变化时，必须在同一个版本中同步修改客户端、示例配置、API 文档和健康检查或集成测试。

## 发布检查清单

1. 执行单元测试和编译检查。
2. 在 Windows 上测试平台的任务创建、上传和分段流程。
3. 在 Linux 上分别用一个短视频测试 LocateAnything 分段任务和 SAM3.1 目标框任务。
4. 检查导出 YOLO 中的类别 ID。
5. 确认暂存区不包含 `*.local.json`、凭据、视频或模型权重。
6. 为可部署提交打标签，并记录两个 GPU 服务各自使用的环境版本。
