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
python -m compileall utils local_workbench workflow_platform gpu_services
python scripts/check-services.py --platform http://127.0.0.1:8088
```

集成测试应使用短小的合成视频和标注。不得提交真实生产视频、任务目录、凭据、模型权重或生成的压缩包。

本地数据集和探索性模型代码统一放在 `examples/local-data/`，避免污染仓库根目录或被意外提交到 Git。

## 代码职责边界

- 任务编排逻辑放在 `workflow_platform/`。
- 交互式标注与采样实现放在 `utils/annotator/` 和 `utils/frame_sampler/`。
- 本地工作台的单进程装配逻辑放在 `local_workbench/`。
- 本地工作台和平台共用的跟踪与格式转换逻辑放在 `utils/mot_pipeline/`。
- GPU HTTP 接口集中在 `gpu_services/`，以 `/api/sam31` 和 `/api/locateanything` 分隔。
- LocateAnything 是外部运行时；项目内适配层通过 `LOCATEANYTHING_ROOT` 加载其 `locateanything_worker.py`。

远端 API 发生变化时，必须在同一个版本中同步修改客户端、示例配置、API 文档和健康检查或集成测试。

## 发布检查清单

1. 执行单元测试和编译检查。
2. 在 Windows 上测试平台的任务创建、上传和分段流程。
3. 在 Linux 上通过同一个 GPU 服务分别用一个短视频测试 LocateAnything 分段任务和 SAM3.1 目标框任务。
4. 检查导出 YOLO 中的类别 ID。
5. 确认暂存区不包含 `*.local.json`、凭据、视频或模型权重。
6. 为可部署提交打标签，并记录统一服务、LocateAnything 运行时与 SAM3.1 ComfyUI 运行时的版本。
