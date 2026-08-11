# LocateAnything 性能与输出对照测试

这个测试脚本在同一组视频帧上逐项运行 LocateAnything，并比较速度、显存和框输出。它不连接 GPU Service，也不会提交或修改现有 Job。

每个 case 使用独立 Python 子进程。模型加载失败、CUDA OOM、batch runtime 缺失或单项超时都会写入报告，然后继续下一项。这样可以安全测试较大的 batch size，也能在每个 case 结束后由操作系统彻底释放 CUDA 上下文。

视频只在父进程中解码一次，固定测试帧会以 JPEG 保存并被所有 case 共用。报告单独记录帧提取时间和图片加载/缩放时间；FPS 指标专门衡量模型调用及同步，避免不同 case 重复视频 seek 干扰结论。

## 默认对照项

| Case | 用途 |
|---|---|
| `standard-slow` | 复现当前 GPU Service 的纯 NTP 基线 |
| `standard-hybrid` | PBD/MTP 优先，异常块回退 NTP |
| `standard-fast` | 只走快速 MTP，用来观察速度上限和输出稳定性 |
| `batch-hybrid-2` | 新上游 batch runtime，batch size 2 |
| `batch-hybrid-4` | 新上游 batch runtime，batch size 4，同时测试显存余量 |

`standard-slow` 仅作为输出一致性参考，不代表人工标注真值。其他 case 会计算相对 slow 的框级 IoU50 precision/recall，并生成画框预览供人工检查。

Standard 与 batch 使用不同的 attention 接入方式。默认 `--standard-attn sdpa`，避免安装 FlashAttention 后标准模型错误地自动选择其尚未实现的 LLM `flash_attention_2` 分支。`--batch-attn la_flash` 只作用于 batch runtime。5090 上安装并验证 MagiAttention 后，可另开输出目录使用 `--standard-attn magi` 对比。

## 前提

- 在部署 LocateAnything 的 Linux Python 环境中运行，确保 PyTorch 能识别目标 GPU。
- `LOCATE_ROOT` 必须包含 `locateanything_worker.py`。
- `MODEL_PATH` 必须是本地模型目录。
- batch case 还要求 Hugging Face 发布目录中的 `batch_utils/` 和 `kernel_utils/` 可导入。缺失时 batch case 会标记为 `unsupported`，standard case 仍继续。
- 测试前确认指定 CUDA 卡没有承载正式 Job。脚本不会抢占、停止或查询 GPU Service 任务。

## 一键运行

```bash
chmod +x scripts/linux/benchmark-locateanything.sh

./scripts/linux/benchmark-locateanything.sh \
  /path/to/Eagle/Embodied \
  /path/to/LocateAnything-3B \
  /path/to/test.mp4 \
  cuda:0 \
  /path/to/benchmark-output
```

前五个参数依次是：上游库目录、模型目录、测试视频、CUDA 设备、全新或空的输出目录。

更接近当前服务配置的完整示例：

```bash
./scripts/linux/benchmark-locateanything.sh \
  /data2/DET_Group/ZZS/locateAnything/eagle/Embodied \
  /data2/DET_Group/ZZS/locateAnything/eagle/Embodied/pretrain/LocateAnything-3B \
  /data/test/sample.mp4 \
  cuda:0 \
  /data/test/locany-benchmark-01 \
  --prompt 'person</c>car' \
  --dtype bf16 \
  --frames 8 \
  --warmup-frames 1 \
  --frame-step 1 \
  --resize-long-edge 1024 \
  --max-new-tokens 512 \
  --standard-attn sdpa \
  --timeout-per-case 1800
```

如果 LocateAnything 安装在专用虚拟环境，可指定解释器：

```bash
PYTHON_BIN=/path/to/venv/bin/python \
  ./scripts/linux/benchmark-locateanything.sh LOCATE_ROOT MODEL_PATH VIDEO cuda:0 OUTPUT_DIR
```

## 常用变体

先跑三个不依赖 batch runtime 的生成模式：

```bash
./scripts/linux/benchmark-locateanything.sh LOCATE_ROOT MODEL_PATH VIDEO cuda:0 OUTPUT_DIR \
  --cases standard-slow,standard-hybrid,standard-fast
```

单独摸 batch 上限：

```bash
./scripts/linux/benchmark-locateanything.sh LOCATE_ROOT MODEL_PATH VIDEO cuda:0 OUTPUT_DIR \
  --cases batch-hybrid-1,batch-hybrid-2,batch-hybrid-4,batch-hybrid-8
```

取视频中间隔一秒的帧进行覆盖面更高的测试，例如 25 FPS 视频：

```bash
./scripts/linux/benchmark-locateanything.sh LOCATE_ROOT MODEL_PATH VIDEO cuda:0 OUTPUT_DIR \
  --frames 12 --frame-step 25
```

在支持的环境中明确测试 FlashAttention 视觉路径：

```bash
./scripts/linux/benchmark-locateanything.sh LOCATE_ROOT MODEL_PATH VIDEO cuda:0 OUTPUT_DIR \
  --vision-attn flash_attention_2 --batch-attn la_flash
```

注意：不要给 standard case 选择 LLM `flash_attention_2`。LocateAnything 的标准 Qwen2 forward 仅使用 `sdpa` 或 `magi`；LA Flash 的 LLM 替换逻辑属于 batch runtime。

## OOM 和错误处理

- 每个 case 是独立进程；case 结束或被超时终止后，CUDA 显存由进程退出释放。
- batch case OOM 时自动执行 `empty_cache()`，将 batch size 对半降低并重试当前批次，直到 batch 1。
- batch 1 仍 OOM 时，该 case 标记为 `oom`，保存错误和已完成帧，然后继续下一 case。
- 缺少 `batch_utils`、`kernel_utils` 或可选 attention 依赖时，batch case 标记为 `unsupported`。
- 超过 `--timeout-per-case` 时父进程终止该子进程并标记为 `timeout`。
- 输出目录非空时拒绝运行，避免覆盖此前基准结果。
- 默认不因某一个 case 失败而终止整轮。

不要为了规避 OOM 自动缩小图片尺寸后仍把结果当成同一对照。需要测试 768 或 512 长边时，分别使用新的输出目录运行，并明确传入 `--resize-long-edge`。

## 输出

```text
benchmark-output/
  benchmark_config.json
  frames.json
  frames/
  cases/
    standard-slow.json
    standard-slow_previews/
    standard-hybrid.json
    standard-hybrid_previews/
    ...
  summary.json
  summary.csv
  summary.md
```

重点查看 `summary.md`：

- `FPS`：测量帧吞吐；
- `vs slow`：相对当前 slow 基线的加速倍数；
- `P50/P95`：单帧等效延迟；
- `Peak GiB`：测量阶段的 PyTorch 峰值 reserved memory；
- `P@IoU50/R@IoU50`：相对 slow 输出的匹配程度；
- `Status`：`ok`、`oom`、`unsupported`、`timeout` 或 `failed`。

人工检查时重点比较：漏框、重复框、密集目标中的框漂移、类别串位，以及 `fast` 是否比 `hybrid` 更容易输出格式异常。
