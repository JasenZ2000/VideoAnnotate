# 跟踪方法说明

通过配置中的 `tracking.method` 选择跟踪方法。每种方法都读取逐帧 YOLO 检测框，并向统一的轨迹融合阶段返回轨迹片段。

| 方法 | 配置值 | 行为与取舍 |
| --- | --- | --- |
| IoU 与卡尔曼滤波 | `iou_kalman` | 使用贪心 IoU 关联和运动预测的保守基线方法 |
| SORT 别名 | `sort` | 与基线实现相同，用于兼容常见配置名称 |
| ByteTrack 风格 | `bytetrack` | 先关联高分检测，再用低分检测恢复轨迹；需要真实置信度才有明显效果 |
| OC-SORT 风格 | `oc_sort` | 使用最近观测到的运动方向，提高短时遮挡后的恢复能力 |
| BoT-SORT 启发实现 | `bot_sort` | 增加置信度融合 IoU 和观测间隔速度；本项目不包含 ReID 和相机运动补偿 |
| Hybrid-SORT 启发实现 | `hybrid_sort` | 结合运动方向、置信度和目标框高度一致性 |
| Deep OC-SORT 启发实现 | `deep_oc_sort` | YOLO TXT 输入不包含 ReID 特征，因此使用本地运动和弱线索后备实现 |
| C-BIoU 启发实现 | `cbiou` | 使用级联扩框 IoU 关联，适合不规则运动和短暂无重叠的检测框 |
| SparseTrack 启发实现 | `sparse_track` | 根据伪深度对检测框分组并由近到远匹配，是当前拥挤场景的默认方法 |

这些方法都是只使用目标框的本地适配实现。对于依赖外观特征、相机标定或检测器内部信息的原始算法，本项目并未进行完整复现。

## 重要参数

| 字段 | 作用 |
| --- | --- |
| `iou_match` | 主要关联阈值 |
| `max_missed` | 轨迹允许连续未匹配的帧数 |
| `class_agnostic` | 是否允许跨类别 ID 匹配，通常保持为 `false` |
| `score_high`、`score_low` | 两阶段关联方法使用的高低置信度区间 |
| `new_track_score` | 创建新轨迹所需的最低置信度 |
| `low_iou_match`、`recover_iou_match` | 恢复阶段使用的宽松 IoU 阈值 |
| `velocity_weight` | 运动方向在线索融合中的权重 |
| `height_weight` | 目标框高度一致性的权重 |
| `cbiou_small_buffer`、`cbiou_large_buffer` | C-BIoU 的扩框比例 |
| `sparse_depth_bins` | 伪深度分组数量 |
| `sparse_cross_depth_iou` | 跨相邻深度组匹配时使用的高 IoU 后备阈值 |

如果输入没有第六列置信度，解析器会把分数视为 `1.0`，此时依赖置信度的恢复逻辑基本不会产生实际差异。

## 配置示例

```json
{
  "tracking": {
    "method": "sparse_track",
    "iou_match": 0.3,
    "max_missed": 15,
    "class_agnostic": false,
    "sparse_depth_bins": 4,
    "sparse_cross_depth_iou": 0.75
  }
}
```
