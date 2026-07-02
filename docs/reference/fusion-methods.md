# Fusion 方法分析与新增策略

本项目的 tracker 会分别产出 `forward_tracks` 和 `backward_tracks`。Fusion 阶段负责把这些 tracklet 组成最终轨迹组件，然后交给 `build_final_tracks` 做框融合、短轨过滤和平滑。

## 已有方法分析

| 方法名 | `fusion.method` | 逻辑 | 优点 | 主要问题 |
| --- | --- | --- | --- | --- |
| 双向互选 IoU | `bidirectional_iou` | 前向和后向 tracklet 只在重叠帧足够、平均 IoU 达标、首尾中心差不过大，并且互为最佳匹配时才合并。未合并的前向和后向 tracklet 都会单独进入结果。 | 很保守，不容易把两个不同目标强行串在一起；后向轨迹能补回前向漏掉的目标。 | 互选失败或阈值偏严时，后向重复轨迹也会保留下来，容易出现同一目标的影子 ID；对前后向断裂但重叠不足的轨迹无能为力。 |

## 新增方法

| 方法名 | `fusion.method` | 逻辑 | 更适合解决的问题 | 可能代价 |
| --- | --- | --- | --- | --- |
| 只用前向 | `forward_only` | 完全忽略后向 tracklet，每条前向 tracklet 单独形成最终组件。 | 建立最干净的基线；避免所有后向重复 ID；适合后向质量明显差于前向时。 | 失去后向补漏能力，前向断轨不会被修复。 |
| 只用后向 | `backward_only` | 完全忽略前向 tracklet，每条后向 tracklet 单独形成最终组件。 | 调试对照；适合检查后向是否比前向稳定。 | 正常生产中较少作为默认策略。 |
| 前向主导互选 | `bidirectional_iou_forward_primary` | 以前向 tracklet 为主，只把互选成功的后向 tracklet 合进去，丢弃未匹配后向 tracklet。 | 减少后向影子轨迹，同时保留一部分双向框融合和平滑收益。 | 后向独有的真实目标会被丢弃。 |
| 前向主导 + 唯一后向 | `bidirectional_iou_forward_unique` | 以前向为主，互选成功则合并；未互选但与任一前向高度重叠的后向视为重复并丢弃；完全不像任何前向的后向轨迹保留。 | 在减少重复 ID 的同时，保留后向补漏能力。通常是比默认更均衡的选择。 | 如果前向和后向对相邻不同目标有高重叠，可能误丢后向目标。 |
| 全兼容合并 | `bidirectional_iou_all_pairs` | 不要求互为最佳，只要前后向 pair 达到 IoU/中心约束就 union 到同一组件。 | 更积极地合并碎片，减少同一目标多段 ID。 | 拥挤场景下有串轨风险，比默认更激进。 |
| 双向互选 + NMS 去重 | `bidirectional_iou_nms` | 先跑原 `bidirectional_iou`，再按组件质量保留高分组件，丢弃与已保留组件高度重叠的重复组件。 | 针对默认方法产生的后向影子 ID 做后处理去重，改动相对稳。 | 如果两个真实目标长时间重叠，可能误删较短或较低质量的一条。 |

## 推荐尝试顺序

| 场景 | 建议方法 |
| --- | --- |
| 想确认问题是否来自后向 fusion | `forward_only` |
| 默认方法有明显重复 ID/影子轨迹 | `bidirectional_iou_forward_unique` 或 `bidirectional_iou_nms` |
| 默认方法断轨较多、目标不拥挤 | `bidirectional_iou_all_pairs` |
| 后向结果整体不可信 | `bidirectional_iou_forward_primary` 或 `forward_only` |
| 需要最大限度保持旧行为但减少重复 | `bidirectional_iou_nms` |

## 示例

```json
{
  "fusion": {
    "method": "bidirectional_iou_forward_unique",
    "iou_fuse": 0.5,
    "min_track_len": 10,
    "smooth_window": 5
  }
}
```
