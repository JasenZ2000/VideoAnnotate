# Tracking Methods

Set `tracking.method` in configuration. Every method consumes frame-level YOLO boxes and returns tracklets for the shared fusion stage.

| Method | Value | Behavior and trade-off |
| --- | --- | --- |
| IoU + Kalman | `iou_kalman` | Conservative baseline using greedy IoU association and motion prediction |
| SORT alias | `sort` | Alias of the baseline for familiar configuration names |
| ByteTrack style | `bytetrack` | Matches high-score detections first, then recovers tracks with low-score boxes; needs real confidence values |
| OC-SORT style | `oc_sort` | Uses recent observed direction to improve short occlusion recovery |
| BoT-SORT inspired | `bot_sort` | Adds confidence-fused IoU and observation-gap velocity; this project does not provide ReID or camera-motion compensation |
| Hybrid-SORT inspired | `hybrid_sort` | Combines motion direction, confidence and bbox-height consistency |
| Deep OC-SORT inspired | `deep_oc_sort` | Uses the local motion/weak-cue fallback because ReID embeddings are not part of YOLO TXT input |
| C-BIoU inspired | `cbiou` | Cascaded buffered-IoU matching for irregular motion and temporarily non-overlapping boxes |
| SparseTrack inspired | `sparse_track` | Groups boxes by pseudo-depth and matches near-to-far; current project default for crowded scenes |

These are local bbox-only adaptations, not full reproductions of methods that depend on appearance embeddings, camera calibration or detector internals.

## Important Parameters

| Field | Purpose |
| --- | --- |
| `iou_match` | Primary association threshold |
| `max_missed` | Frames a track may remain unmatched |
| `class_agnostic` | Allow matches across class IDs; normally keep `false` |
| `score_high`, `score_low` | High/low detection bands used by two-stage methods |
| `new_track_score` | Minimum score for a new trajectory |
| `low_iou_match`, `recover_iou_match` | Relaxed recovery thresholds |
| `velocity_weight` | Motion direction contribution |
| `height_weight` | Bbox-height consistency contribution |
| `cbiou_small_buffer`, `cbiou_large_buffer` | C-BIoU expansion ratios |
| `sparse_depth_bins` | Number of pseudo-depth groups |
| `sparse_cross_depth_iou` | High-IoU fallback across neighboring depth groups |

Input without a sixth confidence field is treated as score `1.0`; confidence-based recovery then has little practical effect.

## Example

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
