from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from mot_pipeline.models import BBox


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[BBox]:
    # 将框裁剪到图像边界内，若裁剪后退化为无效框则返回 None。
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, image_w: int, image_h: int
) -> Optional[BBox]:
    # YOLO 标注是归一化中心点格式，这里统一转成像素级 xyxy，便于后续跟踪与裁剪。
    px = cx * image_w
    py = cy * image_h
    pw = w * image_w
    ph = h * image_h
    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0
    return clip_bbox([x1, y1, x2, y2], image_w, image_h)


def bbox_to_xywh(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def xywh_to_bbox(cx: float, cy: float, w: float, h: float) -> BBox:
    half_w = max(0.0, w) / 2.0
    half_h = max(0.0, h) / 2.0
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    # IOU 是跟踪匹配和轨迹融合的核心相似度度量。
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = bbox_area(b1) + bbox_area(b2) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def round_bbox_int(bbox: Sequence[float], width: int, height: int) -> List[int]:
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return [0, 0, 0, 0]
    return [int(round(v)) for v in clipped]
