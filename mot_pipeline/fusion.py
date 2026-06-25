from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from mot_pipeline.models import BBox, FinalTrack, Tracklet
from mot_pipeline.utils.bbox import bbox_center, clip_bbox, iou, mean


def track_overlap_stats(a: Tracklet, b: Tracklet) -> Optional[Dict[str, float]]:
    # 计算两个前后向轨迹片段在重叠时间段内的一致性，用于后续融合判定。
    overlap_frames = sorted(set(a.frames) & set(b.frames))
    if len(overlap_frames) < 3:
        return None
    frame_ious = [iou(a.frames[f], b.frames[f]) for f in overlap_frames]
    mean_iou = mean(frame_ious)
    overlap_ratio = len(overlap_frames) / float(min(a.length, b.length))
    start_center_a = bbox_center(a.frames[overlap_frames[0]])
    start_center_b = bbox_center(b.frames[overlap_frames[0]])
    end_center_a = bbox_center(a.frames[overlap_frames[-1]])
    end_center_b = bbox_center(b.frames[overlap_frames[-1]])
    center_gap = mean(
        [
            math.hypot(start_center_a[0] - start_center_b[0], start_center_a[1] - start_center_b[1]),
            math.hypot(end_center_a[0] - end_center_b[0], end_center_a[1] - end_center_b[1]),
        ]
    )
    avg_diag = mean(
        [
            math.hypot(a.frames[f][2] - a.frames[f][0], a.frames[f][3] - a.frames[f][1])
            for f in overlap_frames
        ]
        + [
            math.hypot(b.frames[f][2] - b.frames[f][0], b.frames[f][3] - b.frames[f][1])
            for f in overlap_frames
        ]
    )
    normalized_gap = center_gap / max(1.0, avg_diag)
    return {
        "mean_iou": mean_iou,
        "overlap_ratio": overlap_ratio,
        "overlap_count": float(len(overlap_frames)),
        "normalized_gap": normalized_gap,
        "score": mean_iou * overlap_ratio,
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def fuse_tracklets(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    # 只在前向和后向轨迹间做互选式匹配，避免一次性全连接合并带来的错误串联。
    all_tracklets = list(forward_tracks) + list(backward_tracks)
    uf = UnionFind(len(all_tracklets))
    if not forward_tracks or not backward_tracks:
        components: Dict[int, List[Tracklet]] = defaultdict(list)
        for idx, tracklet in enumerate(all_tracklets):
            components[uf.find(idx)].append(tracklet)
        return list(components.values())

    offset = len(forward_tracks)
    forward_best: Dict[int, Tuple[float, int]] = {}
    backward_best: Dict[int, Tuple[float, int]] = {}

    for f_idx, f_track in enumerate(forward_tracks):
        for b_idx, b_track in enumerate(backward_tracks):
            stats = track_overlap_stats(f_track, b_track)
            if stats is None:
                continue
            if stats["mean_iou"] < iou_fuse:
                continue
            if stats["normalized_gap"] > 1.5:
                continue
            score = stats["score"]
            if score > forward_best.get(f_idx, (-1.0, -1))[0]:
                forward_best[f_idx] = (score, b_idx)
            if score > backward_best.get(b_idx, (-1.0, -1))[0]:
                backward_best[b_idx] = (score, f_idx)

    for f_idx, (_, b_idx) in forward_best.items():
        back_match = backward_best.get(b_idx)
        if back_match is None or back_match[1] != f_idx:
            continue
        uf.union(f_idx, offset + b_idx)

    components: Dict[int, List[Tracklet]] = defaultdict(list)
    for idx, tracklet in enumerate(all_tracklets):
        components[uf.find(idx)].append(tracklet)
    return list(components.values())


def _as_components(tracklets: Sequence[Tracklet]) -> List[List[Tracklet]]:
    return [[tracklet] for tracklet in tracklets]


def _compatible_pair_candidates(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[Tuple[float, int, int, Dict[str, float]]]:
    candidates: List[Tuple[float, int, int, Dict[str, float]]] = []
    for f_idx, f_track in enumerate(forward_tracks):
        for b_idx, b_track in enumerate(backward_tracks):
            stats = track_overlap_stats(f_track, b_track)
            if stats is None:
                continue
            if stats["mean_iou"] < iou_fuse:
                continue
            if stats["normalized_gap"] > 1.5:
                continue
            candidates.append((stats["score"], f_idx, b_idx, stats))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _mutual_best_pairs(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> Tuple[List[Tuple[int, int]], List[Tuple[float, int, int, Dict[str, float]]]]:
    candidates = _compatible_pair_candidates(forward_tracks, backward_tracks, iou_fuse)
    forward_best: Dict[int, Tuple[float, int]] = {}
    backward_best: Dict[int, Tuple[float, int]] = {}
    for score, f_idx, b_idx, _ in candidates:
        if score > forward_best.get(f_idx, (-1.0, -1))[0]:
            forward_best[f_idx] = (score, b_idx)
        if score > backward_best.get(b_idx, (-1.0, -1))[0]:
            backward_best[b_idx] = (score, f_idx)

    pairs = []
    for f_idx, (_, b_idx) in forward_best.items():
        back_match = backward_best.get(b_idx)
        if back_match is None or back_match[1] != f_idx:
            continue
        pairs.append((f_idx, b_idx))
    return pairs, candidates


def fuse_forward_only(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    return _as_components(forward_tracks)


def fuse_backward_only(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    return _as_components(backward_tracks)


def fuse_tracklets_forward_primary(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    if not forward_tracks:
        return _as_components(backward_tracks)
    pairs, _ = _mutual_best_pairs(forward_tracks, backward_tracks, iou_fuse)
    backward_by_forward = {f_idx: b_idx for f_idx, b_idx in pairs}

    components: List[List[Tracklet]] = []
    for f_idx, f_track in enumerate(forward_tracks):
        component = [f_track]
        b_idx = backward_by_forward.get(f_idx)
        if b_idx is not None:
            component.append(backward_tracks[b_idx])
        components.append(component)
    return components


def fuse_tracklets_forward_unique(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    if not forward_tracks:
        return _as_components(backward_tracks)
    pairs, candidates = _mutual_best_pairs(forward_tracks, backward_tracks, iou_fuse)
    matched_backward = {b_idx for _, b_idx in pairs}
    duplicate_backward = {b_idx for _, _, b_idx, _ in candidates}
    backward_by_forward = {f_idx: b_idx for f_idx, b_idx in pairs}

    components: List[List[Tracklet]] = []
    for f_idx, f_track in enumerate(forward_tracks):
        component = [f_track]
        b_idx = backward_by_forward.get(f_idx)
        if b_idx is not None:
            component.append(backward_tracks[b_idx])
        components.append(component)

    for b_idx, b_track in enumerate(backward_tracks):
        if b_idx in matched_backward or b_idx in duplicate_backward:
            continue
        components.append([b_track])
    return components


def fuse_tracklets_all_pairs(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    all_tracklets = list(forward_tracks) + list(backward_tracks)
    uf = UnionFind(len(all_tracklets))
    offset = len(forward_tracks)
    for _, f_idx, b_idx, _ in _compatible_pair_candidates(forward_tracks, backward_tracks, iou_fuse):
        uf.union(f_idx, offset + b_idx)

    components: Dict[int, List[Tracklet]] = defaultdict(list)
    for idx, tracklet in enumerate(all_tracklets):
        components[uf.find(idx)].append(tracklet)
    return list(components.values())


def _component_score(component: Sequence[Tracklet]) -> float:
    if not component:
        return 0.0
    length = max(tracklet.length for tracklet in component)
    quality = max(tracklet.quality for tracklet in component)
    support = len(component)
    return length + quality + 0.25 * support


def _components_overlap(
    a: Sequence[Tracklet],
    b: Sequence[Tracklet],
    iou_fuse: float,
) -> bool:
    for a_track in a:
        for b_track in b:
            stats = track_overlap_stats(a_track, b_track)
            if stats is None:
                continue
            if stats["mean_iou"] < iou_fuse:
                continue
            if stats["overlap_ratio"] < 0.6:
                continue
            if stats["normalized_gap"] > 1.5:
                continue
            return True
    return False


def fuse_tracklets_nms(
    forward_tracks: Sequence[Tracklet],
    backward_tracks: Sequence[Tracklet],
    iou_fuse: float,
) -> List[List[Tracklet]]:
    components = fuse_tracklets(forward_tracks, backward_tracks, iou_fuse)
    ordered = sorted(components, key=_component_score, reverse=True)
    kept: List[List[Tracklet]] = []
    for component in ordered:
        if any(_components_overlap(component, kept_component, iou_fuse) for kept_component in kept):
            continue
        kept.append(component)
    kept.sort(key=lambda component: min(tracklet.start_frame for tracklet in component))
    return kept


def merge_component(
    tracklets: Sequence[Tracklet],
    image_w: int,
    image_h: int,
) -> Tuple[Dict[int, BBox], Dict[int, int], int]:
    # 同一融合组件内，按帧整合多个候选框；优先保留质量更高且空间一致的结果。
    frame_to_entries: Dict[int, List[Tuple[Tracklet, BBox]]] = defaultdict(list)
    video_frames: Dict[int, int] = {}
    class_votes = Counter()

    for tracklet in tracklets:
        class_votes[tracklet.class_id] += tracklet.length
        for frame_id, bbox in tracklet.frames.items():
            frame_to_entries[frame_id].append((tracklet, bbox))
            video_frames[frame_id] = tracklet.video_frames[frame_id]

    merged_frames: Dict[int, BBox] = {}
    for frame_id, entries in frame_to_entries.items():
        if len(entries) == 1:
            merged_frames[frame_id] = list(entries[0][1])
            continue
        entries = sorted(entries, key=lambda item: item[0].quality, reverse=True)
        primary_track, primary_bbox = entries[0]
        blended = [list(primary_bbox)]
        for _, other_bbox in entries[1:]:
            if iou(primary_bbox, other_bbox) >= 0.5:
                blended.append(list(other_bbox))
                continue
            primary_center = bbox_center(primary_bbox)
            other_center = bbox_center(other_bbox)
            dist = math.hypot(primary_center[0] - other_center[0], primary_center[1] - other_center[1])
            diag = math.hypot(primary_bbox[2] - primary_bbox[0], primary_bbox[3] - primary_bbox[1])
            if dist <= max(5.0, 0.5 * diag):
                blended.append(list(other_bbox))
        averaged = [mean([bbox[i] for bbox in blended]) for i in range(4)]
        clipped = clip_bbox(averaged, image_w, image_h)
        merged_frames[frame_id] = clipped if clipped is not None else list(primary_bbox)

    majority_class = class_votes.most_common(1)[0][0]
    return merged_frames, video_frames, majority_class


def smooth_track(
    frames: Dict[int, BBox],
    image_w: int,
    image_h: int,
    window: int,
) -> Dict[int, BBox]:
    # 对最终轨迹做滑动平均，主要用于减小裁剪视频中的抖动感。
    if window <= 1 or len(frames) <= 1:
        return {k: list(v) for k, v in frames.items()}
    ordered_ids = sorted(frames)
    ordered_boxes = [frames[frame_id] for frame_id in ordered_ids]
    radius = max(0, window // 2)
    smoothed: Dict[int, BBox] = {}
    for idx, frame_id in enumerate(ordered_ids):
        start = max(0, idx - radius)
        end = min(len(ordered_boxes), idx + radius + 1)
        neighbors = ordered_boxes[start:end]
        averaged = [mean([bbox[i] for bbox in neighbors]) for i in range(4)]
        clipped = clip_bbox(averaged, image_w, image_h)
        smoothed[frame_id] = clipped if clipped is not None else list(frames[frame_id])
    return smoothed


def build_final_tracks(
    components: Sequence[Sequence[Tracklet]],
    image_w: int,
    image_h: int,
    min_track_len: int,
    smooth_window: int,
) -> List[FinalTrack]:
    # 将融合结果转成最终全局轨迹，并在这里完成短轨过滤与平滑。
    final_tracks: List[FinalTrack] = []
    next_global_id = 1
    for component in components:
        merged_frames, video_frames, class_id = merge_component(component, image_w, image_h)
        if len(merged_frames) < min_track_len:
            continue
        smoothed_frames = smooth_track(merged_frames, image_w, image_h, smooth_window)
        final_tracks.append(
            FinalTrack(
                track_id=next_global_id,
                class_id=class_id,
                frames=smoothed_frames,
                video_frames=video_frames,
            )
        )
        next_global_id += 1
    final_tracks.sort(key=lambda track: track.track_id)
    return final_tracks


FUSION_REGISTRY = {
    "bidirectional_iou": fuse_tracklets,
    "forward_only": fuse_forward_only,
    "backward_only": fuse_backward_only,
    "bidirectional_iou_forward_primary": fuse_tracklets_forward_primary,
    "bidirectional_iou_forward_unique": fuse_tracklets_forward_unique,
    "bidirectional_iou_all_pairs": fuse_tracklets_all_pairs,
    "bidirectional_iou_nms": fuse_tracklets_nms,
}
