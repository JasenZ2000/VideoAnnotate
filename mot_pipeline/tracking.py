from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import cv2

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from mot_pipeline.models import BBox, Detection, FrameDetections, Tracklet
from mot_pipeline.utils.bbox import bbox_center, bbox_to_xywh, iou, mean, xywh_to_bbox


class MotionModel:
    """Kalman-backed motion model with a constant-velocity fallback."""

    def __init__(self, bbox: Sequence[float], use_kalman: bool) -> None:
        self.use_kalman = bool(use_kalman and np is not None)
        cx, cy, w, h = bbox_to_xywh(bbox)
        self.last_measurement = [cx, cy, w, h]
        self.velocity = [0.0, 0.0, 0.0, 0.0]
        self.last_prediction = [cx, cy, w, h]
        self.kf = None
        if self.use_kalman:
            self.kf = cv2.KalmanFilter(8, 4)
            self.kf.transitionMatrix = np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            self.kf.measurementMatrix = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
            self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
            self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
            self.kf.errorCovPost = np.eye(8, dtype=np.float32)
            self.kf.statePost = np.array(
                [[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32
            )

    def predict(self) -> BBox:
        # 先预测当前帧目标位置；若未启用 Kalman，则退化为简单速度外推。
        if self.use_kalman and self.kf is not None:
            prediction = self.kf.predict()
            cx = float(prediction[0, 0])
            cy = float(prediction[1, 0])
            w = max(1.0, float(prediction[2, 0]))
            h = max(1.0, float(prediction[3, 0]))
        else:
            cx = self.last_measurement[0] + self.velocity[0]
            cy = self.last_measurement[1] + self.velocity[1]
            w = max(1.0, self.last_measurement[2] + self.velocity[2])
            h = max(1.0, self.last_measurement[3] + self.velocity[3])
        self.last_prediction = [cx, cy, w, h]
        return xywh_to_bbox(cx, cy, w, h)

    def update(self, bbox: Sequence[float]) -> BBox:
        # 用当前检测更新状态，让后续匹配尽量跟随平滑后的轨迹而不是原始抖动框。
        cx, cy, w, h = bbox_to_xywh(bbox)
        if self.use_kalman and self.kf is not None:
            measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
            corrected = self.kf.correct(measurement)
            cx = float(corrected[0, 0])
            cy = float(corrected[1, 0])
            w = max(1.0, float(corrected[2, 0]))
            h = max(1.0, float(corrected[3, 0]))
            self.velocity = [
                float(corrected[4, 0]),
                float(corrected[5, 0]),
                float(corrected[6, 0]),
                float(corrected[7, 0]),
            ]
        else:
            self.velocity = [
                cx - self.last_measurement[0],
                cy - self.last_measurement[1],
                w - self.last_measurement[2],
                h - self.last_measurement[3],
            ]
        self.last_measurement = [cx, cy, w, h]
        return xywh_to_bbox(cx, cy, w, h)


class LocalTrack:
    def __init__(self, track_id: int, detection: Detection, use_kalman: bool) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.model = MotionModel(detection.bbox, use_kalman)
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.last_frame_id = detection.frame_id
        self.last_video_idx = detection.video_frame_idx
        self.predicted_bbox = list(detection.bbox)
        self.frames: Dict[int, BBox] = {detection.frame_id: list(detection.bbox)}
        self.video_frames: Dict[int, int] = {detection.frame_id: detection.video_frame_idx}
        self.frame_classes: Dict[int, int] = {detection.frame_id: detection.class_id}

    def predict(self) -> BBox:
        self.predicted_bbox = self.model.predict()
        self.age += 1
        return self.predicted_bbox

    def update(self, detection: Detection) -> None:
        smoothed_bbox = self.model.update(detection.bbox)
        self.class_id = detection.class_id
        self.hits += 1
        self.misses = 0
        self.last_frame_id = detection.frame_id
        self.last_video_idx = detection.video_frame_idx
        self.frames[detection.frame_id] = smoothed_bbox
        self.video_frames[detection.frame_id] = detection.video_frame_idx
        self.frame_classes[detection.frame_id] = detection.class_id
        self.predicted_bbox = smoothed_bbox

    def mark_missed(self) -> None:
        self.misses += 1

    def to_tracklet(self, direction: str) -> Tracklet:
        # 将在线跟踪状态冻结为可融合的轨迹片段，并计算简单稳定性评分。
        ordered_frames = [self.frames[k] for k in sorted(self.frames)]
        centers = [bbox_center(bbox) for bbox in ordered_frames]
        jumps = []
        for idx in range(1, len(centers)):
            dx = centers[idx][0] - centers[idx - 1][0]
            dy = centers[idx][1] - centers[idx - 1][1]
            jumps.append(math.hypot(dx, dy))
        jitter = mean(jumps)
        quality = float(len(self.frames)) / (1.0 + jitter)
        majority_class = Counter(self.frame_classes.values()).most_common(1)[0][0]
        return Tracklet(
            local_track_id=self.track_id,
            direction=direction,
            class_id=majority_class,
            frames={k: list(v) for k, v in self.frames.items()},
            video_frames=dict(self.video_frames),
            frame_classes=dict(self.frame_classes),
            hits=self.hits,
            misses=self.misses,
            jitter=jitter,
            quality=quality,
        )


def greedy_match(
    tracks: Sequence[LocalTrack],
    detections: Sequence[Detection],
    min_iou: float,
) -> List[Tuple[int, int, float]]:
    # 这里保持原始实现：按 IOU 从高到低贪心匹配，不额外引入 Hungarian 等复杂依赖。
    scored_pairs: List[Tuple[float, int, int]] = []
    for track_idx, track in enumerate(tracks):
        pred_bbox = track.predicted_bbox
        for det_idx, detection in enumerate(detections):
            score = iou(pred_bbox, detection.bbox)
            if score >= min_iou:
                scored_pairs.append((score, track_idx, det_idx))
    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    matched_tracks = set()
    matched_dets = set()
    matches: List[Tuple[int, int, float]] = []
    for score, track_idx, det_idx in scored_pairs:
        if track_idx in matched_tracks or det_idx in matched_dets:
            continue
        matched_tracks.add(track_idx)
        matched_dets.add(det_idx)
        matches.append((track_idx, det_idx, score))
    return matches


def run_tracking_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
) -> List[Tracklet]:
    # 单方向跟踪主循环：预测、匹配、更新、删失配轨迹，并为未匹配检测创建新轨迹。
    ordered_frames = list(frames if direction == "forward" else reversed(frames))
    active_tracks: List[LocalTrack] = []
    finalized: List[Tracklet] = []
    next_track_id = 1

    for frame in ordered_frames:
        for track in active_tracks:
            track.predict()

        matches = greedy_match(active_tracks, frame.detections, min_iou)
        matched_track_indices = {track_idx for track_idx, _, _ in matches}
        matched_det_indices = {det_idx for _, det_idx, _ in matches}

        for track_idx, det_idx, _ in matches:
            active_tracks[track_idx].update(frame.detections[det_idx])

        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                track.mark_missed()

        survivors: List[LocalTrack] = []
        for track in active_tracks:
            if track.misses > max_missed:
                finalized.append(track.to_tracklet(direction))
            else:
                survivors.append(track)
        active_tracks = survivors

        for det_idx, detection in enumerate(frame.detections):
            if det_idx in matched_det_indices:
                continue
            active_tracks.append(LocalTrack(next_track_id, detection, use_kalman))
            next_track_id += 1

    for track in active_tracks:
        finalized.append(track.to_tracklet(direction))
    return finalized


TRACKER_REGISTRY = {
    "iou_kalman": run_tracking_pass,
}
