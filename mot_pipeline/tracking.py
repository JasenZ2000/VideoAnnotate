from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from mot_pipeline.models import BBox, Detection, FrameDetections, Tracklet
from mot_pipeline.utils.bbox import bbox_center, bbox_to_xywh, iou, mean, xywh_to_bbox


TrackerConfig = Optional[Mapping[str, Any]]
PairScoreFn = Callable[["LocalTrack", Detection], Optional[float]]


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
        if self.use_kalman and self.kf is not None:
            prediction = self.kf.predict()
            cx = float(prediction[0, 0])
            cy = float(prediction[1, 0])
            w = max(1.0, float(prediction[2, 0]))
            h = max(1.0, float(prediction[3, 0]))
        else:
            cx = self.last_prediction[0] + self.velocity[0]
            cy = self.last_prediction[1] + self.velocity[1]
            w = max(1.0, self.last_prediction[2] + self.velocity[2])
            h = max(1.0, self.last_prediction[3] + self.velocity[3])
        self.last_prediction = [cx, cy, w, h]
        return xywh_to_bbox(cx, cy, w, h)

    def update(self, bbox: Sequence[float]) -> BBox:
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
        self.last_prediction = [cx, cy, w, h]
        return xywh_to_bbox(cx, cy, w, h)

    def set_velocity_from_delta(
        self,
        previous_bbox: Sequence[float],
        current_bbox: Sequence[float],
        frame_gap: int,
    ) -> None:
        gap = max(1, abs(int(frame_gap)))
        prev = bbox_to_xywh(previous_bbox)
        cur = bbox_to_xywh(current_bbox)
        self.velocity = [(cur[idx] - prev[idx]) / float(gap) for idx in range(4)]
        if self.use_kalman and self.kf is not None:
            for idx, value in enumerate(self.velocity, start=4):
                self.kf.statePost[idx, 0] = float(value)


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
        self.last_score = detection.score
        self.predicted_bbox = list(detection.bbox)
        self.frames: Dict[int, BBox] = {detection.frame_id: list(detection.bbox)}
        self.observations: Dict[int, BBox] = {detection.frame_id: list(detection.bbox)}
        self.video_frames: Dict[int, int] = {detection.frame_id: detection.video_frame_idx}
        self.frame_classes: Dict[int, int] = {detection.frame_id: detection.class_id}

    @property
    def last_observation(self) -> BBox:
        return self.observations[self.last_frame_id]

    def previous_observation(self, max_history: int = 5) -> Optional[Tuple[int, BBox]]:
        ordered = sorted(self.observations)
        if len(ordered) < 2:
            return None
        current_pos = ordered.index(self.last_frame_id)
        start_pos = max(0, current_pos - max(1, max_history))
        for frame_id in reversed(ordered[start_pos:current_pos]):
            return frame_id, self.observations[frame_id]
        return None

    def observation_velocity(self, max_history: int = 5) -> Optional[Tuple[float, float]]:
        previous = self.previous_observation(max_history=max_history)
        if previous is None:
            return None
        previous_frame_id, previous_bbox = previous
        current_center = bbox_center(self.last_observation)
        previous_center = bbox_center(previous_bbox)
        frame_gap = max(1, abs(self.last_frame_id - previous_frame_id))
        vx = (current_center[0] - previous_center[0]) / float(frame_gap)
        vy = (current_center[1] - previous_center[1]) / float(frame_gap)
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            return None
        return vx / norm, vy / norm

    def predict(self) -> BBox:
        self.predicted_bbox = self.model.predict()
        self.age += 1
        return self.predicted_bbox

    def update(self, detection: Detection, use_observation_gap: bool = False) -> None:
        previous_bbox = self.last_observation
        previous_video_idx = self.last_video_idx
        smoothed_bbox = self.model.update(detection.bbox)
        if use_observation_gap:
            frame_gap = abs(detection.video_frame_idx - previous_video_idx)
            self.model.set_velocity_from_delta(previous_bbox, detection.bbox, frame_gap)
        self.class_id = detection.class_id
        self.hits += 1
        self.misses = 0
        self.last_frame_id = detection.frame_id
        self.last_video_idx = detection.video_frame_idx
        self.last_score = detection.score
        self.frames[detection.frame_id] = smoothed_bbox
        self.observations[detection.frame_id] = list(detection.bbox)
        self.video_frames[detection.frame_id] = detection.video_frame_idx
        self.frame_classes[detection.frame_id] = detection.class_id
        self.predicted_bbox = smoothed_bbox

    def mark_missed(self) -> None:
        self.misses += 1

    def to_tracklet(self, direction: str) -> Tracklet:
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


def _cfg(config: TrackerConfig) -> Mapping[str, Any]:
    return config or {}


def _float_config(config: TrackerConfig, key: str, default: float) -> float:
    try:
        return float(_cfg(config).get(key, default))
    except (TypeError, ValueError):
        return default


def _bool_config(config: TrackerConfig, key: str, default: bool) -> bool:
    value = _cfg(config).get(key, default)
    return bool(value)


def _class_compatible(track: LocalTrack, detection: Detection, config: TrackerConfig) -> bool:
    return _bool_config(config, "class_agnostic", False) or track.class_id == detection.class_id


def _unit_vector(start: Sequence[float], end: Sequence[float]) -> Optional[Tuple[float, float]]:
    sx, sy = bbox_center(start)
    ex, ey = bbox_center(end)
    dx = ex - sx
    dy = ey - sy
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return dx / norm, dy / norm


def _direction_affinity(track: LocalTrack, detection: Detection) -> float:
    track_velocity = track.observation_velocity()
    det_direction = _unit_vector(track.last_observation, detection.bbox)
    if track_velocity is None or det_direction is None:
        return 0.5
    dot = max(-1.0, min(1.0, track_velocity[0] * det_direction[0] + track_velocity[1] * det_direction[1]))
    return (dot + 1.0) / 2.0


def _height_affinity(track: LocalTrack, detection: Detection) -> float:
    _, _, _, track_h = bbox_to_xywh(track.last_observation)
    _, _, _, det_h = bbox_to_xywh(detection.bbox)
    larger = max(track_h, det_h, 1.0)
    smaller = max(1.0, min(track_h, det_h))
    return smaller / larger


def _expand_bbox(bbox: Sequence[float], ratio: float) -> BBox:
    cx, cy, w, h = bbox_to_xywh(bbox)
    scale = max(0.0, float(ratio))
    return xywh_to_bbox(cx, cy, w * (1.0 + scale), h * (1.0 + scale))


def _buffered_iou(b1: Sequence[float], b2: Sequence[float], buffer_ratio: float) -> float:
    return iou(_expand_bbox(b1, buffer_ratio), _expand_bbox(b2, buffer_ratio))


def _pseudo_depth(bbox: Sequence[float], height_weight: float) -> float:
    _, _, _, h = bbox_to_xywh(bbox)
    bottom_y = float(bbox[3])
    return bottom_y + max(0.0, height_weight) * h


def _depth_bin(depth: float, min_depth: float, max_depth: float, bins: int) -> int:
    if bins <= 1 or max_depth <= min_depth:
        return 0
    rel = (max_depth - depth) / (max_depth - min_depth)
    return max(0, min(bins - 1, int(math.floor(rel * bins))))


def _detection_pairs(frame: FrameDetections) -> List[Tuple[int, Detection]]:
    return list(enumerate(frame.detections))


def _match_subset(
    tracks: Sequence[LocalTrack],
    track_indices: Sequence[int],
    detection_pairs: Sequence[Tuple[int, Detection]],
    pair_score: PairScoreFn,
) -> List[Tuple[int, int, float]]:
    scored_pairs: List[Tuple[float, int, int]] = []
    for track_idx in track_indices:
        track = tracks[track_idx]
        for det_idx, detection in detection_pairs:
            score = pair_score(track, detection)
            if score is not None:
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


def _iou_pair_score(
    min_iou: float,
    config: TrackerConfig,
    fuse_score: bool = False,
) -> PairScoreFn:
    def score(track: LocalTrack, detection: Detection) -> Optional[float]:
        if not _class_compatible(track, detection, config):
            return None
        overlap = iou(track.predicted_bbox, detection.bbox)
        if overlap < min_iou:
            return None
        if fuse_score:
            return overlap * (0.5 + 0.5 * detection.score)
        return overlap

    return score


def _oc_pair_score(min_iou: float, config: TrackerConfig) -> PairScoreFn:
    velocity_weight = _float_config(config, "velocity_weight", 0.20)
    recover_iou = _float_config(config, "recover_iou_match", max(0.01, min_iou * 0.25))
    min_direction = _float_config(config, "velocity_min_score", 0.60)

    def score(track: LocalTrack, detection: Detection) -> Optional[float]:
        if not _class_compatible(track, detection, config):
            return None
        predicted_iou = iou(track.predicted_bbox, detection.bbox)
        observed_iou = iou(track.last_observation, detection.bbox)
        overlap = max(predicted_iou, observed_iou)
        direction_score = _direction_affinity(track, detection)
        recovered = overlap >= recover_iou and direction_score >= min_direction
        if overlap < min_iou and not recovered:
            return None
        miss_penalty = 0.01 * track.misses
        return overlap + velocity_weight * direction_score - miss_penalty

    return score


def _hybrid_pair_score(min_iou: float, config: TrackerConfig) -> PairScoreFn:
    velocity_weight = _float_config(config, "velocity_weight", 0.18)
    height_weight = _float_config(config, "height_weight", 0.08)
    score_weight = _float_config(config, "score_weight", 0.06)
    recover_iou = _float_config(config, "recover_iou_match", max(0.01, min_iou * 0.25))
    min_direction = _float_config(config, "velocity_min_score", 0.58)

    def score(track: LocalTrack, detection: Detection) -> Optional[float]:
        if not _class_compatible(track, detection, config):
            return None
        predicted_iou = iou(track.predicted_bbox, detection.bbox)
        observed_iou = iou(track.last_observation, detection.bbox)
        overlap = max(predicted_iou, observed_iou)
        direction_score = _direction_affinity(track, detection)
        height_score = _height_affinity(track, detection)
        recovered = overlap >= recover_iou and direction_score >= min_direction
        if overlap < min_iou and not recovered:
            return None
        return (
            overlap
            + velocity_weight * direction_score
            + height_weight * height_score
            + score_weight * detection.score
            - 0.01 * track.misses
        )

    return score


def _cbiou_pair_score(
    min_iou: float,
    buffer_ratio: float,
    config: TrackerConfig,
) -> PairScoreFn:
    def score(track: LocalTrack, detection: Detection) -> Optional[float]:
        if not _class_compatible(track, detection, config):
            return None
        overlap = _buffered_iou(track.predicted_bbox, detection.bbox, buffer_ratio)
        if overlap < min_iou:
            return None
        return overlap * (0.5 + 0.5 * detection.score)

    return score


def _sparse_depth_pair_score(
    min_iou: float,
    config: TrackerConfig,
    min_depth: float,
    max_depth: float,
) -> PairScoreFn:
    height_weight = _float_config(config, "sparse_height_weight", 0.5)
    depth_weight = _float_config(config, "sparse_depth_weight", 0.15)
    score_weight = _float_config(config, "score_weight", 0.06)
    depth_span = max(1.0, max_depth - min_depth)

    def score(track: LocalTrack, detection: Detection) -> Optional[float]:
        if not _class_compatible(track, detection, config):
            return None
        overlap = iou(track.predicted_bbox, detection.bbox)
        if overlap < min_iou:
            return None
        track_depth = _pseudo_depth(track.predicted_bbox, height_weight)
        det_depth = _pseudo_depth(detection.bbox, height_weight)
        depth_affinity = 1.0 - min(1.0, abs(track_depth - det_depth) / depth_span)
        return overlap + depth_weight * depth_affinity + score_weight * detection.score

    return score


def greedy_match(
    tracks: Sequence[LocalTrack],
    detections: Sequence[Detection],
    min_iou: float,
) -> List[Tuple[int, int, float]]:
    def legacy_score(track: LocalTrack, detection: Detection) -> Optional[float]:
        overlap = iou(track.predicted_bbox, detection.bbox)
        if overlap < min_iou:
            return None
        return overlap

    det_pairs = list(enumerate(detections))
    track_indices = list(range(len(tracks)))
    return _match_subset(tracks, track_indices, det_pairs, legacy_score)


def _finalize_expired_tracks(
    active_tracks: Sequence[LocalTrack],
    max_missed: int,
    direction: str,
) -> Tuple[List[LocalTrack], List[Tracklet]]:
    survivors: List[LocalTrack] = []
    finalized: List[Tracklet] = []
    for track in active_tracks:
        if track.misses > max_missed:
            finalized.append(track.to_tracklet(direction))
        else:
            survivors.append(track)
    return survivors, finalized


def run_tracking_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
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

        active_tracks, expired = _finalize_expired_tracks(active_tracks, max_missed, direction)
        finalized.extend(expired)

        for det_idx, detection in enumerate(frame.detections):
            if det_idx in matched_det_indices:
                continue
            active_tracks.append(LocalTrack(next_track_id, detection, use_kalman))
            next_track_id += 1

    finalized.extend(track.to_tracklet(direction) for track in active_tracks)
    return finalized


def _run_single_stage_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig,
    pair_score: PairScoreFn,
    use_observation_gap: bool,
) -> List[Tracklet]:
    ordered_frames = list(frames if direction == "forward" else reversed(frames))
    cfg = _cfg(tracking_config)
    score_low = float(cfg.get("score_low", 0.0))
    new_track_score = float(cfg.get("new_track_score", cfg.get("score_high", 0.5)))

    active_tracks: List[LocalTrack] = []
    finalized: List[Tracklet] = []
    next_track_id = 1

    for frame in ordered_frames:
        for track in active_tracks:
            track.predict()

        detections = [(idx, det) for idx, det in _detection_pairs(frame) if det.score >= score_low]
        matches = _match_subset(active_tracks, list(range(len(active_tracks))), detections, pair_score)
        matched_track_indices = {track_idx for track_idx, _, _ in matches}
        matched_det_indices = {det_idx for _, det_idx, _ in matches}

        for track_idx, det_idx, _ in matches:
            active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=use_observation_gap)

        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                track.mark_missed()

        active_tracks, expired = _finalize_expired_tracks(active_tracks, max_missed, direction)
        finalized.extend(expired)

        for det_idx, detection in detections:
            if det_idx in matched_det_indices or detection.score < new_track_score:
                continue
            active_tracks.append(LocalTrack(next_track_id, detection, use_kalman))
            next_track_id += 1

    finalized.extend(track.to_tracklet(direction) for track in active_tracks)
    return finalized


def _run_byte_style_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig,
    high_pair_score: PairScoreFn,
    low_pair_score: PairScoreFn,
    use_observation_gap: bool = False,
) -> List[Tracklet]:
    ordered_frames = list(frames if direction == "forward" else reversed(frames))
    cfg = _cfg(tracking_config)
    score_high = float(cfg.get("score_high", cfg.get("track_thresh", 0.5)))
    score_low = float(cfg.get("score_low", 0.1))
    new_track_score = float(cfg.get("new_track_score", cfg.get("new_track_thresh", score_high)))

    active_tracks: List[LocalTrack] = []
    finalized: List[Tracklet] = []
    next_track_id = 1

    for frame in ordered_frames:
        for track in active_tracks:
            track.predict()

        high_detections = []
        low_detections = []
        for det_idx, detection in _detection_pairs(frame):
            if detection.score >= score_high:
                high_detections.append((det_idx, detection))
            elif detection.score >= score_low:
                low_detections.append((det_idx, detection))

        track_indices = list(range(len(active_tracks)))
        high_matches = _match_subset(active_tracks, track_indices, high_detections, high_pair_score)
        matched_track_indices = {track_idx for track_idx, _, _ in high_matches}
        matched_det_indices = {det_idx for _, det_idx, _ in high_matches}

        for track_idx, det_idx, _ in high_matches:
            active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=use_observation_gap)

        unmatched_tracks = [idx for idx in track_indices if idx not in matched_track_indices]
        low_matches = _match_subset(active_tracks, unmatched_tracks, low_detections, low_pair_score)
        for track_idx, det_idx, _ in low_matches:
            active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=use_observation_gap)
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)

        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                track.mark_missed()

        active_tracks, expired = _finalize_expired_tracks(active_tracks, max_missed, direction)
        finalized.extend(expired)

        for det_idx, detection in high_detections:
            if det_idx in matched_det_indices or detection.score < new_track_score:
                continue
            active_tracks.append(LocalTrack(next_track_id, detection, use_kalman))
            next_track_id += 1

    finalized.extend(track.to_tracklet(direction) for track in active_tracks)
    return finalized


def _run_cbiou_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig,
) -> List[Tracklet]:
    ordered_frames = list(frames if direction == "forward" else reversed(frames))
    cfg = _cfg(tracking_config)
    score_low = float(cfg.get("score_low", 0.0))
    new_track_score = float(cfg.get("new_track_score", cfg.get("score_high", 0.5)))
    small_buffer = _float_config(tracking_config, "cbiou_small_buffer", 0.08)
    large_buffer = _float_config(tracking_config, "cbiou_large_buffer", 0.28)
    second_iou = _float_config(tracking_config, "cbiou_second_iou", max(0.05, min_iou * 0.5))

    active_tracks: List[LocalTrack] = []
    finalized: List[Tracklet] = []
    next_track_id = 1

    for frame in ordered_frames:
        for track in active_tracks:
            track.predict()

        detections = [(idx, det) for idx, det in _detection_pairs(frame) if det.score >= score_low]
        track_indices = list(range(len(active_tracks)))
        first_matches = _match_subset(
            active_tracks,
            track_indices,
            detections,
            _cbiou_pair_score(min_iou, small_buffer, tracking_config),
        )
        matched_track_indices = {track_idx for track_idx, _, _ in first_matches}
        matched_det_indices = {det_idx for _, det_idx, _ in first_matches}
        for track_idx, det_idx, _ in first_matches:
            active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=True)

        unmatched_tracks = [idx for idx in track_indices if idx not in matched_track_indices]
        unmatched_detections = [(idx, det) for idx, det in detections if idx not in matched_det_indices]
        second_matches = _match_subset(
            active_tracks,
            unmatched_tracks,
            unmatched_detections,
            _cbiou_pair_score(second_iou, large_buffer, tracking_config),
        )
        for track_idx, det_idx, _ in second_matches:
            active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=True)
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)

        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                track.mark_missed()

        active_tracks, expired = _finalize_expired_tracks(active_tracks, max_missed, direction)
        finalized.extend(expired)

        for det_idx, detection in detections:
            if det_idx in matched_det_indices or detection.score < new_track_score:
                continue
            active_tracks.append(LocalTrack(next_track_id, detection, use_kalman))
            next_track_id += 1

    finalized.extend(track.to_tracklet(direction) for track in active_tracks)
    return finalized


def _run_sparse_depth_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig,
) -> List[Tracklet]:
    ordered_frames = list(frames if direction == "forward" else reversed(frames))
    cfg = _cfg(tracking_config)
    score_low = float(cfg.get("score_low", 0.0))
    new_track_score = float(cfg.get("new_track_score", cfg.get("score_high", 0.5)))
    bins = max(1, int(cfg.get("sparse_depth_bins", 4)))
    neighbor_bins = max(0, int(cfg.get("sparse_neighbor_bins", 1)))
    cross_depth_iou = _float_config(tracking_config, "sparse_cross_depth_iou", 0.75)
    height_weight = _float_config(tracking_config, "sparse_height_weight", 0.5)

    active_tracks: List[LocalTrack] = []
    finalized: List[Tracklet] = []
    next_track_id = 1

    for frame in ordered_frames:
        for track in active_tracks:
            track.predict()

        detections = [(idx, det) for idx, det in _detection_pairs(frame) if det.score >= score_low]
        track_indices = list(range(len(active_tracks)))
        matched_track_indices = set()
        matched_det_indices = set()

        depth_values = [
            _pseudo_depth(active_tracks[idx].predicted_bbox, height_weight) for idx in track_indices
        ] + [
            _pseudo_depth(det.bbox, height_weight) for _, det in detections
        ]
        if depth_values:
            min_depth = min(depth_values)
            max_depth = max(depth_values)
        else:
            min_depth = 0.0
            max_depth = 1.0

        track_bins = {
            idx: _depth_bin(
                _pseudo_depth(active_tracks[idx].predicted_bbox, height_weight),
                min_depth,
                max_depth,
                bins,
            )
            for idx in track_indices
        }
        det_bins = {
            det_idx: _depth_bin(_pseudo_depth(det.bbox, height_weight), min_depth, max_depth, bins)
            for det_idx, det in detections
        }
        pair_score = _sparse_depth_pair_score(min_iou, tracking_config, min_depth, max_depth)

        for depth_level in range(bins):
            unmatched_tracks = [
                idx
                for idx in track_indices
                if idx not in matched_track_indices
                and abs(track_bins.get(idx, depth_level) - depth_level) <= neighbor_bins
            ]
            level_detections = [
                (idx, det)
                for idx, det in detections
                if idx not in matched_det_indices
                and det_bins.get(idx, depth_level) == depth_level
            ]
            if not unmatched_tracks or not level_detections:
                continue
            matches = _match_subset(active_tracks, unmatched_tracks, level_detections, pair_score)
            for track_idx, det_idx, _ in matches:
                active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=True)
                matched_track_indices.add(track_idx)
                matched_det_indices.add(det_idx)

        if cross_depth_iou > 0.0:
            unmatched_tracks = [idx for idx in track_indices if idx not in matched_track_indices]
            unmatched_detections = [(idx, det) for idx, det in detections if idx not in matched_det_indices]
            fallback_matches = _match_subset(
                active_tracks,
                unmatched_tracks,
                unmatched_detections,
                _iou_pair_score(cross_depth_iou, tracking_config),
            )
            for track_idx, det_idx, _ in fallback_matches:
                active_tracks[track_idx].update(frame.detections[det_idx], use_observation_gap=True)
                matched_track_indices.add(track_idx)
                matched_det_indices.add(det_idx)

        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                track.mark_missed()

        active_tracks, expired = _finalize_expired_tracks(active_tracks, max_missed, direction)
        finalized.extend(expired)

        for det_idx, detection in detections:
            if det_idx in matched_det_indices or detection.score < new_track_score:
                continue
            active_tracks.append(LocalTrack(next_track_id, detection, use_kalman))
            next_track_id += 1

    finalized.extend(track.to_tracklet(direction) for track in active_tracks)
    return finalized


def run_bytetrack_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    cfg = _cfg(tracking_config)
    first_iou = float(cfg.get("match_iou", min_iou))
    second_iou = float(cfg.get("low_iou_match", max(0.1, min_iou * 0.5)))
    return _run_byte_style_pass(
        frames=frames,
        direction=direction,
        min_iou=min_iou,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=tracking_config,
        high_pair_score=_iou_pair_score(first_iou, tracking_config),
        low_pair_score=_iou_pair_score(second_iou, tracking_config),
    )


def run_oc_sort_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    return _run_single_stage_pass(
        frames=frames,
        direction=direction,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=tracking_config,
        pair_score=_oc_pair_score(min_iou, tracking_config),
        use_observation_gap=True,
    )


def run_bot_sort_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    cfg = _cfg(tracking_config)
    first_iou = float(cfg.get("match_iou", min_iou))
    second_iou = float(cfg.get("low_iou_match", max(0.1, min_iou * 0.5)))
    fuse_score = bool(cfg.get("fuse_score", True))
    return _run_byte_style_pass(
        frames=frames,
        direction=direction,
        min_iou=min_iou,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=tracking_config,
        high_pair_score=_iou_pair_score(first_iou, tracking_config, fuse_score=fuse_score),
        low_pair_score=_hybrid_pair_score(second_iou, tracking_config),
        use_observation_gap=True,
    )


def run_hybrid_sort_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    cfg = _cfg(tracking_config)
    first_iou = float(cfg.get("match_iou", min_iou))
    second_iou = float(cfg.get("low_iou_match", max(0.1, min_iou * 0.5)))
    return _run_byte_style_pass(
        frames=frames,
        direction=direction,
        min_iou=min_iou,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=tracking_config,
        high_pair_score=_hybrid_pair_score(first_iou, tracking_config),
        low_pair_score=_hybrid_pair_score(second_iou, tracking_config),
        use_observation_gap=True,
    )


def run_deep_oc_sort_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    cfg = dict(_cfg(tracking_config))
    cfg.setdefault("velocity_weight", 0.24)
    cfg.setdefault("height_weight", 0.06)
    cfg.setdefault("score_weight", 0.04)
    return _run_single_stage_pass(
        frames=frames,
        direction=direction,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=cfg,
        pair_score=_hybrid_pair_score(min_iou, cfg),
        use_observation_gap=True,
    )


def run_cbiou_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    return _run_cbiou_pass(
        frames=frames,
        direction=direction,
        min_iou=min_iou,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=tracking_config,
    )


def run_sparse_track_pass(
    frames: Sequence[FrameDetections],
    direction: str,
    min_iou: float,
    max_missed: int,
    use_kalman: bool,
    tracking_config: TrackerConfig = None,
) -> List[Tracklet]:
    return _run_sparse_depth_pass(
        frames=frames,
        direction=direction,
        min_iou=min_iou,
        max_missed=max_missed,
        use_kalman=use_kalman,
        tracking_config=tracking_config,
    )


TRACKER_REGISTRY = {
    "iou_kalman": run_tracking_pass,
    "sort": run_tracking_pass,
    "bytetrack": run_bytetrack_pass,
    "oc_sort": run_oc_sort_pass,
    "bot_sort": run_bot_sort_pass,
    "hybrid_sort": run_hybrid_sort_pass,
    "deep_oc_sort": run_deep_oc_sort_pass,
    "cbiou": run_cbiou_pass,
    "sparse_track": run_sparse_track_pass,
}
