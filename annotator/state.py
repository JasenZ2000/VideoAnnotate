from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Track:
    track_id: int
    class_id: int
    frames: Dict[int, List[float]] = field(default_factory=dict)  # frame_idx -> [x1,y1,x2,y2]

    @property
    def start_frame(self) -> Optional[int]:
        return min(self.frames) if self.frames else None

    @property
    def end_frame(self) -> Optional[int]:
        return max(self.frames) if self.frames else None


def track_color(track_id: int) -> Tuple[int, int, int]:
    return (
        int((37 * track_id + 80) % 256),
        int((97 * track_id + 140) % 256),
        int((17 * track_id + 220) % 256),
    )


def interpolate_bbox(a: List[float], b: List[float], t: float) -> List[float]:
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def bbox_size(bbox: List[float]) -> Tuple[float, float, float]:
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width, height, width * height


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def clip_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x1 = max(0.0, min(float(width), bbox[0]))
    y1 = max(0.0, min(float(height), bbox[1]))
    x2 = max(0.0, min(float(width), bbox[2]))
    y2 = max(0.0, min(float(height), bbox[3]))
    return [x1, y1, x2, y2]


class AnnotationState:
    def __init__(self) -> None:
        self.video_path: str = ""
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0
        self.tracks: Dict[int, Track] = {}
        self.next_track_id: int = 1

    def load_video_metadata(self, video_path: str) -> None:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.video_path = video_path
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def clear_tracks(self) -> None:
        self.tracks = {}
        self.next_track_id = 1

    def add_track(self, class_id: int = 0) -> int:
        tid = self.next_track_id
        self.tracks[tid] = Track(track_id=tid, class_id=class_id)
        self.next_track_id += 1
        return tid

    def delete_track(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)

    def set_frame(self, track_id: int, frame_idx: int, bbox: List[float]) -> None:
        if track_id not in self.tracks:
            return
        clamped = clip_bbox(bbox, self.width, self.height)
        self.tracks[track_id].frames[frame_idx] = clamped

    def delete_frame(self, track_id: int, frame_idx: int) -> None:
        if track_id not in self.tracks:
            return
        self.tracks[track_id].frames.pop(frame_idx, None)

    def delete_frames_after(self, track_id: int, frame_idx: int) -> int:
        """Delete all annotations strictly after frame_idx. Returns count deleted."""
        if track_id not in self.tracks:
            return 0
        to_remove = [k for k in self.tracks[track_id].frames if k > frame_idx]
        for k in to_remove:
            del self.tracks[track_id].frames[k]
        return len(to_remove)

    def delete_frames_before(self, track_id: int, frame_idx: int) -> int:
        """Delete all annotations strictly before frame_idx. Returns count deleted."""
        if track_id not in self.tracks:
            return 0
        to_remove = [k for k in self.tracks[track_id].frames if k < frame_idx]
        for k in to_remove:
            del self.tracks[track_id].frames[k]
        return len(to_remove)

    def delete_frames_between(self, track_id: int, frame_a: int, frame_b: int) -> int:
        """Delete all annotations between frame_a and frame_b, inclusive."""
        if track_id not in self.tracks:
            return 0
        start = min(frame_a, frame_b)
        end = max(frame_a, frame_b)
        to_remove = [k for k in self.tracks[track_id].frames if start <= k <= end]
        for k in to_remove:
            del self.tracks[track_id].frames[k]
        return len(to_remove)

    def merge_tracks(self, track_id_a: int, track_id_b: int) -> int:
        """Merge track B into track A. Returns the surviving track ID.
        For overlapping frames, track A's annotations take priority."""
        if track_id_a not in self.tracks or track_id_b not in self.tracks:
            raise ValueError("Track not found")
        if track_id_a == track_id_b:
            raise ValueError("Cannot merge a track with itself")
        track_a = self.tracks[track_id_a]
        track_b = self.tracks[track_id_b]
        for frame_idx, bbox in track_b.frames.items():
            if frame_idx not in track_a.frames:
                track_a.frames[frame_idx] = bbox
        del self.tracks[track_id_b]
        return track_id_a

    def set_class_id(self, track_id: int, class_id: int) -> None:
        if track_id not in self.tracks:
            return
        self.tracks[track_id].class_id = class_id

    def get_bbox_at_frame(self, track_id: int, frame_idx: int) -> Optional[List[float]]:
        if track_id not in self.tracks:
            return None
        return self.tracks[track_id].frames.get(frame_idx)

    def get_all_bboxes_at_frame(self, frame_idx: int) -> List[Tuple[int, List[float]]]:
        results = []
        for tid, track in self.tracks.items():
            bbox = track.frames.get(frame_idx)
            if bbox is not None:
                results.append((tid, bbox))
        return results

    def interpolate_range(self, track_id: int, frame_a: int, frame_b: int) -> int:
        """Linearly interpolate between frame_a and frame_b for the given track.
        Both frame_a and frame_b must already have annotations.
        Returns the number of frames filled."""
        if track_id not in self.tracks:
            return 0
        track = self.tracks[track_id]
        if frame_a not in track.frames or frame_b not in track.frames:
            return 0
        start = min(frame_a, frame_b)
        end = max(frame_a, frame_b)
        if end - start <= 1:
            return 0
        bbox_start = track.frames[start]
        bbox_end = track.frames[end]
        count = 0
        for idx in range(start + 1, end):
            t = (idx - start) / float(end - start)
            track.frames[idx] = interpolate_bbox(bbox_start, bbox_end, t)
            count += 1
        return count

    def interpolate_short_gaps(self, max_gap: int) -> Dict[str, Any]:
        """Fill every internal track gap whose missing-frame count is less than max_gap."""
        max_gap = max(0, int(max_gap))
        total_filled = 0
        filled_tracks = 0
        gaps_out = []

        for tid in sorted(self.tracks.keys()):
            track = self.tracks[tid]
            frame_ids = sorted(track.frames.keys())
            if len(frame_ids) < 2:
                continue

            track_filled = 0
            for prev_frame, next_frame in zip(frame_ids, frame_ids[1:]):
                missing = next_frame - prev_frame - 1
                if missing <= 0 or missing >= max_gap:
                    continue
                filled = self.interpolate_range(tid, prev_frame, next_frame)
                if filled <= 0:
                    continue
                track_filled += filled
                total_filled += filled
                gaps_out.append({
                    "track_id": tid,
                    "start": prev_frame,
                    "end": next_frame,
                    "missing": missing,
                    "filled": filled,
                })

            if track_filled > 0:
                filled_tracks += 1

        return {
            "filled_frames": total_filled,
            "filled_tracks": filled_tracks,
            "gaps": gaps_out,
        }

    def fix_bbox_spikes(
        self,
        track_id: int,
        area_ratio: float = 4.0,
        size_ratio: float = 3.0,
        history: int = 10,
        min_history: int = 3,
        max_run: int = 10,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Replace short bbox-size spikes with interpolation from surrounding frames."""
        if track_id not in self.tracks:
            return {"fixed_frames": 0, "intervals": []}
        track = self.tracks[track_id]
        frame_ids = sorted(track.frames.keys())
        if len(frame_ids) < min_history + 2:
            return {"fixed_frames": 0, "intervals": []}

        history = max(1, int(history))
        min_history = max(1, int(min_history))
        max_run = max(1, int(max_run))
        area_ratio = max(1.0, float(area_ratio))
        size_ratio = max(1.0, float(size_ratio))

        normal_sizes: List[Tuple[float, float, float]] = []
        anomalies: List[int] = []
        for frame_idx in frame_ids:
            bbox = track.frames[frame_idx]
            width, height, area = bbox_size(bbox)
            if area <= 0:
                anomalies.append(frame_idx)
                continue

            is_anomaly = False
            if len(normal_sizes) >= min_history:
                med_w = median(item[0] for item in normal_sizes)
                med_h = median(item[1] for item in normal_sizes)
                med_area = median(item[2] for item in normal_sizes)
                if med_area > 0 and area > med_area * area_ratio:
                    is_anomaly = True
                if med_w > 0 and width > med_w * size_ratio:
                    is_anomaly = True
                if med_h > 0 and height > med_h * size_ratio:
                    is_anomaly = True

            if is_anomaly:
                if (start_frame is None or frame_idx >= start_frame) and (end_frame is None or frame_idx <= end_frame):
                    anomalies.append(frame_idx)
            else:
                normal_sizes.append((width, height, area))
                if len(normal_sizes) > history:
                    normal_sizes.pop(0)

        if not anomalies:
            return {"fixed_frames": 0, "intervals": []}

        anomaly_set = set(anomalies)
        groups: List[List[int]] = []
        for frame_idx in anomalies:
            if not groups or frame_idx != groups[-1][-1] + 1:
                groups.append([frame_idx])
            else:
                groups[-1].append(frame_idx)

        fixed_frames = 0
        intervals = []
        for group in groups:
            if len(group) > max_run:
                continue
            start = group[0]
            end = group[-1]
            before = next((idx for idx in reversed(frame_ids) if idx < start and idx not in anomaly_set), None)
            after = next((idx for idx in frame_ids if idx > end and idx not in anomaly_set), None)
            if before is None or after is None or after <= before:
                continue

            bbox_before = track.frames[before]
            bbox_after = track.frames[after]
            for frame_idx in group:
                t = (frame_idx - before) / float(after - before)
                track.frames[frame_idx] = interpolate_bbox(bbox_before, bbox_after, t)
                fixed_frames += 1
            intervals.append({
                "start": start,
                "end": end,
                "anchor_before": before,
                "anchor_after": after,
                "frames": len(group),
            })

        return {"fixed_frames": fixed_frames, "intervals": intervals}

    def detect_area_anomaly_segments(
        self,
        high_area_ratio: float = 3.0,
        low_area_ratio: float = 0.25,
        frame_area_change_ratio: float = 2.5,
        frame_area_change_abs: float = 800.0,
        robust_z_threshold: float = 6.0,
        min_track_frames: int = 8,
        min_area: float = 1.0,
        max_gap: int = 1,
    ) -> Dict[str, Any]:
        """Find suspicious intervals from each track's full-run area distribution."""
        high_area_ratio = max(1.0, float(high_area_ratio))
        low_area_ratio = max(0.0, min(1.0, float(low_area_ratio)))
        frame_area_change_ratio = max(1.0, float(frame_area_change_ratio))
        frame_area_change_abs = max(0.0, float(frame_area_change_abs))
        robust_z_threshold = max(0.0, float(robust_z_threshold))
        min_track_frames = max(1, int(min_track_frames))
        min_area = max(0.0, float(min_area))
        max_gap = max(0, int(max_gap))

        tracks_out = []
        for tid in sorted(self.tracks.keys()):
            track = self.tracks[tid]
            frame_ids = sorted(track.frames.keys())
            area_rows = []
            for frame_idx in frame_ids:
                width, height, area = bbox_size(track.frames[frame_idx])
                area_rows.append({
                    "frame_idx": frame_idx,
                    "width": width,
                    "height": height,
                    "area": area,
                })

            positive_areas = [row["area"] for row in area_rows if row["area"] >= min_area]
            if len(positive_areas) < min_track_frames:
                tracks_out.append({
                    "track_id": tid,
                    "class_id": track.class_id,
                    "num_frames": len(frame_ids),
                    "median_area": 0.0,
                    "q1_area": 0.0,
                    "q3_area": 0.0,
                    "mad_area": 0.0,
                    "segments": [],
                })
                continue

            median_area = median(positive_areas)
            q1_area = percentile(positive_areas, 0.25)
            q3_area = percentile(positive_areas, 0.75)
            mad_area = median([abs(area - median_area) for area in positive_areas])
            suspicious = []
            previous_positive_area = None

            for row in area_rows:
                area = row["area"]
                reasons = []
                area_ratio = area / median_area if median_area > 0 else 0.0
                frame_change_ratio = 1.0
                frame_change_abs = 0.0
                robust_z = 0.0
                if mad_area > 0:
                    robust_z = 0.6745 * (area - median_area) / mad_area

                if previous_positive_area is not None and area >= min_area:
                    smaller = max(min(previous_positive_area, area), min_area)
                    larger = max(previous_positive_area, area)
                    frame_change_ratio = larger / smaller if smaller > 0 else 1.0
                    frame_change_abs = abs(area - previous_positive_area)

                if area < min_area:
                    reasons.append("degenerate_area")
                elif median_area > 0 and area_ratio >= high_area_ratio:
                    reasons.append("high_area_ratio")
                elif median_area > 0 and low_area_ratio > 0 and area_ratio <= low_area_ratio:
                    reasons.append("low_area_ratio")

                if (
                    previous_positive_area is not None
                    and area >= min_area
                    and frame_change_ratio >= frame_area_change_ratio
                    and frame_change_abs >= frame_area_change_abs
                ):
                    reasons.append("frame_area_jump")

                if robust_z_threshold > 0 and abs(robust_z) >= robust_z_threshold:
                    reasons.append("robust_z")

                if reasons:
                    suspicious.append({
                        **row,
                        "area_ratio": area_ratio,
                        "frame_change_ratio": frame_change_ratio,
                        "frame_change_abs": frame_change_abs,
                        "robust_z": robust_z,
                        "reasons": sorted(set(reasons)),
                    })

                if area >= min_area:
                    previous_positive_area = area

            groups: List[List[Dict[str, Any]]] = []
            for row in suspicious:
                if not groups or row["frame_idx"] - groups[-1][-1]["frame_idx"] > max_gap + 1:
                    groups.append([row])
                else:
                    groups[-1].append(row)

            segments = []
            for group in groups:
                areas = [row["area"] for row in group]
                ratios = [row["area_ratio"] for row in group]
                change_ratios = [row.get("frame_change_ratio", 1.0) for row in group]
                change_abs_values = [row.get("frame_change_abs", 0.0) for row in group]
                robust_zs = [row["robust_z"] for row in group]
                reason_set = sorted({reason for row in group for reason in row["reasons"]})
                segments.append({
                    "start": int(group[0]["frame_idx"]),
                    "end": int(group[-1]["frame_idx"]),
                    "frames": len(group),
                    "reasons": reason_set,
                    "max_area": max(areas),
                    "min_area": min(areas),
                    "max_area_ratio": max(ratios) if ratios else 0.0,
                    "min_area_ratio": min(ratios) if ratios else 0.0,
                    "max_frame_change_ratio": max(change_ratios) if change_ratios else 1.0,
                    "max_frame_change_abs": max(change_abs_values) if change_abs_values else 0.0,
                    "max_abs_robust_z": max((abs(value) for value in robust_zs), default=0.0),
                })

            tracks_out.append({
                "track_id": tid,
                "class_id": track.class_id,
                "num_frames": len(frame_ids),
                "median_area": median_area,
                "q1_area": q1_area,
                "q3_area": q3_area,
                "mad_area": mad_area,
                "segments": segments,
            })

        return {
            "metadata": {
                "video_path": self.video_path,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "frame_count": self.frame_count,
                "num_tracks": len(tracks_out),
            },
            "parameters": {
                "high_area_ratio": high_area_ratio,
                "low_area_ratio": low_area_ratio,
                "frame_area_change_ratio": frame_area_change_ratio,
                "frame_area_change_abs": frame_area_change_abs,
                "robust_z_threshold": robust_z_threshold,
                "min_track_frames": min_track_frames,
                "min_area": min_area,
                "max_gap": max_gap,
            },
            "tracks": tracks_out,
        }

    def get_track_ids(self) -> List[int]:
        return sorted(self.tracks.keys())

    def has_annotation(self, track_id: int, frame_idx: int) -> bool:
        if track_id not in self.tracks:
            return False
        return frame_idx in self.tracks[track_id].frames

    def get_annotated_frame_indices(self, track_id: int) -> List[int]:
        if track_id not in self.tracks:
            return []
        return sorted(self.tracks[track_id].frames.keys())

    def export_tracking_results(self) -> Dict[str, Any]:
        tracks_out = []
        for tid in sorted(self.tracks.keys()):
            track = self.tracks[tid]
            if not track.frames:
                continue
            sorted_frames = sorted(track.frames.keys())
            frames_list = []
            for fidx in sorted_frames:
                bbox = track.frames[fidx]
                bbox_int = [int(round(v)) for v in clip_bbox(bbox, self.width, self.height)]
                frames_list.append({
                    "frame_id": fidx,
                    "video_frame_idx": fidx,
                    "bbox_xyxy": bbox_int,
                })
            tracks_out.append({
                "track_id": tid,
                "class_id": track.class_id,
                "num_frames": len(frames_list),
                "start_frame_id": sorted_frames[0],
                "end_frame_id": sorted_frames[-1],
                "start_video_frame_idx": sorted_frames[0],
                "end_video_frame_idx": sorted_frames[-1],
                "clip_size": [0, 0],
                "clip_path": None,
                "frames": frames_list,
            })

        return {
            "metadata": {
                "video_path": self.video_path,
                "annotation_dir": None,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "frame_count": self.frame_count,
                "frame_offset": 0,
                "num_tracks": len(tracks_out),
            },
            "tracks": tracks_out,
        }

    def save_project(self, path: str) -> None:
        data = {
            "video_path": self.video_path,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "next_track_id": self.next_track_id,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_id": t.class_id,
                    "frames": {str(k): v for k, v in t.frames.items()},
                }
                for t in self.tracks.values()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_project(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.video_path = data["video_path"]
        self.fps = data["fps"]
        self.width = data["width"]
        self.height = data["height"]
        self.frame_count = data["frame_count"]
        self.next_track_id = data["next_track_id"]
        self.tracks = {}
        for t in data["tracks"]:
            track = Track(
                track_id=t["track_id"],
                class_id=t["class_id"],
                frames={int(k): v for k, v in t["frames"].items()},
            )
            self.tracks[track.track_id] = track

    def import_tracking_results(self, path: str) -> None:
        """Import tracking_results.json — all frames are real annotations."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data["metadata"]
        self.video_path = meta.get("video_path", "")
        self.fps = float(meta.get("fps", 30.0))
        self.width = int(meta["width"])
        self.height = int(meta["height"])
        self.frame_count = int(meta.get("frame_count", 0))
        self.tracks = {}
        self.next_track_id = 1

        for track_data in data["tracks"]:
            tid = int(track_data["track_id"])
            track = Track(
                track_id=tid,
                class_id=int(track_data["class_id"]),
            )
            for frame in track_data["frames"]:
                fidx = int(frame["video_frame_idx"])
                bbox = [float(v) for v in frame["bbox_xyxy"]]
                track.frames[fidx] = bbox
            self.tracks[tid] = track
            self.next_track_id = max(self.next_track_id, tid + 1)
