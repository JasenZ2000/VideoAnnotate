from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

from mot_pipeline.models import Detection, FinalTrack, FrameDetections
from mot_pipeline.utils.bbox import round_bbox_int, yolo_to_xyxy


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frame_index(path: Path) -> Optional[int]:
    match = re.search(r"_([0-9]+)\.txt$", path.name)
    if not match:
        return None
    return int(match.group(1))


def get_video_metadata(video_path: Path) -> Tuple[int, int, int, float]:
    # 统一从视频读取尺寸、总帧数和 FPS，供解析、裁剪和输出模块共用。
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    return width, height, frame_count, fps


def _percentile(values: List[float], q: float) -> float:
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


def _bbox_area_xyxy(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def build_area_anomaly_report(payload: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", True))
    high_area_ratio = max(1.0, float(cfg.get("high_area_ratio", 3.0)))
    low_area_ratio = max(0.0, min(1.0, float(cfg.get("low_area_ratio", 0.25))))
    frame_area_change_ratio = max(1.0, float(cfg.get("frame_area_change_ratio", 2.5)))
    frame_area_change_abs = max(0.0, float(cfg.get("frame_area_change_abs", 800.0)))
    robust_z_threshold = max(0.0, float(cfg.get("robust_z_threshold", 6.0)))
    min_track_frames = max(1, int(cfg.get("min_track_frames", 8)))
    min_area = max(0.0, float(cfg.get("min_area", 1.0)))
    max_gap = max(0, int(cfg.get("max_gap", 1)))

    tracks_out = []
    if enabled:
        for track in payload.get("tracks", []):
            rows = []
            for frame in track.get("frames", []):
                bbox = [float(value) for value in frame["bbox_xyxy"]]
                rows.append({
                    "frame_idx": int(frame["video_frame_idx"]),
                    "area": _bbox_area_xyxy(bbox),
                })

            positive_areas = [row["area"] for row in rows if row["area"] >= min_area]
            if len(positive_areas) < min_track_frames:
                tracks_out.append({
                    "track_id": int(track["track_id"]),
                    "class_id": int(track["class_id"]),
                    "num_frames": len(rows),
                    "median_area": 0.0,
                    "q1_area": 0.0,
                    "q3_area": 0.0,
                    "mad_area": 0.0,
                    "segments": [],
                })
                continue

            median_area = median(positive_areas)
            q1_area = _percentile(positive_areas, 0.25)
            q3_area = _percentile(positive_areas, 0.75)
            mad_area = median([abs(area - median_area) for area in positive_areas])
            suspicious = []
            previous_positive_area = None
            for row in rows:
                area = row["area"]
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
                reasons = []
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
                segments.append({
                    "start": int(group[0]["frame_idx"]),
                    "end": int(group[-1]["frame_idx"]),
                    "frames": len(group),
                    "reasons": sorted({reason for row in group for reason in row["reasons"]}),
                    "max_area": max(areas),
                    "min_area": min(areas),
                    "max_area_ratio": max(ratios) if ratios else 0.0,
                    "min_area_ratio": min(ratios) if ratios else 0.0,
                    "max_frame_change_ratio": max(change_ratios) if change_ratios else 1.0,
                    "max_frame_change_abs": max(change_abs_values) if change_abs_values else 0.0,
                    "max_abs_robust_z": max((abs(value) for value in robust_zs), default=0.0),
                })

            tracks_out.append({
                "track_id": int(track["track_id"]),
                "class_id": int(track["class_id"]),
                "num_frames": len(rows),
                "median_area": median_area,
                "q1_area": q1_area,
                "q3_area": q3_area,
                "mad_area": mad_area,
                "segments": segments,
            })

    return {
        "metadata": dict(payload.get("metadata", {})),
        "parameters": {
            "enabled": enabled,
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


def write_area_anomaly_report(
    payload: Dict[str, Any],
    out_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    cfg = dict(config or {})
    path = out_dir / str(cfg.get("filename", "tracking_area_anomalies.json"))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_area_anomaly_report(payload, cfg), handle, indent=2)
    return path


def load_annotations(
    ann_dir: Path,
    image_w: int,
    image_h: int,
    video_frame_count: int,
) -> Tuple[List[FrameDetections], int, List[str]]:
    # 读取并排序逐帧 YOLO 标注，同时把归一化框转换为像素坐标。
    # 若标注文件从 1 开始编号，则自动映射到视频的 0-based 帧索引。
    warnings: List[str] = []
    txt_files = sorted(ann_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No annotation txt files found in {ann_dir}")

    indexed_files: List[Tuple[int, Path]] = []
    for path in txt_files:
        frame_idx = extract_frame_index(path)
        if frame_idx is None:
            warnings.append(f"Skipping malformed filename: {path.name}")
            continue
        indexed_files.append((frame_idx, path))
    if not indexed_files:
        raise RuntimeError(f"No usable annotation filenames found in {ann_dir}")

    indexed_files.sort(key=lambda item: item[0])
    min_frame_idx = indexed_files[0][0]
    frame_offset = 1 if min_frame_idx == 1 else 0

    frames: List[FrameDetections] = []
    for frame_id, path in indexed_files:
        video_idx = frame_id - frame_offset
        if video_idx < 0 or video_idx >= video_frame_count:
            warnings.append(
                f"Skipping frame {frame_id} from {path.name}: video index {video_idx} out of range."
            )
            continue
        detections: List[Detection] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) not in (5, 6):
                    warnings.append(
                        f"Skipping malformed annotation line {line_no} in {path.name}: {line}"
                    )
                    continue
                try:
                    class_id = int(float(parts[0]))
                    cx, cy, w, h = [float(v) for v in parts[1:5]]
                    score = float(parts[5]) if len(parts) == 6 else 1.0
                except ValueError:
                    warnings.append(
                        f"Skipping non-numeric annotation line {line_no} in {path.name}: {line}"
                    )
                    continue
                score = max(0.0, min(1.0, score))
                bbox = yolo_to_xyxy(cx, cy, w, h, image_w, image_h)
                if bbox is None:
                    warnings.append(
                        f"Skipping degenerate bbox in {path.name} line {line_no}: {line}"
                    )
                    continue
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        video_frame_idx=video_idx,
                        class_id=class_id,
                        bbox=bbox,
                        score=score,
                    )
                )
        frames.append(
            FrameDetections(
                frame_id=frame_id,
                video_frame_idx=video_idx,
                txt_path=path,
                detections=detections,
            )
        )
    if not frames:
        raise RuntimeError("No valid annotations remained after parsing.")
    return frames, frame_offset, warnings


def write_tracking_outputs(
    out_dir: Path,
    video_path: Path,
    ann_dir: Path,
    fps: float,
    image_w: int,
    image_h: int,
    frame_count: int,
    frame_offset: int,
    final_tracks: Sequence[FinalTrack],
    area_anomaly_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    # 同时输出结构化 JSON 和扁平 CSV，便于后续分析和可视化工具接入。
    json_path = out_dir / "tracking_results.json"
    csv_path = out_dir / "tracking_results.csv"

    json_tracks = []
    csv_rows = []
    for track in final_tracks:
        sorted_frames = sorted(track.frames)
        track_frames = []
        for frame_id in sorted_frames:
            video_idx = track.video_frames[frame_id]
            bbox_int = round_bbox_int(track.frames[frame_id], image_w, image_h)
            entry = {
                "frame_id": frame_id,
                "video_frame_idx": video_idx,
                "bbox_xyxy": bbox_int,
            }
            track_frames.append(entry)
            csv_rows.append(
                {
                    "frame_id": frame_id,
                    "video_frame_idx": video_idx,
                    "track_id": track.track_id,
                    "class_id": track.class_id,
                    "x1": bbox_int[0],
                    "y1": bbox_int[1],
                    "x2": bbox_int[2],
                    "y2": bbox_int[3],
                }
            )

        json_tracks.append(
            {
                "track_id": track.track_id,
                "class_id": track.class_id,
                "num_frames": len(track_frames),
                "start_frame_id": sorted_frames[0],
                "end_frame_id": sorted_frames[-1],
                "start_video_frame_idx": track.video_frames[sorted_frames[0]],
                "end_video_frame_idx": track.video_frames[sorted_frames[-1]],
                "clip_size": list(track.clip_size),
                "clip_path": str(track.clip_path) if track.clip_path else None,
                "frames": track_frames,
            }
        )

    payload = {
        "metadata": {
            "video_path": str(video_path),
            "annotation_dir": str(ann_dir),
            "fps": fps,
            "width": image_w,
            "height": image_h,
            "frame_count": frame_count,
            "frame_offset": frame_offset,
            "num_tracks": len(final_tracks),
        },
        "tracks": json_tracks,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_area_anomaly_report(payload, out_dir, area_anomaly_config)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "video_frame_idx",
                "track_id",
                "class_id",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    return json_path, csv_path


def print_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        print(f"[warning] {warning}")
