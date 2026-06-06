from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
                if len(parts) != 5:
                    warnings.append(
                        f"Skipping malformed annotation line {line_no} in {path.name}: {line}"
                    )
                    continue
                try:
                    class_id = int(float(parts[0]))
                    cx, cy, w, h = [float(v) for v in parts[1:]]
                except ValueError:
                    warnings.append(
                        f"Skipping non-numeric annotation line {line_no} in {path.name}: {line}"
                    )
                    continue
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
