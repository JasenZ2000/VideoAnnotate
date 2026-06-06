from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

from mot_pipeline.models import BBox, FinalTrack
from mot_pipeline.utils.bbox import bbox_center
from mot_pipeline.utils.io import ensure_dir


def make_even(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else value + 1


def compute_clip_size(
    frames: Dict[int, BBox],
    crop_margin: float,
    crop_min_size: int,
) -> Tuple[int, int]:
    # 每个 ID 使用固定裁剪尺寸，取该轨迹历史最大框并附加一定 margin，减少画面缩放抖动。
    widths = [bbox[2] - bbox[0] for bbox in frames.values()]
    heights = [bbox[3] - bbox[1] for bbox in frames.values()]
    clip_w = make_even(max(crop_min_size, int(math.ceil(max(widths) * crop_margin))))
    clip_h = make_even(max(crop_min_size, int(math.ceil(max(heights) * crop_margin))))
    return clip_w, clip_h


def interpolate_bbox(bbox_a: BBox, bbox_b: BBox, t: float) -> BBox:
    return [bbox_a[i] + (bbox_b[i] - bbox_a[i]) * t for i in range(4)]


def build_dense_track_boxes(
    track: FinalTrack,
    pad_frames: int,
    max_video_frame_idx: int,
) -> Dict[int, BBox]:
    # 把稀疏轨迹补成逐帧轨迹：中间线性插值，前后 padding 段保持边界框不变。
    ordered_items = sorted(track.video_frames.items(), key=lambda item: item[1])
    observed_video_frames = [video_idx for _, video_idx in ordered_items]
    observed_boxes = [track.frames[frame_id] for frame_id, _ in ordered_items]
    start_video_idx = max(0, observed_video_frames[0] - pad_frames)
    end_video_idx = min(max_video_frame_idx, observed_video_frames[-1] + pad_frames)
    dense_boxes: Dict[int, BBox] = {}

    for video_idx in range(start_video_idx, observed_video_frames[0]):
        dense_boxes[video_idx] = list(observed_boxes[0])

    for idx in range(len(observed_video_frames) - 1):
        left_idx = observed_video_frames[idx]
        right_idx = observed_video_frames[idx + 1]
        left_bbox = observed_boxes[idx]
        right_bbox = observed_boxes[idx + 1]
        dense_boxes[left_idx] = list(left_bbox)
        gap = right_idx - left_idx
        if gap <= 1:
            continue
        for video_idx in range(left_idx + 1, right_idx):
            t = (video_idx - left_idx) / float(gap)
            dense_boxes[video_idx] = interpolate_bbox(left_bbox, right_bbox, t)

    dense_boxes[observed_video_frames[-1]] = list(observed_boxes[-1])
    for video_idx in range(observed_video_frames[-1] + 1, end_video_idx + 1):
        dense_boxes[video_idx] = list(observed_boxes[-1])
    return dense_boxes


def crop_with_padding(frame, center_x: float, center_y: float, crop_w: int, crop_h: int):
    # 裁剪窗口允许越界，越界区域补黑边，保证输出视频尺寸恒定。
    height, width = frame.shape[:2]
    x1 = int(round(center_x - crop_w / 2.0))
    y1 = int(round(center_y - crop_h / 2.0))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(width, x2)
    src_y2 = min(height, y2)
    crop = frame[src_y1:src_y2, src_x1:src_x2]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
        crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    return crop


def track_color(track_id: int) -> Tuple[int, int, int]:
    # 用确定性颜色区分不同 ID，避免每次运行颜色随机变化。
    return (
        int((37 * track_id + 80) % 256),
        int((97 * track_id + 140) % 256),
        int((17 * track_id + 220) % 256),
    )


def draw_target_on_frame(
    frame,
    bbox: BBox,
    box_thickness: int,
    font_scale: float,
):
    # clip 只绘制当前目标一个框，文本固定为 target，不显示全局 ID。
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    label = "target"
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        max(1, box_thickness),
    )
    text_x = max(0, x1)
    text_y = max(text_h + 4, y1 - 6)
    bg_x2 = min(width - 1, text_x + text_w + 8)
    bg_y1 = max(0, text_y - text_h - 6)
    bg_y2 = min(height - 1, text_y + baseline)
    cv2.rectangle(frame, (text_x, bg_y1), (bg_x2, bg_y2), color, thickness=-1)
    cv2.putText(
        frame,
        label,
        (text_x + 4, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        max(1, box_thickness),
        cv2.LINE_AA,
    )
    return frame


def build_frame_draw_map(final_tracks: Sequence[FinalTrack]) -> Dict[int, List[Tuple[int, BBox]]]:
    # 预先整理每一帧要绘制的轨迹列表，overview 渲染时直接查表。
    frame_draw_map: Dict[int, List[Tuple[int, BBox]]] = defaultdict(list)
    for track in final_tracks:
        for frame_id, bbox in track.frames.items():
            video_idx = track.video_frames[frame_id]
            frame_draw_map[video_idx].append((track.track_id, bbox))
    return frame_draw_map


def draw_tracks_on_frame(
    frame,
    draws: Sequence[Tuple[int, BBox]],
    box_thickness: int,
    font_scale: float,
):
    # 在 overview 帧上绘制所有轨迹框和 ID 文本。
    height, width = frame.shape[:2]
    for track_id, bbox in draws:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = track_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        label = f"ID {track_id}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, box_thickness),
        )
        text_x = max(0, x1)
        text_y = max(text_h + 4, y1 - 6)
        bg_x2 = min(width - 1, text_x + text_w + 8)
        bg_y1 = max(0, text_y - text_h - 6)
        bg_y2 = min(height - 1, text_y + baseline)
        cv2.rectangle(frame, (text_x, bg_y1), (bg_x2, bg_y2), color, thickness=-1)
        cv2.putText(
            frame,
            label,
            (text_x + 4, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, box_thickness),
            cv2.LINE_AA,
        )
    return frame


def prepare_track_clips(
    final_tracks: Sequence[FinalTrack],
    pad_frames: int,
    frame_count: int,
    crop_margin: float,
    crop_min_size: int,
) -> List[FinalTrack]:
    # 为视频导出准备轨迹：先确定固定裁剪大小，再把轨迹转换为逐视频帧的稠密表示。
    max_video_frame_idx = frame_count - 1
    prepared_tracks: List[FinalTrack] = []
    for track in final_tracks:
        track.clip_size = compute_clip_size(track.frames, crop_margin, crop_min_size)
        dense_boxes = build_dense_track_boxes(track, pad_frames, max_video_frame_idx)
        if not dense_boxes:
            continue
        remapped_frames = {video_idx: bbox for video_idx, bbox in dense_boxes.items()}
        track.video_frames = {video_idx: video_idx for video_idx in remapped_frames}
        track.frames = remapped_frames
        prepared_tracks.append(track)
    return prepared_tracks


def build_clip_render_specs(
    clips_dir: Path,
    final_tracks: Sequence[FinalTrack],
    frame_count: int,
) -> Tuple[Dict[int, Dict[str, object]], Dict[int, List[int]], Dict[int, List[int]]]:
    # 为每个轨迹准备裁剪区间、逐帧框和 writer 占位结构。
    render_specs: Dict[int, Dict[str, object]] = {}
    start_map: Dict[int, List[int]] = defaultdict(list)
    end_map: Dict[int, List[int]] = defaultdict(list)

    for track in final_tracks:
        dense_boxes = build_dense_track_boxes(track, pad_frames=0, max_video_frame_idx=frame_count - 1)
        ordered_video_frames = sorted(dense_boxes)
        start_idx = ordered_video_frames[0]
        end_idx = ordered_video_frames[-1]
        clip_path = clips_dir / f"track_{track.track_id:04d}.mp4"
        track.clip_path = clip_path
        render_specs[track.track_id] = {
            "dense_boxes": dense_boxes,
            "start": start_idx,
            "end": end_idx,
            "clip_size": track.clip_size,
            "clip_path": clip_path,
            "writer": None,
        }
        start_map[start_idx].append(track.track_id)
        end_map[end_idx].append(track.track_id)

    return render_specs, start_map, end_map


def render_tracking_overview(
    video_path: Path,
    output_path: Path,
    final_tracks: Sequence[FinalTrack],
    fps: float,
    frame_count: int,
    codec: str,
    box_thickness: int,
    font_scale: float,
) -> None:
    # 在整段原视频上绘制最终轨迹框和 ID，输出一个总览视频用于整体检查跟踪效果。
    if not final_tracks:
        return

    frame_draw_map = build_frame_draw_map(final_tracks)
    max_overview_frame = max(frame_draw_map) if frame_draw_map else -1
    if max_overview_frame < 0:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for overview rendering: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open writer for {output_path}")

    frame_idx = 0
    while frame_idx <= max_overview_frame:
        ok, frame = cap.read()
        if not ok:
            break
        rendered = draw_tracks_on_frame(
            frame,
            frame_draw_map.get(frame_idx, []),
            box_thickness=box_thickness,
            font_scale=font_scale,
        )
        writer.write(rendered)
        frame_idx += 1

    cap.release()
    writer.release()


def extract_track_clips(
    video_path: Path,
    clips_dir: Path,
    final_tracks: Sequence[FinalTrack],
    fps: float,
    frame_count: int,
    codec: str,
    box_thickness: int = 2,
    font_scale: float = 0.7,
) -> None:
    # clip 基于原视频裁剪，但仅叠加当前轨迹一个框，标记为 target。
    if not final_tracks:
        return

    ensure_dir(clips_dir)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    render_specs, start_map, end_map = build_clip_render_specs(clips_dir, final_tracks, frame_count)

    max_end = max(spec["end"] for spec in render_specs.values())
    active_ids = set()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for clip extraction: {video_path}")

    frame_idx = 0
    while frame_idx <= max_end:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id in start_map.get(frame_idx, []):
            spec = render_specs[track_id]
            crop_w, crop_h = spec["clip_size"]
            spec["writer"] = cv2.VideoWriter(str(spec["clip_path"]), fourcc, fps, (crop_w, crop_h))
            if not spec["writer"].isOpened():
                raise RuntimeError(f"Unable to open writer for {spec['clip_path']}")
            active_ids.add(track_id)

        for track_id in list(active_ids):
            spec = render_specs[track_id]
            bbox = spec["dense_boxes"].get(frame_idx)
            if bbox is None:
                continue
            rendered_frame = draw_target_on_frame(
                frame.copy(),
                bbox,
                box_thickness=box_thickness,
                font_scale=font_scale,
            )
            center_x, center_y = bbox_center(bbox)
            crop_w, crop_h = spec["clip_size"]
            crop = crop_with_padding(rendered_frame, center_x, center_y, crop_w, crop_h)
            spec["writer"].write(crop)

        for track_id in end_map.get(frame_idx, []):
            spec = render_specs[track_id]
            writer = spec["writer"]
            if writer is not None:
                writer.release()
            active_ids.discard(track_id)
        frame_idx += 1

    cap.release()
    for spec in render_specs.values():
        writer = spec["writer"]
        if writer is not None:
            writer.release()
