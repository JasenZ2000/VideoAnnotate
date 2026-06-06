from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mot_pipeline.utils.bbox import bbox_to_xywh, clip_bbox


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def to_posix_path(path: str | Path) -> str:
    return Path(path).as_posix()


def build_label_studio_video_ref(video_path: str | Path, prefix: str) -> str:
    # 默认按 Label Studio local-files 形式写视频路径。
    return f"{prefix}{to_posix_path(video_path)}"


def get_class_label(class_id: int, class_labels: Dict[str, str]) -> str:
    return class_labels.get(str(class_id), f"class_{class_id}")


def resolve_class_id(label: str, label_to_class_id: Dict[str, int]) -> int:
    if label in label_to_class_id:
        return label_to_class_id[label]
    match = re.fullmatch(r"class_(\d+)", label)
    if match:
        class_id = int(match.group(1))
        label_to_class_id[label] = class_id
        return class_id
    class_id = len(label_to_class_id)
    label_to_class_id[label] = class_id
    return class_id


def pixel_xyxy_to_yolo_line(
    bbox_xyxy: Iterable[float],
    class_id: int,
    image_w: int,
    image_h: int,
) -> str:
    cx, cy, w, h = bbox_to_xywh(list(bbox_xyxy))
    return (
        f"{class_id} "
        f"{cx / image_w:.6f} "
        f"{cy / image_h:.6f} "
        f"{w / image_w:.6f} "
        f"{h / image_h:.6f}"
    )


def export_tracking_results_to_yolo(
    tracking_results_json: str | Path,
    output_dir: str | Path,
    file_prefix: Optional[str] = None,
) -> Path:
    # 将 tracking_results.json 按帧导出成 YOLO txt；无目标帧也会生成空文件。
    json_path = Path(tracking_results_json)
    payload = load_json(json_path)
    metadata = payload["metadata"]
    image_w = int(metadata["width"])
    image_h = int(metadata["height"])
    frame_count = int(metadata.get("frame_count", 0))
    frame_offset = int(metadata.get("frame_offset", 0))
    video_path = Path(metadata.get("video_path", "video"))

    prefix = file_prefix or video_path.stem
    output_path = Path(output_dir)
    ensure_dir(output_path)

    frame_lines: Dict[int, List[str]] = defaultdict(list)
    seen_frame_ids = set()
    for track in payload["tracks"]:
        class_id = int(track["class_id"])
        for frame in track["frames"]:
            frame_id = int(frame["frame_id"])
            seen_frame_ids.add(frame_id)
            frame_lines[frame_id].append(
                pixel_xyxy_to_yolo_line(
                    frame["bbox_xyxy"],
                    class_id=class_id,
                    image_w=image_w,
                    image_h=image_h,
                )
            )

    if frame_count > 0:
        frame_ids = [idx + frame_offset for idx in range(frame_count)]
    else:
        frame_ids = sorted(seen_frame_ids)

    for frame_id in frame_ids:
        txt_path = output_path / f"{prefix}_{frame_id}.txt"
        lines = frame_lines.get(frame_id, [])
        with txt_path.open("w", encoding="utf-8") as handle:
            if lines:
                handle.write("\n".join(lines))
                handle.write("\n")
    return output_path


def track_to_label_studio_box(
    track: Dict[str, Any],
    metadata: Dict[str, Any],
    class_labels: Dict[str, str],
) -> Dict[str, Any]:
    width = float(metadata["width"])
    height = float(metadata["height"])
    fps = float(metadata["fps"])
    frames_count = int(metadata.get("frame_count", 0)) or int(track["end_video_frame_idx"]) + 1
    duration = frames_count / fps if fps > 0 else 0.0

    sequence = []
    sorted_frames = sorted(track["frames"], key=lambda item: item["video_frame_idx"])
    for frame in sorted_frames:
        bbox = frame["bbox_xyxy"]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box_w = x2 - x1
        box_h = y2 - y1
        ls_frame = int(frame["video_frame_idx"]) + 1
        sequence.append(
            {
                "frame": ls_frame,
                "enabled": True,
                "rotation": 0,
                "x": x1 / width * 100.0,
                "y": y1 / height * 100.0,
                "width": box_w / width * 100.0,
                "height": box_h / height * 100.0,
                "time": ls_frame / fps if fps > 0 else 0.0,
            }
        )

    last_ls_frame = int(track["end_video_frame_idx"]) + 1
    if last_ls_frame < frames_count and sequence:
        tail = dict(sequence[-1])
        tail["frame"] = last_ls_frame + 1
        tail["enabled"] = False
        tail["time"] = tail["frame"] / fps if fps > 0 else 0.0
        sequence.append(tail)

    return {
        "framesCount": frames_count,
        "duration": duration,
        "sequence": sequence,
        "labels": [get_class_label(int(track["class_id"]), class_labels)],
    }


def build_label_studio_result_item(track: Dict[str, Any], metadata: Dict[str, Any], class_labels: Dict[str, str]) -> Dict[str, Any]:
    # 将单条轨迹包装成 Label Studio annotations.result 中的一项 videorectangle 标注。
    return {
        "id": f"track-{track['track_id']}",
        "from_name": "box",
        "to_name": "video",
        "type": "videorectangle",
        "origin": "manual",
        "value": track_to_label_studio_box(track, metadata, class_labels),
    }


def export_tracking_results_to_label_studio(
    tracking_results_json: str | Path,
    output_json: str | Path,
    local_files_prefix: str = "/data/local-files/?d=",
    class_labels: Optional[Dict[str, str]] = None,
    task_id: int = 1,
) -> Path:
    # 将 tracking_results.json 导出为 Label Studio 任务导入/导出兼容格式。
    json_path = Path(tracking_results_json)
    payload = load_json(json_path)
    metadata = payload["metadata"]
    class_labels = class_labels or {}

    video_ref = build_label_studio_video_ref(metadata["video_path"], local_files_prefix)
    result_items = [
        build_label_studio_result_item(track, metadata, class_labels)
        for track in payload["tracks"]
    ]
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    task = {
        "id": task_id,
        "annotations": [
            {
                "id": 1,
                "completed_by": 1,
                "result": result_items,
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": timestamp,
                "updated_at": timestamp,
                "lead_time": 0.0,
                "prediction": {},
                "result_count": len(result_items),
                "task": task_id,
            }
        ],
        "predictions": [],
        "drafts": [],
        "data": {
            "video": video_ref,
            "id": task_id,
        },
    }

    output_path = Path(output_json)
    dump_json(output_path, [task])
    return output_path


def parse_video_info_from_name(video_ref: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    # 样例视频名中包含宽高和 FPS，文件不可直接访问时可用它做回退解析。
    name = Path(video_ref).name
    stem = Path(name).stem
    tokens = stem.split("_")
    for idx in range(len(tokens) - 2):
        width_token, height_token, fps_token = tokens[idx : idx + 3]
        if not (width_token.isdigit() and height_token.isdigit() and fps_token.isdigit()):
            continue
        width = int(width_token)
        height = int(height_token)
        fps = float(fps_token)
        if width >= 128 and height >= 128 and 1 <= fps <= 240:
            return width, height, fps
    return None, None, None


def resolve_label_studio_video_path(video_ref: str, local_files_prefix: str) -> Path:
    if video_ref.startswith(local_files_prefix):
        return Path(video_ref[len(local_files_prefix) :])
    return Path(video_ref)


def infer_frame_count_from_label_studio(task: Dict[str, Any]) -> int:
    box_items = extract_label_studio_box_items(task)
    counts = [int(box.get("framesCount", 0)) for box in box_items if box.get("framesCount")]
    if counts:
        return max(counts)
    max_frame = 0
    for box in box_items:
        for item in box.get("sequence", []):
            max_frame = max(max_frame, int(item.get("frame", 0)))
    return max_frame


def extract_label_studio_box_items(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 同时兼容两种输入：
    # 1. 旧的简化结构：{video, box}
    # 2. Label Studio 任务结构：{data:{video,...}, annotations:[{result:[{value:...}]}]}
    if isinstance(task.get("box"), list):
        return task["box"]

    data_section = task.get("data", {})
    if isinstance(data_section.get("box"), list):
        return data_section["box"]

    annotations = task.get("annotations", [])
    if annotations:
        result = annotations[0].get("result", [])
        return [item["value"] for item in result if isinstance(item, dict) and isinstance(item.get("value"), dict)]

    return []


def extract_label_studio_video_ref(task: Dict[str, Any]) -> str:
    if "video" in task:
        return task["video"]
    data_section = task.get("data", {})
    if "video" in data_section:
        return data_section["video"]
    raise RuntimeError("Label Studio JSON does not contain a video field.")


def interpolate_ls_box(start_box: Dict[str, Any], end_box: Dict[str, Any], t: float) -> List[float]:
    sx = float(start_box["x"])
    sy = float(start_box["y"])
    sw = float(start_box["width"])
    sh = float(start_box["height"])
    ex = float(end_box["x"])
    ey = float(end_box["y"])
    ew = float(end_box["width"])
    eh = float(end_box["height"])
    return [
        sx + (ex - sx) * t,
        sy + (ey - sy) * t,
        sw + (ew - sw) * t,
        sh + (eh - sh) * t,
    ]


def ls_percent_box_to_xyxy(
    percent_box: List[float],
    image_w: int,
    image_h: int,
) -> List[int]:
    x = percent_box[0] / 100.0 * image_w
    y = percent_box[1] / 100.0 * image_h
    w = percent_box[2] / 100.0 * image_w
    h = percent_box[3] / 100.0 * image_h
    clipped = clip_bbox([x, y, x + w, y + h], image_w, image_h)
    if clipped is None:
        return [0, 0, 0, 0]
    return [int(round(v)) for v in clipped]


def dense_frames_from_label_studio_sequence(
    sequence: List[Dict[str, Any]],
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    # 将 Label Studio 的关键帧序列展开为逐帧 tracking_results 形式。
    if not sequence:
        return []
    sequence = sorted(sequence, key=lambda item: int(item["frame"]))
    dense_frames: List[Dict[str, Any]] = []

    for idx, current in enumerate(sequence):
        current_frame = int(current["frame"])
        current_enabled = bool(current.get("enabled", True))
        if not current_enabled:
            break

        if idx == len(sequence) - 1:
            bbox_xyxy = ls_percent_box_to_xyxy(
                [current["x"], current["y"], current["width"], current["height"]],
                image_w,
                image_h,
            )
            dense_frames.append(
                {
                    "frame_id": current_frame,
                    "video_frame_idx": current_frame - 1,
                    "bbox_xyxy": bbox_xyxy,
                }
            )
            break

        next_item = sequence[idx + 1]
        next_frame = int(next_item["frame"])
        gap = max(1, next_frame - current_frame)
        for frame_id in range(current_frame, next_frame):
            t = (frame_id - current_frame) / float(gap)
            percent_box = interpolate_ls_box(current, next_item, t)
            bbox_xyxy = ls_percent_box_to_xyxy(percent_box, image_w, image_h)
            dense_frames.append(
                {
                    "frame_id": frame_id,
                    "video_frame_idx": frame_id - 1,
                    "bbox_xyxy": bbox_xyxy,
                }
            )
        if not bool(next_item.get("enabled", True)):
            break

    return dense_frames


def import_label_studio_to_tracking_results(
    label_studio_json: str | Path,
    output_json: str | Path,
    local_files_prefix: str = "/data/local-files/?d=",
) -> Path:
    # 读取 Label Studio 视频跟踪标注，并展开为当前项目使用的 tracking_results.json 结构。
    input_path = Path(label_studio_json)
    data = load_json(input_path)
    tasks = data if isinstance(data, list) else [data]
    if not tasks:
        raise RuntimeError("Empty Label Studio payload.")

    task = tasks[0]
    video_ref = extract_label_studio_video_ref(task)
    video_path = resolve_label_studio_video_path(video_ref, local_files_prefix)

    width = height = None
    fps = None
    width, height, fps = parse_video_info_from_name(video_ref)
    if width is None or height is None or fps is None:
        raise RuntimeError(
            "Unable to resolve video metadata from Label Studio JSON filename. "
            "Expected width/height/fps tokens in the video name."
        )
    frame_count = infer_frame_count_from_label_studio(task)

    label_to_class_id: Dict[str, int] = {}
    tracks_output = []
    max_video_frame_idx = -1
    box_items = extract_label_studio_box_items(task)

    for track_index, box in enumerate(box_items, start=1):
        labels = box.get("labels", [])
        label = labels[0] if labels else "class_0"
        class_id = resolve_class_id(label, label_to_class_id)
        dense_frames = dense_frames_from_label_studio_sequence(
            box.get("sequence", []),
            image_w=int(width),
            image_h=int(height),
        )
        if not dense_frames:
            continue
        max_video_frame_idx = max(max_video_frame_idx, dense_frames[-1]["video_frame_idx"])
        tracks_output.append(
            {
                "track_id": track_index,
                "class_id": class_id,
                "label": label,
                "num_frames": len(dense_frames),
                "start_frame_id": dense_frames[0]["frame_id"],
                "end_frame_id": dense_frames[-1]["frame_id"],
                "start_video_frame_idx": dense_frames[0]["video_frame_idx"],
                "end_video_frame_idx": dense_frames[-1]["video_frame_idx"],
                "clip_size": [0, 0],
                "clip_path": None,
                "frames": dense_frames,
            }
        )
        frame_count = max(frame_count, int(box.get("framesCount", 0)))

    payload = {
        "metadata": {
            "video_path": str(video_path),
            "annotation_dir": None,
            "fps": fps,
            "width": int(width),
            "height": int(height),
            "frame_count": int(frame_count or (max_video_frame_idx + 1)),
            "frame_offset": 1,
            "num_tracks": len(tracks_output),
            "source_format": "label_studio",
            "label_to_class_id": label_to_class_id,
        },
        "tracks": tracks_output,
    }

    output_path = Path(output_json)
    dump_json(output_path, payload)
    return output_path
