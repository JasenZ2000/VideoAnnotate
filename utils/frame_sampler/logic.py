from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2

from utils.mot_pipeline.utils.converters import pixel_xyxy_to_yolo_line


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass(slots=True)
class SamplingSegment:
    start_frame: int
    end_frame: int
    interval: int
    label: str = ""


@dataclass(slots=True)
class SamplingPlan:
    default_interval: int = 0
    include_empty_frames: bool = True
    file_prefix: str = "frame"
    segments: list[SamplingSegment] | None = None

    def __post_init__(self) -> None:
        if self.segments is None:
            self.segments = []


def find_first_video(path: Path) -> Optional[Path]:
    for candidate in sorted(path.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    return None


def load_tracking_results(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_sampling_plan(plan: SamplingPlan, frame_count: int) -> SamplingPlan:
    if frame_count <= 0:
        raise ValueError("Frame count must be positive")
    if plan.default_interval < 0:
        raise ValueError("Default interval must be >= 0")

    validated_segments: list[SamplingSegment] = []
    for segment in sorted(plan.segments or [], key=lambda item: (item.start_frame, item.end_frame)):
        if segment.interval < 1:
            raise ValueError("Segment interval must be >= 1")
        if segment.start_frame < 0 or segment.end_frame < 0:
            raise ValueError("Segment frames must be >= 0")
        if segment.start_frame > segment.end_frame:
            raise ValueError("Segment start frame must be <= end frame")
        if segment.end_frame >= frame_count:
            raise ValueError(f"Segment end frame {segment.end_frame} is outside video range")
        if validated_segments and segment.start_frame <= validated_segments[-1].end_frame:
            raise ValueError("Sampling segments must not overlap")
        validated_segments.append(
            SamplingSegment(
                start_frame=int(segment.start_frame),
                end_frame=int(segment.end_frame),
                interval=int(segment.interval),
                label=(segment.label or "").strip(),
            )
        )

    return SamplingPlan(
        default_interval=int(plan.default_interval),
        include_empty_frames=bool(plan.include_empty_frames),
        file_prefix=(plan.file_prefix or "frame").strip() or "frame",
        segments=validated_segments,
    )


def frame_is_covered(frame_idx: int, segments: Iterable[SamplingSegment]) -> bool:
    for segment in segments:
        if segment.start_frame <= frame_idx <= segment.end_frame:
            return True
    return False


def sample_segment_frames(segment: SamplingSegment) -> list[int]:
    frames = list(range(segment.start_frame, segment.end_frame + 1, segment.interval))
    if not frames or frames[-1] != segment.end_frame:
        frames.append(segment.end_frame)
    return frames


def build_sampled_frame_indices(frame_count: int, plan: SamplingPlan) -> list[int]:
    validated = validate_sampling_plan(plan, frame_count)
    frame_indices: set[int] = set()

    if validated.default_interval > 0:
        for frame_idx in range(0, frame_count, validated.default_interval):
            if not frame_is_covered(frame_idx, validated.segments):
                frame_indices.add(frame_idx)

    for segment in validated.segments:
        frame_indices.update(sample_segment_frames(segment))

    return sorted(frame_indices)


def build_annotations_by_frame(payload: dict[str, Any]) -> dict[int, list[tuple[int, list[float]]]]:
    annotations: dict[int, list[tuple[int, list[float]]]] = {}
    for track in payload.get("tracks", []):
        class_id = int(track["class_id"])
        for frame in track.get("frames", []):
            frame_idx = int(frame.get("video_frame_idx", frame.get("frame_id", 0)))
            annotations.setdefault(frame_idx, []).append(
                (class_id, [float(value) for value in frame["bbox_xyxy"]])
            )
    return annotations


def save_sampling_plan(path: str | Path, plan: SamplingPlan, *, video_path: str, tracking_path: str) -> Path:
    payload = {
        "video_path": str(video_path),
        "tracking_path": str(tracking_path),
        "default_interval": int(plan.default_interval),
        "include_empty_frames": bool(plan.include_empty_frames),
        "file_prefix": plan.file_prefix,
        "segments": [asdict(segment) for segment in plan.segments or []],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return out_path


def export_sampled_yolo_dataset(
    video_path: str | Path,
    tracking_results_path: str | Path,
    output_dir: str | Path,
    plan: SamplingPlan,
    *,
    image_quality: int = 95,
) -> dict[str, Any]:
    payload = load_tracking_results(tracking_results_path)
    metadata = payload["metadata"]
    frame_count = int(metadata.get("frame_count", 0))
    image_w = int(metadata["width"])
    image_h = int(metadata["height"])

    validated_plan = validate_sampling_plan(plan, frame_count)
    selected_frames = build_sampled_frame_indices(frame_count, validated_plan)
    annotations = build_annotations_by_frame(payload)

    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    written_images = 0
    written_labels = 0
    selected_with_annotations = 0

    try:
        for frame_idx in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame {frame_idx} from {video_path}")

            frame_annotations = annotations.get(frame_idx, [])
            if frame_annotations:
                selected_with_annotations += 1
            elif not validated_plan.include_empty_frames:
                continue

            stem = f"{validated_plan.file_prefix}_{frame_idx:06d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(image_quality)])
            written_images += 1

            with label_path.open("w", encoding="utf-8") as handle:
                lines = [
                    pixel_xyxy_to_yolo_line(
                        bbox_xyxy=bbox,
                        class_id=class_id,
                        image_w=image_w,
                        image_h=image_h,
                    )
                    for class_id, bbox in frame_annotations
                ]
                if lines:
                    handle.write("\n".join(lines))
                    handle.write("\n")
            written_labels += 1
    finally:
        cap.release()

    manifest_path = save_sampling_plan(
        out_dir / "sampling_plan.json",
        validated_plan,
        video_path=str(video_path),
        tracking_path=str(tracking_results_path),
    )

    return {
        "output_dir": str(out_dir),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "sampling_plan_path": str(manifest_path),
        "selected_frames": len(selected_frames),
        "frames_exported": written_images,
        "labels_written": written_labels,
        "frames_with_annotations": selected_with_annotations,
        "include_empty_frames": validated_plan.include_empty_frames,
    }
