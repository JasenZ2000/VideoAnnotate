from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from mot_pipeline.clips import extract_track_clips, prepare_track_clips, render_tracking_overview
from mot_pipeline.fusion import FUSION_REGISTRY, build_final_tracks
from mot_pipeline.models import FinalTrack
from mot_pipeline.tracking import TRACKER_REGISTRY
from mot_pipeline.utils.converters import (
    export_tracking_results_to_label_studio,
    export_tracking_results_to_yolo,
)
from mot_pipeline.utils.io import (
    ensure_dir,
    get_video_metadata,
    load_annotations,
    print_warnings,
    write_tracking_outputs,
)


def clone_final_tracks(tracks: List[FinalTrack]) -> List[FinalTrack]:
    # 裁剪阶段会把轨迹改写成逐视频帧表示，这里复制一份以保留原始输出轨迹。
    return [
        FinalTrack(
            track_id=track.track_id,
            class_id=track.class_id,
            frames={k: list(v) for k, v in track.frames.items()},
            video_frames=dict(track.video_frames),
            clip_size=track.clip_size,
            clip_path=track.clip_path,
        )
        for track in tracks
    ]


def run_pipeline(video_path: Path, ann_dir: Path, out_dir: Path, config: Dict[str, Any]) -> None:
    # 顶层流程编排：解析标注、双向跟踪、轨迹融合、overview 渲染、clip 导出、结果落盘。
    ensure_dir(out_dir)
    clips_dir = out_dir / "clips"
    overview_video_path = out_dir / config["clips"]["overview_filename"]

    image_w, image_h, frame_count, fps = get_video_metadata(video_path)
    frames, frame_offset, parse_warnings = load_annotations(
        ann_dir, image_w, image_h, frame_count
    )
    print_warnings(parse_warnings)

    tracking_cfg = config["tracking"]
    fusion_cfg = config["fusion"]
    clips_cfg = config["clips"]

    tracker_name = tracking_cfg.get("method", "iou_kalman")
    if tracker_name not in TRACKER_REGISTRY:
        raise ValueError(f"Unknown tracking method: {tracker_name}")
    fusion_name = fusion_cfg.get("method", "bidirectional_iou")
    if fusion_name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion method: {fusion_name}")

    use_kalman = not tracking_cfg.get("disable_kalman", False)
    if np is None and use_kalman:
        print("[warning] NumPy is unavailable, falling back to simple motion prediction.")
        use_kalman = False

    print(
        f"Loaded {len(frames)} annotated frames from {ann_dir} "
        f"for video {video_path.name} ({image_w}x{image_h}, {frame_count} frames @ {fps:.3f} fps)."
    )

    tracker = TRACKER_REGISTRY[tracker_name]
    forward_tracks = tracker(
        frames=frames,
        direction="forward",
        min_iou=tracking_cfg["iou_match"],
        max_missed=tracking_cfg["max_missed"],
        use_kalman=use_kalman,
    )
    backward_tracks = tracker(
        frames=frames,
        direction="backward",
        min_iou=tracking_cfg["iou_match"],
        max_missed=tracking_cfg["max_missed"],
        use_kalman=use_kalman,
    )
    print(
        f"Tracking produced {len(forward_tracks)} forward tracklets and "
        f"{len(backward_tracks)} backward tracklets."
    )

    fuse_fn = FUSION_REGISTRY[fusion_name]
    fused_components = fuse_fn(forward_tracks, backward_tracks, fusion_cfg["iou_fuse"])
    final_tracks = build_final_tracks(
        fused_components,
        image_w=image_w,
        image_h=image_h,
        min_track_len=fusion_cfg["min_track_len"],
        smooth_window=fusion_cfg["smooth_window"],
    )
    if not final_tracks:
        print("No final tracks survived the filtering stage.")
        return

    output_tracks = clone_final_tracks(final_tracks)
    clip_tracks = prepare_track_clips(
        final_tracks=clone_final_tracks(final_tracks),
        pad_frames=clips_cfg["pad_frames"],
        frame_count=frame_count,
        crop_margin=clips_cfg["crop_margin"],
        crop_min_size=clips_cfg["crop_min_size"],
    )

    clip_track_by_id = {track.track_id: track for track in clip_tracks}
    for output_track in output_tracks:
        output_track.clip_size = clip_track_by_id[output_track.track_id].clip_size

    render_tracking_overview(
        video_path=video_path,
        output_path=overview_video_path,
        final_tracks=output_tracks,
        fps=fps,
        frame_count=frame_count,
        codec=clips_cfg["codec"],
        box_thickness=clips_cfg["overview_box_thickness"],
        font_scale=clips_cfg["overview_font_scale"],
    )

    extract_track_clips(
        video_path=video_path,
        clips_dir=clips_dir,
        final_tracks=clip_tracks,
        fps=fps,
        frame_count=frame_count,
        codec=clips_cfg["codec"],
        box_thickness=clips_cfg["overview_box_thickness"],
        font_scale=clips_cfg["overview_font_scale"],
    )

    for output_track in output_tracks:
        output_track.clip_path = clip_track_by_id[output_track.track_id].clip_path

    json_path, csv_path = write_tracking_outputs(
        out_dir=out_dir,
        video_path=video_path,
        ann_dir=ann_dir,
        fps=fps,
        image_w=image_w,
        image_h=image_h,
        frame_count=frame_count,
        frame_offset=frame_offset,
        final_tracks=output_tracks,
    )

    exports_cfg = config["exports"]
    yolo_dir = None
    label_studio_path = None
    if exports_cfg.get("export_yolo_from_tracking", True):
        yolo_dir = export_tracking_results_to_yolo(
            tracking_results_json=json_path,
            output_dir=out_dir / exports_cfg["yolo_dirname"],
        )
    if exports_cfg.get("export_label_studio", True):
        label_studio_path = export_tracking_results_to_label_studio(
            tracking_results_json=json_path,
            output_json=out_dir / exports_cfg["label_studio_filename"],
            local_files_prefix=exports_cfg["label_studio_local_files_prefix"],
            class_labels=exports_cfg.get("class_labels", {}),
        )

    print(
        f"Saved {len(output_tracks)} final tracks to {json_path.name} and {csv_path.name}. "
        f"Clips written to {clips_dir}. Overview video: {overview_video_path.name}."
    )
    if yolo_dir is not None:
        print(f"YOLO frame labels exported to {yolo_dir}.")
    if label_studio_path is not None:
        print(f"Label Studio JSON exported to {label_studio_path}.")
