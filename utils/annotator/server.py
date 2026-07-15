"""FastAPI backend for video annotation tool — workspace-based."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import os
import posixpath
import shutil
import sys
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from utils.annotator.state import AnnotationState
from utils.annotator.results import append_tracking_results
from utils.annotator.export_formats import write_jpeg, write_voc_xml

app = FastAPI(title="视频标注工作台")
STATE = AnnotationState()
_cap: Optional[cv2.VideoCapture] = None
_cap_pos: int = -1  # tracks current read position to avoid unnecessary seeks
_cap_lock = threading.Lock()
_workspace: Optional[Path] = None
_workspace_version: int = 0
_sam31_jobs: dict[str, dict[str, Any]] = {}

STATIC_DIR = (
    Path(sys._MEIPASS) / "utils" / "annotator" / "static"
    if getattr(sys, "frozen", False)
    else Path(__file__).parent / "static"
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WorkspaceReq(BaseModel):
    path: str

class AnnotationReq(BaseModel):
    track_id: int
    frame_idx: int
    bbox: list[float]

class DeleteAnnotationReq(BaseModel):
    track_id: int
    frame_idx: int

class DeleteRangeReq(BaseModel):
    track_id: int
    frame_idx: int

class DeleteBetweenReq(BaseModel):
    track_id: int
    frame_a: int
    frame_b: int

class InterpolateReq(BaseModel):
    track_id: int
    frame_a: int
    frame_b: int

class InterpolateShortGapsReq(BaseModel):
    max_gap: int = 5

class FixSpikesReq(BaseModel):
    track_id: int
    area_ratio: Optional[float] = None
    size_ratio: Optional[float] = None
    history: Optional[int] = None
    min_history: Optional[int] = None
    max_run: Optional[int] = None

class MergeTracksReq(BaseModel):
    track_id_a: int
    track_id_b: int

class SetClassReq(BaseModel):
    track_id: int
    class_id: int

class AddTrackReq(BaseModel):
    class_id: int = 0

class SplitReq(BaseModel):
    segment_length: int  # frames per segment

class ExportMergedYoloReq(BaseModel):
    interval: int = 1

class AppendTrackingResultsReq(BaseModel):
    source_path: str
    class_id_override: Optional[int] = None

class Sam31BoxTrackReq(BaseModel):
    track_id: int
    frame_idx: int
    bbox: list[float]
    max_frames: int = 0
    replace_after: bool = True
    cuda_device: Optional[int] = None
    use_rect_mask: Optional[bool] = None

class ConfigInspectReq(BaseModel):
    path: Optional[str] = None

class ConfigSaveReq(BaseModel):
    path: Optional[str] = None
    target: str
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _project_config_path() -> Path:
    configured = os.environ.get("ANNOTATOR_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "config.json"

def _find_video(workspace: Path) -> Optional[Path]:
    for f in sorted(workspace.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            return f
    return None


def _find_label_dir(workspace: Path, video_stem: str) -> Optional[Path]:
    candidate = workspace / video_stem
    if candidate.is_dir():
        return candidate
    return None


def _ensure_config(workspace: Path) -> Path:
    config_path = workspace / "config.json"
    if not config_path.exists():
        project_config = _project_config_path()
        if project_config.exists():
            shutil.copy2(str(project_config), str(config_path))
            return config_path
        from utils.mot_pipeline.config import DEFAULT_CONFIG
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    return config_path


def _load_workspace_config(workspace: Optional[Path]) -> dict[str, Any]:
    from utils.mot_pipeline.config import deep_update, load_config

    project_config = _project_config_path()
    config = load_config(str(project_config)) if project_config.exists() else load_config(None)
    if workspace is None:
        return config

    workspace_config = _ensure_config(workspace)
    if workspace_config.exists():
        with open(workspace_config, "r", encoding="utf-8") as f:
            config = deep_update(config, json.load(f))
    return config


def _get_workspace_config() -> dict[str, Any]:
    return _load_workspace_config(_workspace)


def _config_workspace_from_req(path: Optional[str]) -> Path:
    workspace = Path(path) if path else _workspace
    if workspace is None:
        raise HTTPException(400, "No workspace path provided and no workspace open")
    if not workspace.is_dir():
        raise HTTPException(400, f"Not a directory: {workspace}")
    return workspace


def _read_config_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_config_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _segment_config_targets(workspace: Path) -> list[dict[str, Any]]:
    segments_dir = workspace / "segments"
    if not segments_dir.is_dir():
        return []
    targets = []
    for seg_dir in sorted(segments_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        video = _find_video(seg_dir)
        targets.append({
            "name": seg_dir.name,
            "path": str(seg_dir),
            "config_path": str(seg_dir / "config.json"),
            "has_config": (seg_dir / "config.json").exists(),
            "video": video.name if video else None,
        })
    return targets


def _config_payload_for_workspace(workspace: Path) -> dict[str, Any]:
    from utils.mot_pipeline.config import DEFAULT_CONFIG

    root_config_path = _ensure_config(workspace)
    root_raw = _read_config_file(root_config_path)
    root_effective = _load_workspace_config(workspace)
    segments = []
    for target in _segment_config_targets(workspace):
        seg_dir = Path(target["path"])
        seg_config_path = _ensure_config(seg_dir)
        segments.append({
            **target,
            "config_path": str(seg_config_path),
            "has_config": True,
            "raw": _read_config_file(seg_config_path),
            "effective": _load_workspace_config(seg_dir),
        })

    return {
        "workspace": str(workspace),
        "root": {
            "name": workspace.name,
            "path": str(workspace),
            "config_path": str(root_config_path),
            "raw": root_raw,
            "effective": root_effective,
        },
        "segments": segments,
        "default_config": DEFAULT_CONFIG,
    }


def _get_annotator_config() -> dict[str, int]:
    config = _get_workspace_config().get("annotator", {})
    return {
        "frame_buffer_ahead": max(1, int(config.get("frame_buffer_ahead", 30))),
        "frame_batch_size": max(1, int(config.get("frame_batch_size", 15))),
        "frame_cache_limit": max(1, int(config.get("frame_cache_limit", 80))),
        "frame_batch_max": max(1, int(config.get("frame_batch_max", 30))),
        "annotation_buffer_ahead": max(1, int(config.get("annotation_buffer_ahead", 60))),
        "annotation_batch_size": max(1, int(config.get("annotation_batch_size", 60))),
        "annotation_cache_limit": max(1, int(config.get("annotation_cache_limit", 300))),
        "annotation_batch_max": max(1, int(config.get("annotation_batch_max", 200))),
    }


def _get_area_anomaly_config(workspace: Optional[Path] = None) -> dict[str, Any]:
    config = _load_workspace_config(workspace).get("quality_control", {}).get("area_anomaly", {})
    return {
        "enabled": bool(config.get("enabled", True)),
        "high_area_ratio": float(config.get("high_area_ratio", 3.0)),
        "low_area_ratio": float(config.get("low_area_ratio", 0.25)),
        "frame_area_change_ratio": float(config.get("frame_area_change_ratio", 2.5)),
        "frame_area_change_abs": float(config.get("frame_area_change_abs", 800.0)),
        "robust_z_threshold": float(config.get("robust_z_threshold", 6.0)),
        "min_track_frames": int(config.get("min_track_frames", 8)),
        "min_area": float(config.get("min_area", 1.0)),
        "max_gap": int(config.get("max_gap", 1)),
        "filename": str(config.get("filename", "tracking_area_anomalies.json")),
    }


def _area_anomaly_payload(state: AnnotationState = STATE, workspace: Optional[Path] = None) -> dict[str, Any]:
    cfg = _get_area_anomaly_config(workspace)
    if not cfg["enabled"]:
        return {
            "metadata": {
                "video_path": state.video_path,
                "fps": state.fps,
                "width": state.width,
                "height": state.height,
                "frame_count": state.frame_count,
                "num_tracks": len(state.tracks),
            },
            "parameters": cfg,
            "tracks": [],
        }
    return state.detect_area_anomaly_segments(
        high_area_ratio=cfg["high_area_ratio"],
        low_area_ratio=cfg["low_area_ratio"],
        frame_area_change_ratio=cfg["frame_area_change_ratio"],
        frame_area_change_abs=cfg["frame_area_change_abs"],
        robust_z_threshold=cfg["robust_z_threshold"],
        min_track_frames=cfg["min_track_frames"],
        min_area=cfg["min_area"],
        max_gap=cfg["max_gap"],
    )


def _write_area_anomaly_json(workspace: Path, state: AnnotationState = STATE) -> Path:
    filename = _get_area_anomaly_config(workspace)["filename"]
    path = workspace / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_area_anomaly_payload(state, workspace), f, indent=2)
    return path


def _check_client_version(v: Optional[int]) -> None:
    if v is not None and v != _workspace_version:
        raise HTTPException(409, "Workspace changed; discard stale media request")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/config", response_class=HTMLResponse)
async def config_page():
    return FileResponse(str(STATIC_DIR / "config.html"))


@app.post("/api/config/inspect")
async def inspect_config(req: ConfigInspectReq):
    workspace = _config_workspace_from_req(req.path)
    return _config_payload_for_workspace(workspace)


@app.post("/api/config/save")
async def save_config(req: ConfigSaveReq):
    workspace = _config_workspace_from_req(req.path)
    if req.target == "root":
        target_paths = [_ensure_config(workspace)]
    elif req.target == "all_segments":
        segments = _segment_config_targets(workspace)
        if not segments:
            raise HTTPException(400, "No segments directory or no segment folders")
        target_paths = [_ensure_config(Path(item["path"])) for item in segments]
    else:
        seg_dir = workspace / "segments" / req.target
        if not seg_dir.is_dir():
            raise HTTPException(400, f"Segment not found: {req.target}")
        target_paths = [_ensure_config(seg_dir)]

    for path in target_paths:
        _write_config_file(path, req.config)
    return {
        "ok": True,
        "target": req.target,
        "written": [str(path) for path in target_paths],
        "count": len(target_paths),
    }


@app.post("/api/open-workspace")
async def open_workspace(req: WorkspaceReq):
    """Open a workspace directory: auto-detect video + labels, ensure config, load video."""
    global _cap, _workspace, _cap_pos, _workspace_version

    ws = Path(req.path)
    if not ws.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    video = _find_video(ws)
    if not video:
        raise HTTPException(400, f"No video file found in {req.path}")

    label_dir = _find_label_dir(ws, video.stem)
    config_path = _ensure_config(ws)

    # Load video into state
    STATE.load_video_metadata(str(video))
    STATE.clear_tracks()
    with _cap_lock:
        if _cap is not None:
            _cap.release()
        _cap = cv2.VideoCapture(str(video))
        _cap_pos = -1
        _workspace = ws
        _workspace_version += 1
        version = _workspace_version

    # Load existing tracking results if available
    results_path = ws / "tracking_results.json"
    if results_path.exists():
        STATE.import_tracking_results(str(results_path))
    elif not STATE.tracks:
        STATE.add_track()

    return {
        "ok": True,
        "workspace": str(ws),
        "video": video.name,
        "label_dir": label_dir.name if label_dir else None,
        "has_config": True,
        "has_results": results_path.exists(),
        "frame_count": STATE.frame_count,
        "fps": STATE.fps,
        "width": STATE.width,
        "height": STATE.height,
        "num_tracks": len(STATE.tracks),
        "version": version,
    }


@app.get("/api/state")
async def get_state():
    area_payload = _area_anomaly_payload()
    area_by_track = {
        int(track["track_id"]): track
        for track in area_payload.get("tracks", [])
    }
    tracks_info = []
    for tid in STATE.get_track_ids():
        track = STATE.tracks[tid]
        area_info = area_by_track.get(tid, {})
        area_segments = area_info.get("segments", [])
        tracks_info.append({
            "track_id": tid,
            "class_id": track.class_id,
            "start_frame": track.start_frame,
            "end_frame": track.end_frame,
            "num_frames": len(track.frames),
            "annotated_indices": STATE.get_annotated_frame_indices(tid),
            "area_anomaly_segments": area_segments,
            "area_anomaly_count": len(area_segments),
            "area_median": area_info.get("median_area", 0.0),
        })
    return {
        "workspace": str(_workspace) if _workspace else None,
        "video_path": STATE.video_path,
        "fps": STATE.fps,
        "width": STATE.width,
        "height": STATE.height,
        "frame_count": STATE.frame_count,
        "tracks": tracks_info,
        "annotator": _get_annotator_config(),
        "quality_control": {
            "area_anomaly": {
                "enabled": _get_area_anomaly_config()["enabled"],
                "filename": _get_area_anomaly_config()["filename"],
            }
        },
        "version": _workspace_version,
    }


@app.get("/api/frame-image/{frame_idx}")
async def get_frame_image(frame_idx: int, v: Optional[int] = None):
    """Serve a video frame as JPEG. Avoids seeking when reading sequentially."""
    global _cap_pos
    _check_client_version(v)
    with _cap_lock:
        _check_client_version(v)
        if _cap is None or not _cap.isOpened():
            raise HTTPException(400, "No video loaded")
        # Only seek if not already at the right position
        if _cap_pos != frame_idx:
            _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = _cap.read()
        if not ok:
            _cap_pos = -1
            raise HTTPException(404, f"Cannot read frame {frame_idx}")
        _cap_pos = frame_idx + 1
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/frame-batch/{start}/{count}")
async def get_frame_batch(start: int, count: int, v: Optional[int] = None):
    """Serve multiple consecutive frames as concatenated JPEGs with a length header.

    Response format: for each frame, 4 bytes (big-endian uint32) length + JPEG bytes.
    """
    global _cap_pos
    _check_client_version(v)
    with _cap_lock:
        _check_client_version(v)
        if _cap is None or not _cap.isOpened():
            raise HTTPException(400, "No video loaded")
        count = min(count, _get_annotator_config()["frame_batch_max"])
        if _cap_pos != start:
            _cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        chunks = bytearray()
        frames_read = 0
        for i in range(count):
            ok, frame = _cap.read()
            if not ok:
                break
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = jpeg.tobytes()
            chunks.extend(len(data).to_bytes(4, 'big'))
            chunks.extend(data)
            frames_read += 1
        _cap_pos = start + frames_read

    return Response(
        content=bytes(chunks),
        media_type="application/octet-stream",
        headers={
            "X-Frames-Count": str(frames_read),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/frame/{frame_idx}")
async def get_frame_annotations(frame_idx: int, v: Optional[int] = None):
    _check_client_version(v)
    all_bboxes = STATE.get_all_bboxes_at_frame(frame_idx)
    annotations = []
    for tid, bbox in all_bboxes:
        annotations.append({"track_id": tid, "bbox": bbox})
    return {"frame_idx": frame_idx, "annotations": annotations}


@app.get("/api/annotations-batch/{start}/{count}")
async def get_annotations_batch(start: int, count: int, v: Optional[int] = None):
    _check_client_version(v)
    count = min(count, _get_annotator_config()["annotation_batch_max"])
    end = min(STATE.frame_count, max(0, start) + count)
    frames = []
    for frame_idx in range(max(0, start), end):
        annotations = [
            {"track_id": tid, "bbox": bbox}
            for tid, bbox in STATE.get_all_bboxes_at_frame(frame_idx)
        ]
        frames.append({"frame_idx": frame_idx, "annotations": annotations})
    return {"start": max(0, start), "count": len(frames), "frames": frames}


@app.post("/api/annotation")
async def save_annotation(req: AnnotationReq):
    if req.track_id not in STATE.tracks:
        raise HTTPException(400, f"Track {req.track_id} not found")
    STATE.set_frame(req.track_id, req.frame_idx, req.bbox)
    track = STATE.tracks[req.track_id]
    return {"ok": True, "start_frame": track.start_frame, "end_frame": track.end_frame,
            "num_frames": len(track.frames)}


@app.post("/api/delete-annotation")
async def delete_annotation(req: DeleteAnnotationReq):
    STATE.delete_frame(req.track_id, req.frame_idx)
    return {"ok": True}


@app.post("/api/delete-after")
async def delete_after(req: DeleteRangeReq):
    count = STATE.delete_frames_after(req.track_id, req.frame_idx)
    return {"ok": True, "deleted": count}


@app.post("/api/delete-before")
async def delete_before(req: DeleteRangeReq):
    count = STATE.delete_frames_before(req.track_id, req.frame_idx)
    return {"ok": True, "deleted": count}


@app.post("/api/delete-between")
async def delete_between(req: DeleteBetweenReq):
    if req.track_id not in STATE.tracks:
        raise HTTPException(400, f"Track {req.track_id} not found")
    count = STATE.delete_frames_between(req.track_id, req.frame_a, req.frame_b)
    return {"ok": True, "deleted": count}


@app.post("/api/interpolate")
async def interpolate(req: InterpolateReq):
    if req.track_id not in STATE.tracks:
        raise HTTPException(400, f"Track {req.track_id} not found")
    if not STATE.has_annotation(req.track_id, req.frame_a):
        raise HTTPException(400, f"Frame {req.frame_a} has no annotation")
    if not STATE.has_annotation(req.track_id, req.frame_b):
        raise HTTPException(400, f"Frame {req.frame_b} has no annotation")
    count = STATE.interpolate_range(req.track_id, req.frame_a, req.frame_b)
    return {"ok": True, "interpolated_count": count}


@app.post("/api/interpolate-short-gaps")
async def interpolate_short_gaps(req: InterpolateShortGapsReq):
    max_gap = max(0, int(req.max_gap))
    result = STATE.interpolate_short_gaps(max_gap)
    return {"ok": True, "max_gap": max_gap, **result}


def _spike_fix_params(overrides: Optional[FixSpikesReq] = None) -> dict[str, Any]:
    sam_cfg = _get_workspace_config().get("sam31", {})
    params = {
        "area_ratio": float(sam_cfg.get("spike_area_ratio", 4.0)),
        "size_ratio": float(sam_cfg.get("spike_size_ratio", 3.0)),
        "history": int(sam_cfg.get("spike_history", 10)),
        "min_history": int(sam_cfg.get("spike_min_history", 3)),
        "max_run": int(sam_cfg.get("spike_max_run", 10)),
    }
    if overrides is not None:
        for attr, key in (
            ("area_ratio", "area_ratio"),
            ("size_ratio", "size_ratio"),
            ("history", "history"),
            ("min_history", "min_history"),
            ("max_run", "max_run"),
        ):
            value = getattr(overrides, attr)
            if value is not None:
                params[key] = value
    return params


@app.post("/api/fix-bbox-spikes")
async def fix_bbox_spikes(req: FixSpikesReq):
    if req.track_id not in STATE.tracks:
        raise HTTPException(400, f"Track {req.track_id} not found")
    result = STATE.fix_bbox_spikes(req.track_id, **_spike_fix_params(req))
    return {"ok": True, **result}


@app.post("/api/track")
async def add_track(req: AddTrackReq):
    tid = STATE.add_track(req.class_id)
    return {"ok": True, "track_id": tid}


@app.delete("/api/track/{track_id}")
async def delete_track(track_id: int):
    STATE.delete_track(track_id)
    return {"ok": True}


@app.post("/api/merge-tracks")
async def merge_tracks(req: MergeTracksReq):
    """Merge track B into track A. B is deleted, A gets all of B's frames."""
    try:
        surviving_id = STATE.merge_tracks(req.track_id_a, req.track_id_b)
    except ValueError as e:
        raise HTTPException(400, str(e))
    track = STATE.tracks[surviving_id]
    return {"ok": True, "surviving_track_id": surviving_id,
            "num_frames": len(track.frames)}


@app.post("/api/set-class")
async def set_class(req: SetClassReq):
    """Set the class_id for a track."""
    if req.track_id not in STATE.tracks:
        raise HTTPException(400, f"Track {req.track_id} not found")
    STATE.set_class_id(req.track_id, req.class_id)
    return {"ok": True}


def _apply_sam31_tracking_result(
    result_path: Path,
    track_id: int,
    start_frame: int,
    replace_after: bool,
) -> dict[str, Any]:
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tracks = payload.get("tracks", [])
    if not tracks:
        raise RuntimeError("SAM31 produced no tracks")
    if track_id not in STATE.tracks:
        raise RuntimeError(f"Track {track_id} no longer exists")

    source_track = sorted(tracks, key=lambda item: int(item.get("track_id", 0)))[0]
    target_track = STATE.tracks[track_id]
    if replace_after:
        for frame_idx in [idx for idx in target_track.frames if idx >= start_frame]:
            del target_track.frames[frame_idx]

    updated = 0
    for frame in source_track.get("frames", []):
        frame_idx = int(frame["video_frame_idx"])
        bbox = [float(v) for v in frame["bbox_xyxy"]]
        STATE.set_frame(track_id, frame_idx, bbox)
        updated += 1

    spike_fix = {"fixed_frames": 0, "intervals": []}
    sam_cfg = _get_workspace_config().get("sam31", {})
    if bool(sam_cfg.get("postprocess_spikes", True)):
        spike_fix = STATE.fix_bbox_spikes(track_id, start_frame=start_frame, **_spike_fix_params())

    return {"updated_frames": updated, "spike_fix": spike_fix}


def _json_http_request(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
    service_name: str = "remote service",
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_name} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{service_name} request failed: {exc}") from exc


def _gpu_api_url(server_url: str, service: str, endpoint: str) -> str:
    if service != "sam31":
        raise ValueError(f"Unsupported GPU service namespace: {service}")
    return f"{server_url.rstrip('/')}/api/{service}/{endpoint.lstrip('/')}"


def _map_video_path_for_sam31(video: Path, sam_cfg: dict[str, Any]) -> str:
    local_prefix = str(sam_cfg.get("local_path_prefix", "")).strip()
    remote_prefix = str(sam_cfg.get("remote_path_prefix", "")).strip()
    video_path = str(video.resolve())
    if not local_prefix or not remote_prefix:
        if sys.platform == "win32" and ":" in video_path:
            raise RuntimeError(
                "Local Windows video path cannot be used by the Linux SAM31 server. "
                "Configure sam31.local_path_prefix/remote_path_prefix, or set sam31.video_transfer to sftp."
            )
        return video_path

    local_root = str(Path(local_prefix).resolve())
    cmp_video = video_path.lower() if sys.platform == "win32" else video_path
    cmp_root = local_root.lower() if sys.platform == "win32" else local_root
    if cmp_video == cmp_root:
        rel = ""
    elif cmp_video.startswith(cmp_root.rstrip("\\/") + os.sep):
        rel = os.path.relpath(video_path, local_root)
    else:
        return video_path

    rel_posix = rel.replace("\\", "/")
    return remote_prefix.rstrip("/\\") if not rel_posix else remote_prefix.rstrip("/\\") + "/" + rel_posix


def _sftp_mkdir_p(sftp: Any, remote_dir: str) -> None:
    parts = [part for part in remote_dir.replace("\\", "/").split("/") if part]
    current = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def _remote_video_name(video: Path) -> str:
    stat = video.stat()
    key = f"{video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    digest = hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{video.stem}_{digest}{video.suffix}"


def _upload_video_via_sftp(video: Path, sam_cfg: dict[str, Any], job: dict[str, Any]) -> str:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("SFTP video transfer requires paramiko. Install it with: pip install paramiko") from exc

    host = str(sam_cfg.get("sftp_host", "")).strip()
    username = str(sam_cfg.get("sftp_username", "")).strip()
    remote_dir = str(sam_cfg.get("sftp_remote_dir", "")).strip()
    if not host or not username or not remote_dir:
        raise RuntimeError("sam31.sftp_host, sam31.sftp_username, and sam31.sftp_remote_dir are required for SFTP transfer")

    port = int(sam_cfg.get("sftp_port", 22))
    password_env = str(sam_cfg.get("sftp_password_env", "SAM31_SFTP_PASSWORD")).strip()
    password = sam_cfg.get("sftp_password")
    if password is None:
        password = os.environ.get(password_env) if password_env else None
    key_path = str(sam_cfg.get("sftp_key_path", "")).strip()
    reuse_existing = bool(sam_cfg.get("sftp_reuse_existing", True))

    remote_path = posixpath.join(remote_dir.rstrip("/"), _remote_video_name(video))
    job["message"] = f"Uploading video to {host}:{remote_path}"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": 30,
    }
    if key_path:
        connect_kwargs["key_filename"] = key_path
    if password:
        connect_kwargs["password"] = password
    ssh.connect(**connect_kwargs)
    try:
        sftp = ssh.open_sftp()
        try:
            _sftp_mkdir_p(sftp, remote_dir)
            local_size = video.stat().st_size
            if reuse_existing:
                try:
                    if sftp.stat(remote_path).st_size == local_size:
                        job["message"] = f"Reusing uploaded video {remote_path}"
                        return remote_path
                except OSError:
                    pass
            sftp.put(str(video), remote_path)
            return remote_path
        finally:
            sftp.close()
    finally:
        ssh.close()


def _prepare_video_path_for_sam31(video: Path, sam_cfg: dict[str, Any], job: dict[str, Any]) -> str:
    transfer = str(sam_cfg.get("video_transfer", "path")).lower()
    if transfer == "path":
        return _map_video_path_for_sam31(video, sam_cfg)
    if transfer == "sftp":
        return _upload_video_via_sftp(video, sam_cfg, job)
    raise RuntimeError(f"Unknown sam31.video_transfer: {transfer}")


async def _run_remote_sam31_box_track_job_impl(
    job_id: str,
    req: Sam31BoxTrackReq,
    workspace: Path,
    video: Path,
    sam_cfg: dict[str, Any],
) -> None:
    job = _sam31_jobs[job_id]
    job["status"] = "running"
    server_url = str(sam_cfg.get("server_url", "")).rstrip("/")
    if not server_url:
        raise RuntimeError("sam31.server_url is required when sam31.runner is remote")

    timeout = float(sam_cfg.get("request_timeout", 30))
    poll_interval = float(sam_cfg.get("poll_interval", 2))
    device = str(sam_cfg.get("device", "cuda"))
    if req.cuda_device is not None:
        if req.cuda_device < 0:
            raise RuntimeError("cuda_device must be >= 0")
        device = f"cuda:{req.cuda_device}"

    remote_video_path = await asyncio.to_thread(_prepare_video_path_for_sam31, video, sam_cfg, job)

    payload = {
        "video_path": remote_video_path,
        "bbox": req.bbox,
        "start_frame": req.frame_idx,
        "max_frames": max(0, req.max_frames),
        "class_id": STATE.tracks[req.track_id].class_id,
        "device": device,
        "dtype": str(sam_cfg.get("dtype", "fp16")),
        "min_mask_area": int(sam_cfg.get("min_mask_area", 64)),
        "use_rect_mask": bool(sam_cfg.get("use_rect_mask", False)) if req.use_rect_mask is None else req.use_rect_mask,
    }

    job["message"] = "Submitting remote SAM31 job"
    remote = await asyncio.to_thread(
        _json_http_request,
        "POST",
        _gpu_api_url(server_url, "sam31", "jobs"),
        payload,
        timeout,
    )
    remote_job_id = remote["job_id"]
    job["remote_job_id"] = remote_job_id
    job["remote_server_url"] = server_url
    job["remote_video_path"] = payload["video_path"]

    while True:
        remote_status = await asyncio.to_thread(
            _json_http_request,
            "GET",
            _gpu_api_url(server_url, "sam31", f"jobs/{remote_job_id}"),
            None,
            timeout,
        )
        job["message"] = remote_status.get("message", remote_status.get("status", ""))
        job["remote_status"] = remote_status
        if remote_status.get("status") == "done":
            break
        if remote_status.get("status") == "failed":
            job["stderr"] = remote_status.get("stderr", "")
            raise RuntimeError(remote_status.get("message", "Remote SAM31 failed"))
        await asyncio.sleep(poll_interval)

    result_payload = await asyncio.to_thread(
        _json_http_request,
        "GET",
        _gpu_api_url(server_url, "sam31", f"jobs/{remote_job_id}/tracking-results"),
        None,
        timeout,
    )
    out_dir = workspace / "sam31_runs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "tracking_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    merge_result = _apply_sam31_tracking_result(
        result_path=result_path,
        track_id=req.track_id,
        start_frame=req.frame_idx,
        replace_after=req.replace_after,
    )
    export_path = workspace / "tracking_results.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(STATE.export_tracking_results(), f, indent=2)
    anomaly_path = _write_area_anomaly_json(workspace)

    job["status"] = "done"
    fixed = merge_result["spike_fix"]["fixed_frames"]
    suffix = f"; fixed {fixed} spike frames" if fixed else ""
    job["message"] = f"Updated Track {req.track_id} with {merge_result['updated_frames']} remote SAM31 frames{suffix}"
    job["updated_frames"] = merge_result["updated_frames"]
    job["spike_fix"] = merge_result["spike_fix"]
    job["result_path"] = str(result_path)
    job["export_path"] = str(export_path)
    job["area_anomaly_path"] = str(anomaly_path)


async def _run_local_sam31_box_track_job_impl(job_id: str, req: Sam31BoxTrackReq, workspace: Path, video: Path) -> None:
    job = _sam31_jobs[job_id]
    job["status"] = "running"
    job["message"] = "Starting SAM31"

    config = _load_workspace_config(workspace)
    sam_cfg = config.get("sam31", {})
    script = Path(__file__).resolve().parents[2] / "gpu_services" / "sam31_track.py"
    out_dir = workspace / "sam31_runs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    use_rect_mask = req.use_rect_mask
    if use_rect_mask is None:
        use_rect_mask = bool(sam_cfg.get("use_rect_mask", False))
    device = str(sam_cfg.get("device", "cuda"))
    if req.cuda_device is not None:
        if req.cuda_device < 0:
            raise RuntimeError("cuda_device must be >= 0")
        device = f"cuda:{req.cuda_device}"

    command = [
        sys.executable,
        str(script),
        "--video",
        str(video),
        "--bbox",
        ",".join(f"{v:.3f}" for v in req.bbox),
        "--start-frame",
        str(req.frame_idx),
        "--out-dir",
        str(out_dir),
        "--max-frames",
        str(max(0, req.max_frames)),
        "--class-id",
        str(STATE.tracks[req.track_id].class_id),
        "--comfy-root",
        str(sam_cfg.get("comfy_root", "")),
        "--checkpoint",
        str(sam_cfg.get("checkpoint", "")),
        "--device",
        device,
        "--dtype",
        str(sam_cfg.get("dtype", "fp16")),
        "--min-mask-area",
        str(int(sam_cfg.get("min_mask_area", 64))),
    ]
    if use_rect_mask:
        command.append("--use-rect-mask")

    job["message"] = "SAM31 process running"
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(script.parent),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    job["returncode"] = process.returncode
    job["stdout"] = stdout.decode("utf-8", errors="replace")[-4000:]
    job["stderr"] = stderr.decode("utf-8", errors="replace")[-4000:]

    if process.returncode != 0:
        job["status"] = "failed"
        job["message"] = f"SAM31 failed with exit code {process.returncode}"
        return

    result_path = out_dir / "tracking_results.json"
    if not result_path.exists():
        job["status"] = "failed"
        job["message"] = "SAM31 did not write tracking_results.json"
        return

    merge_result = _apply_sam31_tracking_result(
        result_path=result_path,
        track_id=req.track_id,
        start_frame=req.frame_idx,
        replace_after=req.replace_after,
    )
    export_path = workspace / "tracking_results.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(STATE.export_tracking_results(), f, indent=2)
    anomaly_path = _write_area_anomaly_json(workspace)

    job["status"] = "done"
    fixed = merge_result["spike_fix"]["fixed_frames"]
    suffix = f"; fixed {fixed} spike frames" if fixed else ""
    job["message"] = f"Updated Track {req.track_id} with {merge_result['updated_frames']} SAM31 frames{suffix}"
    job["updated_frames"] = merge_result["updated_frames"]
    job["spike_fix"] = merge_result["spike_fix"]
    job["result_path"] = str(result_path)
    job["export_path"] = str(export_path)
    job["area_anomaly_path"] = str(anomaly_path)


async def _run_sam31_box_track_job(job_id: str, req: Sam31BoxTrackReq, workspace: Path, video: Path) -> None:
    try:
        config = _load_workspace_config(workspace)
        sam_cfg = config.get("sam31", {})
        runner = str(sam_cfg.get("runner", "remote" if sam_cfg.get("server_url") else "local")).lower()
        if runner == "remote":
            await _run_remote_sam31_box_track_job_impl(job_id, req, workspace, video, sam_cfg)
        elif runner == "local":
            await _run_local_sam31_box_track_job_impl(job_id, req, workspace, video)
        else:
            raise RuntimeError(f"Unknown sam31.runner: {runner}")
    except Exception as exc:
        job = _sam31_jobs.get(job_id)
        if job is not None:
            job["status"] = "failed"
            job["message"] = str(exc)


@app.post("/api/sam31/box-track")
async def start_sam31_box_track(req: Sam31BoxTrackReq):
    """Run SAM31 bbox-prompt video tracking in a background subprocess."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    if req.track_id not in STATE.tracks:
        raise HTTPException(400, f"Track {req.track_id} not found")
    if len(req.bbox) != 4:
        raise HTTPException(400, "bbox must contain x1,y1,x2,y2")
    if req.frame_idx < 0 or req.frame_idx >= STATE.frame_count:
        raise HTTPException(400, f"Frame {req.frame_idx} out of range")

    video = _find_video(_workspace)
    if not video:
        raise HTTPException(400, "No video in workspace")

    STATE.set_frame(req.track_id, req.frame_idx, req.bbox)
    job_id = uuid.uuid4().hex
    _sam31_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "message": "Queued",
        "track_id": req.track_id,
        "frame_idx": req.frame_idx,
    }
    asyncio.create_task(_run_sam31_box_track_job(job_id, req, _workspace, video))
    return {"ok": True, "job_id": job_id}


@app.get("/api/sam31/job/{job_id}")
async def get_sam31_job(job_id: str):
    job = _sam31_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "SAM31 job not found")
    return job


@app.post("/api/run-pipeline")
async def run_pipeline_endpoint():
    """Run MOT pipeline using workspace config, output to workspace."""
    global _cap, _cap_pos
    if _workspace is None:
        raise HTTPException(400, "No workspace open")

    video = _find_video(_workspace)
    if not video:
        raise HTTPException(400, "No video in workspace")
    label_dir = _find_label_dir(_workspace, video.stem)
    if not label_dir:
        raise HTTPException(400, f"No label directory '{video.stem}' in workspace")
    config_path = _ensure_config(_workspace)

    from utils.mot_pipeline.config import load_config
    from utils.mot_pipeline.pipeline import run_pipeline

    config = load_config(str(config_path))
    run_pipeline(
        video_path=video,
        ann_dir=label_dir,
        out_dir=_workspace,
        config=config,
    )

    # Reload results into annotator
    results_json = _workspace / "tracking_results.json"
    if not results_json.exists():
        raise HTTPException(500, "Pipeline produced no output")
    STATE.import_tracking_results(str(results_json))
    STATE.load_video_metadata(str(video))
    with _cap_lock:
        if _cap is not None:
            _cap.release()
        _cap = cv2.VideoCapture(str(video))
        _cap_pos = -1
    anomaly_path = _write_area_anomaly_json(_workspace)
    return {"ok": True, "num_tracks": len(STATE.tracks), "area_anomaly_path": str(anomaly_path)}


@app.post("/api/render")
async def render_clips_and_overview():
    """Re-render clips and overview video from current annotations."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")

    video = _find_video(_workspace)
    if not video:
        raise HTTPException(400, "No video in workspace")

    from utils.mot_pipeline.clips import (
        extract_track_clips,
        prepare_track_clips,
        render_tracking_overview,
    )
    from utils.mot_pipeline.config import load_config
    from utils.mot_pipeline.models import FinalTrack
    from utils.mot_pipeline.pipeline import clone_final_tracks

    config_path = _ensure_config(_workspace)
    config = load_config(str(config_path))
    clips_cfg = config["clips"]
    pad_frames = int(clips_cfg.get("pad_frames", 10))

    # Build FinalTrack objects from current annotation state
    final_tracks: list[FinalTrack] = []
    for tid in STATE.get_track_ids():
        track = STATE.tracks[tid]
        if not track.frames:
            continue
        sorted_indices = sorted(track.frames.keys())
        frames_dict = {idx: list(track.frames[idx]) for idx in sorted_indices}
        video_frames_dict = {idx: idx for idx in sorted_indices}
        final_tracks.append(FinalTrack(
            track_id=tid,
            class_id=track.class_id,
            frames=frames_dict,
            video_frames=video_frames_dict,
        ))

    if not final_tracks:
        raise HTTPException(400, "No tracks to render")

    clips_dir = _workspace / "clips"
    overview_path = _workspace / clips_cfg["overview_filename"]

    # Render overview with the original (non-densified) tracks
    output_tracks = clone_final_tracks(final_tracks)
    render_tracking_overview(
        video_path=video,
        output_path=overview_path,
        final_tracks=output_tracks,
        fps=STATE.fps,
        frame_count=STATE.frame_count,
        codec=clips_cfg["codec"],
        box_thickness=clips_cfg["overview_box_thickness"],
        font_scale=clips_cfg["overview_font_scale"],
    )

    # Prepare and extract clips (densifies + pads tracks)
    clip_tracks = prepare_track_clips(
        final_tracks=clone_final_tracks(final_tracks),
        pad_frames=pad_frames,
        frame_count=STATE.frame_count,
        crop_margin=clips_cfg["crop_margin"],
        crop_min_size=clips_cfg["crop_min_size"],
    )
    extract_track_clips(
        video_path=video,
        clips_dir=clips_dir,
        final_tracks=clip_tracks,
        fps=STATE.fps,
        frame_count=STATE.frame_count,
        codec=clips_cfg["codec"],
        pad_frames=pad_frames,
        box_thickness=clips_cfg["overview_box_thickness"],
        font_scale=clips_cfg["overview_font_scale"],
    )

    return {
        "ok": True,
        "num_tracks": len(final_tracks),
        "overview": str(overview_path),
        "clips_dir": str(clips_dir),
    }


@app.post("/api/save")
async def save_project():
    """Save annotation project to workspace."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    path = str(_workspace / "annotation_project.json")
    STATE.save_project(path)
    return {"ok": True, "path": path}


@app.post("/api/load")
async def load_project():
    """Load annotation project from workspace."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    path = _workspace / "annotation_project.json"
    if not path.exists():
        raise HTTPException(400, "No project file in workspace")
    STATE.load_project(str(path))
    return {"ok": True}


@app.post("/api/export")
async def export_results():
    """Export tracking results JSON to workspace."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    payload = STATE.export_tracking_results()
    path = _workspace / "tracking_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    anomaly_path = _write_area_anomaly_json(_workspace)
    return {
        "ok": True,
        "path": str(path),
        "area_anomaly_path": str(anomaly_path),
        "num_tracks": payload["metadata"]["num_tracks"],
    }


@app.post("/api/append-tracking-results")
async def append_tracking_results_endpoint(req: AppendTrackingResultsReq):
    """Append external tracking results into the active workspace without ID conflicts."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    source_path = Path(req.source_path).expanduser().resolve()
    if not source_path.is_file():
        raise HTTPException(400, f"Tracking results file does not exist: {source_path}")
    try:
        additional = json.loads(source_path.read_text(encoding="utf-8"))
        merged = append_tracking_results(
            STATE.export_tracking_results(),
            additional,
            class_id_override=req.class_id_override,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Invalid tracking results: {exc}") from exc

    output_path = _workspace / "tracking_results.json"
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    STATE.import_tracking_results(str(output_path))
    anomaly_path = _write_area_anomaly_json(_workspace)
    return {
        "ok": True,
        "path": str(output_path),
        "source_path": str(source_path),
        "num_tracks": merged["metadata"]["num_tracks"],
        "area_anomaly_path": str(anomaly_path),
    }


@app.post("/api/export-yolo")
async def export_yolo(request: Request):
    """Export images with YOLO TXT and Pascal VOC XML annotations."""
    global _cap_pos
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    if _cap is None or not _cap.isOpened():
        raise HTTPException(400, "No video loaded")

    body = await request.json()
    interval = int(body.get("interval", 1))
    if interval < 1:
        interval = 1

    # Create output dirs
    yoloset_dir = _workspace / "yoloset"
    images_dir = yoloset_dir / "images"
    labels_dir = yoloset_dir / "labels"
    annotations_dir = yoloset_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    raw_class_labels = _get_workspace_config().get("exports", {}).get("class_labels", {})
    class_labels = {
        int(class_id): str(label)
        for class_id, label in raw_class_labels.items()
        if str(class_id).lstrip("-").isdigit()
    }

    # Collect all annotated frames across all tracks
    all_annotations: dict[int, list[tuple[int, list[float]]]] = {}
    for tid in STATE.get_track_ids():
        track = STATE.tracks[tid]
        for frame_idx, bbox in track.frames.items():
            if frame_idx not in all_annotations:
                all_annotations[frame_idx] = []
            all_annotations[frame_idx].append((track.class_id, bbox))

    # Determine which frames to export
    frame_indices = list(range(0, STATE.frame_count, interval))
    w, h = STATE.width, STATE.height
    video_stem = Path(STATE.video_path).stem or "video"

    saved_count = 0
    with _cap_lock:
        for target in frame_indices:
            if _cap_pos != target:
                _cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                _cap_pos = target
            ok, frame = _cap.read()
            if not ok:
                raise HTTPException(500, f"Cannot read frame {target} while exporting")
            _cap_pos = target + 1

            stem = f"{video_stem}_frame_{target:06d}"
            img_name = f"{stem}.jpg"
            annotations = all_annotations.get(target, [])
            write_jpeg(images_dir / img_name, frame)
            write_voc_xml(
                annotations_dir / f"{stem}.xml",
                img_name,
                w,
                h,
                annotations,
                class_labels,
            )

            with (labels_dir / f"{stem}.txt").open("w", encoding="utf-8") as f:
                for class_id, bbox in annotations:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            saved_count += 1

    return {
        "ok": True,
        "output_dir": str(yoloset_dir),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "annotations_dir": str(annotations_dir),
        "frames_saved": saved_count,
        "interval": interval,
    }


@app.post("/api/split-workspace")
async def split_workspace(req: SplitReq):
    """Split the workspace video + labels into fixed-length segment sub-workspaces."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")

    video = _find_video(_workspace)
    if not video:
        raise HTTPException(400, "No video in workspace")
    label_dir = _find_label_dir(_workspace, video.stem)

    # Read video metadata
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    seg_len = max(1, req.segment_length)
    segments_dir = _workspace / "segments"
    segments_dir.mkdir(exist_ok=True)

    # Collect label files indexed by frame number
    label_files: dict[int, Path] = {}
    if label_dir:
        import re
        for txt in sorted(label_dir.glob("*.txt")):
            match = re.search(r"_([0-9]+)\.txt$", txt.name)
            if match:
                label_files[int(match.group(1))] = txt

    segments_info = []
    seg_idx = 0
    frame_num = 0

    while frame_num < frame_count:
        seg_start = frame_num
        seg_end = min(frame_num + seg_len, frame_count)
        seg_name = f"seg_{seg_idx:03d}"
        seg_dir = segments_dir / seg_name
        seg_dir.mkdir(exist_ok=True)

        # Write segment video
        seg_video_path = seg_dir / video.name
        writer = cv2.VideoWriter(str(seg_video_path), fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        for i in range(seg_end - seg_start):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()

        # Copy corresponding label files
        seg_label_dir = seg_dir / video.stem
        seg_label_dir.mkdir(exist_ok=True)
        labels_copied = 0
        for orig_frame_idx in range(seg_start, seg_end):
            # Labels may be 0-indexed or 1-indexed; try both
            src = label_files.get(orig_frame_idx) or label_files.get(orig_frame_idx + 1)
            if src:
                # Rename to local frame index within segment
                local_idx = orig_frame_idx - seg_start
                dst = seg_label_dir / f"{video.stem}_{local_idx + 1}.txt"
                shutil.copy2(str(src), str(dst))
                labels_copied += 1

        # Copy config
        config_src = _workspace / "config.json"
        if config_src.exists():
            shutil.copy2(str(config_src), str(seg_dir / "config.json"))

        segments_info.append({
            "name": seg_name,
            "path": str(seg_dir),
            "start_frame": seg_start,
            "end_frame": seg_end - 1,
            "num_frames": seg_end - seg_start,
            "labels_copied": labels_copied,
        })

        seg_idx += 1
        frame_num = seg_end

    cap.release()
    return {"ok": True, "num_segments": len(segments_info), "segments": segments_info}


@app.get("/api/segments")
async def list_segments():
    """List available segments in the current workspace."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    segments_dir = _workspace / "segments"
    if not segments_dir.is_dir():
        return {"segments": []}
    segments = []
    for seg_dir in sorted(segments_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        info = {"name": seg_dir.name, "path": str(seg_dir)}
        # Check if it has tracking results
        info["has_results"] = (seg_dir / "tracking_results.json").exists()
        # Check video
        seg_video = _find_video(seg_dir)
        if seg_video:
            tmp_cap = cv2.VideoCapture(str(seg_video))
            info["frame_count"] = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tmp_cap.release()
        else:
            info["frame_count"] = 0
        segments.append(info)
    return {"segments": segments}


@app.post("/api/export-merged-yolo")
async def export_merged_yolo(req: ExportMergedYoloReq):
    """Export merged images with YOLO TXT and Pascal VOC XML annotations."""
    if _workspace is None:
        raise HTTPException(400, "No workspace open")

    segments_dir = _workspace / "segments"
    if not segments_dir.is_dir():
        raise HTTPException(400, "No segments directory")

    interval = max(1, req.interval)
    yoloset_dir = _workspace / "yoloset"
    images_dir = yoloset_dir / "images"
    labels_dir = yoloset_dir / "labels"
    annotations_dir = yoloset_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    raw_class_labels = _get_workspace_config().get("exports", {}).get("class_labels", {})
    class_labels = {
        int(class_id): str(label)
        for class_id, label in raw_class_labels.items()
        if str(class_id).lstrip("-").isdigit()
    }

    total_saved = 0

    for seg_dir in sorted(segments_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        results_path = seg_dir / "tracking_results.json"
        if not results_path.exists():
            continue

        seg_video = _find_video(seg_dir)
        if not seg_video:
            continue

        # Load tracking results for this segment
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data["metadata"]
        seg_w = int(meta["width"])
        seg_h = int(meta["height"])
        seg_frame_count = int(meta.get("frame_count", 0))

        # Build per-frame annotations
        frame_anns: dict[int, list[tuple[int, list[float]]]] = {}
        for track in data["tracks"]:
            class_id = int(track["class_id"])
            for frame in track["frames"]:
                fidx = int(frame["video_frame_idx"])
                if fidx not in frame_anns:
                    frame_anns[fidx] = []
                frame_anns[fidx].append((class_id, [float(v) for v in frame["bbox_xyxy"]]))

        # Read video and export frames at interval
        cap = cv2.VideoCapture(str(seg_video))
        if not cap.isOpened():
            continue

        seg_name = seg_dir.name
        for fidx in range(0, seg_frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                break

            img_name = f"{seg_name}_frame_{fidx:06d}.jpg"
            write_jpeg(images_dir / img_name, frame)

            stem = f"{seg_name}_frame_{fidx:06d}"
            annotations = frame_anns.get(fidx, [])
            write_voc_xml(
                annotations_dir / f"{stem}.xml",
                img_name,
                seg_w,
                seg_h,
                annotations,
                class_labels,
            )
            with open(labels_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                for class_id, bbox in annotations:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0 / seg_w
                    cy = (y1 + y2) / 2.0 / seg_h
                    bw = (x2 - x1) / seg_w
                    bh = (y2 - y1) / seg_h
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            total_saved += 1
        cap.release()

    return {
        "ok": True,
        "output_dir": str(yoloset_dir),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "annotations_dir": str(annotations_dir),
        "frames_saved": total_saved,
        "interval": interval,
    }


@app.post("/api/run-pipeline-all-segments")
async def run_pipeline_all_segments():
    """Run MOT pipeline on all segments sequentially."""
    global _cap, _cap_pos, _workspace
    if _workspace is None:
        raise HTTPException(400, "No workspace open")

    segments_dir = _workspace / "segments"
    if not segments_dir.is_dir():
        raise HTTPException(400, "No segments directory — split first")

    from utils.mot_pipeline.config import load_config
    from utils.mot_pipeline.pipeline import run_pipeline

    results = []
    for seg_dir in sorted(segments_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        video = _find_video(seg_dir)
        if not video:
            continue
        label_dir = _find_label_dir(seg_dir, video.stem)
        if not label_dir:
            continue
        config_path = _ensure_config(seg_dir)
        config = load_config(str(config_path))
        try:
            run_pipeline(
                video_path=video,
                ann_dir=label_dir,
                out_dir=seg_dir,
                config=config,
            )
            # Count tracks from output
            results_json = seg_dir / "tracking_results.json"
            num_tracks = 0
            anomaly_path = None
            anomaly_segments = 0
            if results_json.exists():
                with open(results_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                num_tracks = data["metadata"]["num_tracks"]
                seg_state = AnnotationState()
                seg_state.import_tracking_results(str(results_json))
                anomaly_path = _write_area_anomaly_json(seg_dir, seg_state)
                anomaly_payload = _area_anomaly_payload(seg_state, seg_dir)
                anomaly_segments = sum(len(track.get("segments", [])) for track in anomaly_payload.get("tracks", []))
            results.append({
                "name": seg_dir.name,
                "num_tracks": num_tracks,
                "area_anomaly_segments": anomaly_segments,
                "area_anomaly_path": str(anomaly_path) if anomaly_path else None,
                "ok": True,
            })
        except Exception as e:
            results.append({"name": seg_dir.name, "num_tracks": 0, "ok": False, "error": str(e)})

    return {"ok": True, "results": results}


def main(argv: Optional[list[str]] = None):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    default_host = os.environ.get("ANNOTATOR_HOST", "0.0.0.0")
    default_port = int(os.environ.get("ANNOTATOR_PORT") or os.environ.get("PORT", "7860"))
    parser = argparse.ArgumentParser(description="Run the video annotator server.")
    parser.add_argument("--host", default=default_host, help="Bind host, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port, help="Bind port, default: 7860")
    args = parser.parse_args(argv)

    config = uvicorn.Config(app, host=args.host, port=args.port)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
