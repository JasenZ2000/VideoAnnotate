"""Standalone FastAPI app for variable-density frame sampling."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import cv2

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from utils.frame_sampler.logic import (
    SamplingPlan,
    SamplingSegment,
    build_sampled_frame_indices,
    export_sampled_yolo_dataset,
    find_first_video,
    load_tracking_results,
    save_sampling_plan,
    sample_segment_frames,
    validate_sampling_plan,
)


app = FastAPI(title="Video Frame Sampler")
STATIC_DIR = (
    Path(sys._MEIPASS) / "utils" / "frame_sampler" / "static"
    if getattr(sys, "frozen", False)
    else Path(__file__).parent / "static"
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_cap: Optional[cv2.VideoCapture] = None
_cap_pos: int = -1
_cap_lock = threading.Lock()
_workspace: Optional[Path] = None
_video_path: Optional[Path] = None
_tracking_path: Optional[Path] = None
_tracking_payload: Optional[dict[str, Any]] = None
_workspace_version: int = 0
_sampling_plan = SamplingPlan(default_interval=30, include_empty_frames=True, file_prefix="frame", segments=[])


class OpenWorkspaceReq(BaseModel):
    path: str
    video_path: Optional[str] = None
    tracking_path: Optional[str] = None


class SamplingSegmentReq(BaseModel):
    start_frame: int
    end_frame: int
    interval: int
    label: str = ""


class SavePlanReq(BaseModel):
    default_interval: int = 30
    include_empty_frames: bool = True
    file_prefix: str = "frame"
    output_dir: Optional[str] = None
    segments: list[SamplingSegmentReq] = []


class ExportReq(SavePlanReq):
    image_quality: int = 95


def _current_plan_path() -> Path:
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    return _workspace / "sampling_plan.json"


def _default_output_dir() -> Path:
    if _workspace is None:
        raise HTTPException(400, "No workspace open")
    return _workspace / "sampled_yoloset"


def _frame_count() -> int:
    if _tracking_payload is None:
        raise HTTPException(400, "No tracking results loaded")
    return int(_tracking_payload["metadata"].get("frame_count", 0))


def _serialize_plan(plan: SamplingPlan) -> dict[str, Any]:
    selected_frames = build_sampled_frame_indices(_frame_count(), plan)
    return {
        "default_interval": plan.default_interval,
        "include_empty_frames": plan.include_empty_frames,
        "file_prefix": plan.file_prefix,
        "segments": [
            {
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "interval": segment.interval,
                "label": segment.label,
                "sampled_frames": len(sample_segment_frames(segment)),
            }
            for segment in plan.segments or []
        ],
        "stats": {
            "selected_frames": len(selected_frames),
            "first_frame": selected_frames[0] if selected_frames else None,
            "last_frame": selected_frames[-1] if selected_frames else None,
        },
    }


def _plan_from_req(req: SavePlanReq) -> SamplingPlan:
    return validate_sampling_plan(
        SamplingPlan(
            default_interval=req.default_interval,
            include_empty_frames=req.include_empty_frames,
            file_prefix=req.file_prefix,
            segments=[
                SamplingSegment(
                    start_frame=item.start_frame,
                    end_frame=item.end_frame,
                    interval=item.interval,
                    label=item.label,
                )
                for item in req.segments
            ],
        ),
        _frame_count(),
    )


def _load_existing_plan(plan_path: Path) -> Optional[SamplingPlan]:
    if not plan_path.exists():
        return None
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_sampling_plan(
        SamplingPlan(
            default_interval=int(payload.get("default_interval", 30)),
            include_empty_frames=bool(payload.get("include_empty_frames", True)),
            file_prefix=str(payload.get("file_prefix", "frame") or "frame"),
            segments=[
                SamplingSegment(
                    start_frame=int(item["start_frame"]),
                    end_frame=int(item["end_frame"]),
                    interval=int(item["interval"]),
                    label=str(item.get("label", "")),
                )
                for item in payload.get("segments", [])
            ],
        ),
        _frame_count(),
    )


def _check_client_version(version: Optional[int]) -> None:
    if version is not None and version != _workspace_version:
        raise HTTPException(409, "Workspace changed; discard stale media request")


def _find_tracking_path(workspace: Path, explicit: Optional[str]) -> Path:
    if explicit:
        candidate = Path(explicit)
    else:
        candidate = workspace / "tracking_results.json"
    if not candidate.is_file():
        raise HTTPException(400, f"tracking_results.json not found: {candidate}")
    return candidate


def _find_video_path(workspace: Path, explicit: Optional[str]) -> Path:
    candidate = Path(explicit) if explicit else find_first_video(workspace)
    if candidate is None or not candidate.is_file():
        raise HTTPException(400, f"No video file found in {workspace}")
    return candidate


def _current_state() -> dict[str, Any]:
    if _workspace is None or _video_path is None or _tracking_path is None or _tracking_payload is None:
        raise HTTPException(400, "No workspace open")
    metadata = _tracking_payload["metadata"]
    plan_payload = _serialize_plan(_sampling_plan)
    return {
        "workspace": str(_workspace),
        "video_path": str(_video_path),
        "tracking_path": str(_tracking_path),
        "plan_path": str(_current_plan_path()),
        "output_dir": str(_default_output_dir()),
        "fps": float(metadata.get("fps", 0.0)),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "frame_count": int(metadata.get("frame_count", 0)),
        "num_tracks": int(metadata.get("num_tracks", len(_tracking_payload.get("tracks", [])))),
        "plan": plan_payload,
        "version": _workspace_version,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "frame-sampler", "api_schema_version": 1}


@app.post("/api/open-workspace")
async def open_workspace(req: OpenWorkspaceReq) -> dict[str, Any]:
    global _cap, _cap_pos, _workspace, _video_path, _tracking_path, _tracking_payload, _workspace_version, _sampling_plan

    workspace = Path(req.path)
    if not workspace.is_dir():
        raise HTTPException(400, f"Not a directory: {workspace}")

    video_path = _find_video_path(workspace, req.video_path)
    tracking_path = _find_tracking_path(workspace, req.tracking_path)
    tracking_payload = load_tracking_results(tracking_path)

    with _cap_lock:
        if _cap is not None:
            _cap.release()
        _cap = cv2.VideoCapture(str(video_path))
        if not _cap.isOpened():
            raise HTTPException(400, f"Unable to open video: {video_path}")
        _cap_pos = -1
        _workspace = workspace
        _video_path = video_path
        _tracking_path = tracking_path
        _tracking_payload = tracking_payload
        _workspace_version += 1

    try:
        existing_plan = _load_existing_plan(_current_plan_path())
    except ValueError as exc:
        raise HTTPException(400, f"Existing sampling_plan.json is invalid: {exc}") from exc
    if existing_plan is not None:
        _sampling_plan = existing_plan
    else:
        default_interval = max(1, int(round(float(tracking_payload["metadata"].get("fps", 0.0))))) if tracking_payload["metadata"].get("fps") else 30
        _sampling_plan = validate_sampling_plan(
            SamplingPlan(default_interval=default_interval, include_empty_frames=True, file_prefix="frame", segments=[]),
            _frame_count(),
        )
        save_sampling_plan(_current_plan_path(), _sampling_plan, video_path=str(_video_path), tracking_path=str(_tracking_path))

    return _current_state()


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return _current_state()


@app.get("/api/frame-image/{frame_idx}")
async def get_frame_image(frame_idx: int, v: Optional[int] = None) -> Response:
    global _cap_pos
    _check_client_version(v)
    with _cap_lock:
        _check_client_version(v)
        if _cap is None or not _cap.isOpened():
            raise HTTPException(400, "No video loaded")
        if _tracking_payload is None:
            raise HTTPException(400, "No tracking results loaded")
        if frame_idx < 0 or frame_idx >= _frame_count():
            raise HTTPException(404, f"Frame {frame_idx} is outside video range")
        if _cap_pos != frame_idx:
            _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = _cap.read()
        if not ok:
            _cap_pos = -1
            raise HTTPException(404, f"Cannot read frame {frame_idx}")
        _cap_pos = frame_idx + 1
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, f"Failed to encode frame {frame_idx}")
    return Response(content=jpeg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/plan")
async def save_plan(req: SavePlanReq) -> dict[str, Any]:
    global _sampling_plan
    if _workspace is None or _video_path is None or _tracking_path is None:
        raise HTTPException(400, "No workspace open")
    _sampling_plan = _plan_from_req(req)
    plan_path = save_sampling_plan(_current_plan_path(), _sampling_plan, video_path=str(_video_path), tracking_path=str(_tracking_path))
    payload = _current_state()
    payload["plan_saved_to"] = str(plan_path)
    if req.output_dir:
        payload["output_dir"] = str(Path(req.output_dir))
    return payload


@app.post("/api/export")
async def export_dataset(req: ExportReq) -> dict[str, Any]:
    global _sampling_plan
    if _workspace is None or _video_path is None or _tracking_path is None:
        raise HTTPException(400, "No workspace open")
    _sampling_plan = _plan_from_req(req)
    output_dir = Path(req.output_dir) if req.output_dir else _default_output_dir()
    save_sampling_plan(_current_plan_path(), _sampling_plan, video_path=str(_video_path), tracking_path=str(_tracking_path))
    try:
        exported = export_sampled_yolo_dataset(
            video_path=_video_path,
            tracking_results_path=_tracking_path,
            output_dir=output_dir,
            plan=_sampling_plan,
            image_quality=req.image_quality,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = _current_state()
    payload["export"] = exported
    payload["output_dir"] = str(output_dir)
    return payload


def main(argv: Optional[list[str]] = None) -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description="Run the variable-density frame sampling tool.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=7871, help="Bind port, default: 7871")
    args = parser.parse_args(argv)

    config = uvicorn.Config(app, host=args.host, port=args.port)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
