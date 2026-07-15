from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gpu_services.device_pool import GPU_DEVICE_POOL, parse_devices


DEFAULT_COMFY_ROOT = Path(os.environ.get("SAM31_COMFY_ROOT", "/opt/ComfyUI"))
DEFAULT_CHECKPOINT = Path(
    os.environ.get("SAM31_CHECKPOINT", "/models/sam3.1_multiplex_fp16.safetensors")
)
DEFAULT_CACHE_DIR = Path(os.environ.get("SAM31_CACHE_DIR", "/tmp/video-annotation-sam31"))
DEFAULT_RUNNER = Path(
    os.environ.get("SAM31_RUNNER", str(Path(__file__).with_name("sam31_track.py")))
)

router = APIRouter(prefix="/api/sam31", tags=["sam31"])
JOBS: dict[str, dict[str, Any]] = {}
SETTINGS: dict[str, Any] = {
    "comfy_root": DEFAULT_COMFY_ROOT,
    "checkpoint": DEFAULT_CHECKPOINT,
    "cache_dir": DEFAULT_CACHE_DIR,
    "runner_path": DEFAULT_RUNNER,
    "runner_python": os.environ.get("SAM31_PYTHON", sys.executable),
    "devices": parse_devices(os.environ.get("SAM31_DEVICES", os.environ.get("SAM31_DEVICE", "cuda"))),
    "default_dtype": os.environ.get("SAM31_DTYPE", "fp16"),
    "allowed_roots": [
        Path(item).expanduser().resolve()
        for item in os.environ.get("SAM31_ALLOWED_ROOTS", "").split(",")
        if item.strip()
    ],
}


class Sam31JobReq(BaseModel):
    video_path: str
    bbox: list[float]
    start_frame: int = 0
    max_frames: int = 0
    class_id: int = 0
    device: Optional[str] = None
    dtype: Optional[str] = None
    min_mask_area: int = 64
    use_rect_mask: bool = False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_video_path(raw_path: str) -> Path:
    if "\\" in raw_path or (len(raw_path) >= 2 and raw_path[1] == ":"):
        raise HTTPException(
            400,
            "Received a local Windows path. Configure path mapping or SFTP upload on the client side.",
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(400, f"Video does not exist: {path}")
    allowed_roots: list[Path] = SETTINGS.get("allowed_roots", [])
    if allowed_roots and not any(_is_relative_to(path, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise HTTPException(403, f"Video path is outside allowed roots: {roots}")
    return path


def _validate_bbox(bbox: list[float]) -> None:
    if len(bbox) != 4:
        raise HTTPException(400, "bbox must contain x1,y1,x2,y2")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise HTTPException(400, "bbox must satisfy x2>x1 and y2>y1")


async def _run_job(job_id: str, req: Sam31JobReq, video_path: Path) -> None:
    job = JOBS[job_id]
    job["message"] = "Waiting for an available GPU"
    device: Optional[str] = None
    try:
        device = await asyncio.to_thread(GPU_DEVICE_POOL.acquire, SETTINGS["devices"], req.device)
        job["assigned_device"] = device
        job["status"] = "running"
        job["message"] = f"Starting SAM3.1 runner on {device}"

        out_dir = Path(SETTINGS["cache_dir"]) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        runner_path = Path(SETTINGS["runner_path"])
        command = [
            str(SETTINGS["runner_python"]),
            str(runner_path),
            "--video",
            str(video_path),
            "--bbox",
            ",".join(f"{value:.3f}" for value in req.bbox),
            "--start-frame",
            str(req.start_frame),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            str(max(0, req.max_frames)),
            "--class-id",
            str(req.class_id),
            "--comfy-root",
            str(SETTINGS["comfy_root"]),
            "--checkpoint",
            str(SETTINGS["checkpoint"]),
            "--device",
            device,
            "--dtype",
            req.dtype or str(SETTINGS["default_dtype"]),
            "--min-mask-area",
            str(max(1, req.min_mask_area)),
        ]
        if req.use_rect_mask:
            command.append("--use-rect-mask")

        job["command"] = command
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(runner_path.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        job["returncode"] = process.returncode
        job["stdout"] = stdout.decode("utf-8", errors="replace")[-8000:]
        job["stderr"] = stderr.decode("utf-8", errors="replace")[-8000:]

        result_path = out_dir / "tracking_results.json"
        job["result_path"] = str(result_path)
        if process.returncode != 0:
            job["status"] = "failed"
            job["message"] = f"SAM3.1 runner failed with exit code {process.returncode}"
            return
        if not result_path.exists():
            job["status"] = "failed"
            job["message"] = "SAM3.1 runner did not write tracking_results.json"
            return

        job["status"] = "done"
        job["message"] = "Done"
    except Exception as exc:
        job["status"] = "failed"
        job["message"] = str(exc)
    finally:
        if device is not None:
            GPU_DEVICE_POOL.release(device)


def health_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "sam31",
        "cache_dir": str(SETTINGS["cache_dir"]),
        "comfy_root": str(SETTINGS["comfy_root"]),
        "checkpoint": str(SETTINGS["checkpoint"]),
        "runner_path": str(SETTINGS["runner_path"]),
        "runner_python": str(SETTINGS["runner_python"]),
        "devices": list(SETTINGS["devices"]),
        "device_pool": GPU_DEVICE_POOL.snapshot(),
        "allowed_roots": [str(root) for root in SETTINGS.get("allowed_roots", [])],
    }


@router.get("/health")
async def health() -> dict[str, Any]:
    return health_payload()


@router.post("/jobs")
async def create_job(req: Sam31JobReq) -> dict[str, Any]:
    video_path = _resolve_video_path(req.video_path)
    _validate_bbox(req.bbox)
    if req.start_frame < 0:
        raise HTTPException(400, "start_frame must be >= 0")
    try:
        GPU_DEVICE_POOL.validate(SETTINGS["devices"], req.device)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "message": "Queued",
        "video_path": str(video_path),
        "start_frame": req.start_frame,
    }
    asyncio.create_task(_run_job(job_id, req, video_path))
    return {"ok": True, "job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    public = dict(job)
    if "command" in public:
        public["command"] = " ".join(public["command"])
    return public


@router.get("/jobs/{job_id}/tracking-results")
async def get_tracking_results(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(400, f"Job is not done: {job.get('status')}")
    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(404, "tracking_results.json not found")
    with open(result_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def configure(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    comfy_root: Path = DEFAULT_COMFY_ROOT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    runner_path: Path = DEFAULT_RUNNER,
    runner_python: str = os.environ.get("SAM31_PYTHON", sys.executable),
    device: str = os.environ.get("SAM31_DEVICES", os.environ.get("SAM31_DEVICE", "cuda")),
    dtype: str = os.environ.get("SAM31_DTYPE", "fp16"),
    allowed_roots: Optional[list[Path]] = None,
) -> None:
    SETTINGS["cache_dir"] = cache_dir.expanduser().resolve()
    SETTINGS["comfy_root"] = comfy_root.expanduser().resolve()
    SETTINGS["checkpoint"] = checkpoint.expanduser().resolve()
    SETTINGS["runner_path"] = runner_path.expanduser().resolve()
    SETTINGS["runner_python"] = runner_python
    SETTINGS["devices"] = parse_devices(device)
    SETTINGS["default_dtype"] = dtype
    if allowed_roots is not None:
        SETTINGS["allowed_roots"] = [path.expanduser().resolve() for path in allowed_roots]
    SETTINGS["cache_dir"].mkdir(parents=True, exist_ok=True)
