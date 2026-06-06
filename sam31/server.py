from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


DEFAULT_COMFY_ROOT = Path(os.environ.get("SAM31_COMFY_ROOT", "/data2/DET_Group/ZZS/generate/update/ComfyUI"))
DEFAULT_CHECKPOINT = Path(os.environ.get("SAM31_CHECKPOINT", "/data2/DET_Group/ZZS/my_sam3/sam3.1_multiplex_fp16.safetensors"))
DEFAULT_CACHE_DIR = Path(os.environ.get("SAM31_CACHE_DIR", "/tmp/object-reid-sam31"))

app = FastAPI(title="SAM31 Job Server")
JOBS: dict[str, dict[str, Any]] = {}
SETTINGS: dict[str, Any] = {
    "comfy_root": DEFAULT_COMFY_ROOT,
    "checkpoint": DEFAULT_CHECKPOINT,
    "cache_dir": DEFAULT_CACHE_DIR,
    "default_device": os.environ.get("SAM31_DEVICE", "cuda"),
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


def _resolve_video_path(raw_path: str) -> Path:
    if "\\" in raw_path or (len(raw_path) >= 2 and raw_path[1] == ":"):
        raise HTTPException(
            400,
            "Received a local Windows path. Configure path mapping or SFTP upload on the annotator side.",
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(400, f"Video does not exist: {path}")

    allowed_roots: list[Path] = SETTINGS.get("allowed_roots", [])
    if allowed_roots:
        if not any(_is_relative_to(path, root) for root in allowed_roots):
            roots = ", ".join(str(root) for root in allowed_roots)
            raise HTTPException(403, f"Video path is outside allowed roots: {roots}")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_bbox(bbox: list[float]) -> None:
    if len(bbox) != 4:
        raise HTTPException(400, "bbox must contain x1,y1,x2,y2")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise HTTPException(400, "bbox must satisfy x2>x1 and y2>y1")


async def _run_job(job_id: str, req: Sam31JobReq, video_path: Path) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    job["message"] = "Starting sam31_track.py"

    out_dir = Path(SETTINGS["cache_dir"]) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "sam31_track.py"
    command = [
        sys.executable,
        str(script),
        "--video",
        str(video_path),
        "--bbox",
        ",".join(f"{v:.3f}" for v in req.bbox),
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
        req.device or str(SETTINGS["default_device"]),
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
        cwd=str(script.parent),
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
        job["message"] = f"sam31_track.py failed with exit code {process.returncode}"
        return
    if not result_path.exists():
        job["status"] = "failed"
        job["message"] = "sam31_track.py did not write tracking_results.json"
        return

    job["status"] = "done"
    job["message"] = "Done"


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "cache_dir": str(SETTINGS["cache_dir"]),
        "comfy_root": str(SETTINGS["comfy_root"]),
        "checkpoint": str(SETTINGS["checkpoint"]),
        "allowed_roots": [str(root) for root in SETTINGS.get("allowed_roots", [])],
    }


@app.post("/api/jobs")
async def create_job(req: Sam31JobReq):
    video_path = _resolve_video_path(req.video_path)
    _validate_bbox(req.bbox)
    if req.start_frame < 0:
        raise HTTPException(400, "start_frame must be >= 0")

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


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    public = dict(job)
    if "command" in public:
        public["command"] = " ".join(public["command"])
    return public


@app.get("/api/jobs/{job_id}/tracking-results")
async def get_tracking_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(400, f"Job is not done: {job.get('status')}")
    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(404, "tracking_results.json not found")
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SAM31 job server.")
    parser.add_argument("--host", default=os.environ.get("SAM31_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SAM31_PORT", "9001")))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=os.environ.get("SAM31_DEVICE", "cuda"))
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default=os.environ.get("SAM31_DTYPE", "fp16"))
    parser.add_argument(
        "--allowed-root",
        action="append",
        type=Path,
        default=[],
        help="Allowed video root. Repeat to allow multiple roots.",
    )
    args = parser.parse_args(argv)

    SETTINGS["cache_dir"] = args.cache_dir.expanduser().resolve()
    SETTINGS["comfy_root"] = args.comfy_root.expanduser().resolve()
    SETTINGS["checkpoint"] = args.checkpoint.expanduser().resolve()
    SETTINGS["default_device"] = args.device
    SETTINGS["default_dtype"] = args.dtype
    if args.allowed_root:
        SETTINGS["allowed_roots"] = [path.expanduser().resolve() for path in args.allowed_root]
    SETTINGS["cache_dir"].mkdir(parents=True, exist_ok=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
