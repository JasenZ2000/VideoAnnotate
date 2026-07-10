from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field


DEFAULT_CACHE_DIR = Path(os.environ.get("LOCANY_CACHE_DIR", "/tmp/video-annotation-locateanything"))
DEFAULT_MODEL = os.environ.get("LOCANY_MODEL", "nvidia/LocateAnything-3B")
DEFAULT_EXTERNAL_ROOT = os.environ.get("LOCATEANYTHING_ROOT", "")

router = APIRouter(prefix="/api/locateanything", tags=["locateanything"])
JOBS: dict[str, dict[str, Any]] = {}
WORKERS: dict[tuple[str, str], Any] = {}
JOB_RUN_LOCK = threading.Lock()
SETTINGS: dict[str, Any] = {
    "cache_dir": DEFAULT_CACHE_DIR,
    "model": DEFAULT_MODEL,
    "external_root": DEFAULT_EXTERNAL_ROOT,
    "default_device": os.environ.get("LOCANY_DEVICE", "cuda"),
    "default_dtype": os.environ.get("LOCANY_DTYPE", "bf16"),
    "allowed_roots": [
        Path(item).expanduser().resolve()
        for item in os.environ.get("LOCANY_ALLOWED_ROOTS", "").split(",")
        if item.strip()
    ],
}


class LocateAnythingVideoReq(BaseModel):
    video_path: str
    prompt: str = "person"
    categories: list[str] = Field(default_factory=list)
    class_map: dict[str, int] = Field(default_factory=dict)
    task: str = "ground_multi"
    question: str = ""
    class_id: int = 0
    score: float = 1.0
    start_frame: int = 0
    max_frames: int = 0
    frame_step: int = 1
    frame_offset: int = 1
    file_prefix: str = ""
    resize_long_edge: int = 1024
    resize_scale: float = 1.0
    generation_mode: str = "slow"
    max_new_tokens: int = 512
    temperature: float = 0.0
    use_cache: bool = True
    device: Optional[str] = None
    dtype: Optional[str] = None


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("LocateAnything requires a CUDA-matched PyTorch installation") from exc


def _external_root() -> Path:
    raw = str(SETTINGS.get("external_root", "")).strip()
    if not raw:
        raise RuntimeError(
            "LOCATEANYTHING_ROOT is required and must point to the directory containing locateanything_worker.py"
        )
    root = Path(raw).expanduser().resolve()
    worker_path = root / "locateanything_worker.py"
    if not worker_path.is_file():
        raise RuntimeError(f"LOCATEANYTHING_ROOT does not contain locateanything_worker.py: {root}")
    return root


def _worker_type() -> Any:
    root = _external_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("locateanything_worker")
    except Exception as exc:
        raise RuntimeError(f"Unable to import LocateAnything worker from {root}: {exc}") from exc
    worker = getattr(module, "LocateAnythingWorker", None)
    if worker is None:
        raise RuntimeError(f"LocateAnythingWorker is missing from {root / 'locateanything_worker.py'}")
    return worker


def _parse_dtype(raw: str) -> Any:
    torch = _torch()
    value = raw.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {raw}")


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


def _ensure_worker(req: LocateAnythingVideoReq) -> Any:
    device = req.device or str(SETTINGS["default_device"])
    if device == "cuda":
        device = "cuda:0"
    dtype_name = req.dtype or str(SETTINGS["default_dtype"])
    key = (device, dtype_name)
    if key not in WORKERS:
        worker_type = _worker_type()
        WORKERS[key] = worker_type(
            str(SETTINGS["model"]),
            device=device,
            dtype=_parse_dtype(dtype_name),
        )
    return WORKERS[key]


def _run_model(worker: Any, image: Image.Image, req: LocateAnythingVideoReq) -> dict[str, Any]:
    common = {
        "generation_mode": req.generation_mode,
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "use_cache": req.use_cache,
        "verbose": False,
    }
    if req.question:
        return worker.predict(image, req.question, **common)
    if req.task == "detect":
        categories = [item.strip() for item in req.categories if item.strip()] or [req.prompt]
        return worker.detect(image, categories, **common)
    if req.task == "ground_single":
        return worker.ground_single(image, req.prompt, **common)
    if req.task == "ground_multi":
        return worker.ground_multi(image, req.prompt, **common)
    raise RuntimeError(f"Unsupported task: {req.task}")


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _normalized_class_map(req: LocateAnythingVideoReq) -> dict[str, int]:
    return {
        normalized: int(class_id)
        for label, class_id in req.class_map.items()
        if (normalized := _normalize_label(str(label)))
    }


def _class_id_for_item(item: dict[str, Any], req: LocateAnythingVideoReq, class_map: dict[str, int]) -> int:
    label = _normalize_label(str(item.get("label", "")))
    if label and label in class_map:
        return class_map[label]
    prompt = _normalize_label(req.prompt)
    if prompt and prompt in class_map:
        return class_map[prompt]
    return int(req.class_id)


def _resize_for_inference(image: Image.Image, long_edge: int, scale: float) -> tuple[Image.Image, float]:
    ratio = min(1.0, float(long_edge) / float(max(image.size))) if long_edge > 0 else float(scale)
    if ratio <= 0:
        raise ValueError("Resize scale must be > 0")
    if ratio >= 1.0:
        return image, 1.0
    size = (max(1, int(round(image.width * ratio))), max(1, int(round(image.height * ratio))))
    return image.resize(size, Image.Resampling.LANCZOS), ratio


def _extract_items(answer: str, image_width: int, image_height: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    box_pattern = re.compile(
        r"(?:<ref>(?P<label>.*?)</ref>)?"
        r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
    )
    for match in box_pattern.finditer(answer):
        x1, y1, x2, y2 = [int(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        items.append(
            {
                "type": "box",
                "label": (match.group("label") or "").strip(),
                "bbox_xyxy": [
                    x1 / 1000 * image_width,
                    y1 / 1000 * image_height,
                    x2 / 1000 * image_width,
                    y2 / 1000 * image_height,
                ],
            }
        )
    return items


def _box_to_yolo_line(box: list[float], width: int, height: int, class_id: int, score: float) -> Optional[str]:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
    y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    box_width, box_height = x2 - x1, y2 - y1
    return (
        f"{class_id} {(x1 + box_width / 2) / width:.6f} {(y1 + box_height / 2) / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f} {score:.6f}"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_zip(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path.resolve():
                archive.write(path, path.relative_to(source_dir))


def _run_job_sync(job_id: str, req: LocateAnythingVideoReq, video_path: Path) -> None:
    job = JOBS[job_id]
    if not JOB_RUN_LOCK.acquire(blocking=False):
        job["status"] = "queued"
        job["message"] = "Waiting for previous LocateAnything job"
        JOB_RUN_LOCK.acquire()
    job["status"] = "running"
    job["message"] = "Loading LocateAnything model"
    out_dir = Path(SETTINGS["cache_dir"]) / job_id
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_answers.jsonl"
    metadata_path = out_dir / "metadata.json"
    zip_path = out_dir / "locateanything_yolo.zip"
    cap: Optional[cv2.VideoCapture] = None
    try:
        worker = _ensure_worker(req)
        torch = _torch()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")

        start_frame = max(0, req.start_frame)
        end_frame = frame_count if req.max_frames <= 0 else min(frame_count, start_frame + req.max_frames)
        frame_step = max(1, req.frame_step)
        prefix = req.file_prefix.strip() or video_path.stem
        metadata = {
            "video_path": str(video_path), "prompt": req.prompt, "categories": req.categories,
            "class_map": req.class_map, "task": req.task, "question": req.question,
            "class_id": req.class_id, "score": req.score, "width": width, "height": height,
            "fps": fps, "frame_count": frame_count, "start_frame": start_frame, "end_frame": end_frame,
            "frame_step": frame_step, "frame_offset": req.frame_offset, "file_prefix": prefix,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        class_map = _normalized_class_map(req)
        processed = failed = 0
        started = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                failed += 1
                break
            frame_id = frame_idx + int(req.frame_offset)
            txt_path = labels_dir / f"{prefix}_{frame_id}.txt"
            if (frame_idx - start_frame) % frame_step != 0:
                _write_text_atomic(txt_path, "")
                continue
            try:
                original = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inference_image, resize_ratio = _resize_for_inference(
                    original, req.resize_long_edge, req.resize_scale
                )
                result = _run_model(worker, inference_image, req)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                answer = str(result.get("answer", ""))
                lines, boxes = [], []
                for item in _extract_items(answer, width, height):
                    box = [float(value) for value in item["bbox_xyxy"]]
                    mapped_class_id = _class_id_for_item(item, req, class_map)
                    line = _box_to_yolo_line(box, width, height, mapped_class_id, req.score)
                    if line is not None:
                        lines.append(line)
                        boxes.append({"label": item.get("label", ""), "class_id": mapped_class_id, "bbox_xyxy": box})
                _write_text_atomic(txt_path, "\n".join(lines) + ("\n" if lines else ""))
                with open(raw_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "frame_idx": frame_idx, "frame_id": frame_id, "txt": txt_path.name,
                        "answer": answer, "num_boxes": len(lines), "boxes": boxes,
                        "inference_image_size": [inference_image.width, inference_image.height],
                        "inference_resize_ratio": resize_ratio,
                    }, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as exc:
                failed += 1
                _write_text_atomic(txt_path, "")
                with open(raw_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"frame_idx": frame_idx, "frame_id": frame_id, "txt": txt_path.name, "error": str(exc)}, ensure_ascii=False) + "\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if processed and processed % 10 == 0:
                elapsed = max(0.001, time.perf_counter() - started)
                job["message"] = f"Processed {processed} frames ({processed / elapsed:.3f} fps)"
                job["processed_frames"], job["failed_frames"] = processed, failed

        _write_zip(out_dir, zip_path)
        job.update({
            "status": "done", "message": f"Done: processed {processed} frames, failed {failed}",
            "processed_frames": processed, "failed_frames": failed, "result_zip_path": str(zip_path),
            "labels_dir": str(labels_dir), "metadata_path": str(metadata_path),
        })
    except Exception as exc:
        job["status"] = "failed"
        job["message"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
        JOB_RUN_LOCK.release()


async def _run_job(job_id: str, req: LocateAnythingVideoReq, video_path: Path) -> None:
    await asyncio.to_thread(_run_job_sync, job_id, req, video_path)


def health_payload() -> dict[str, Any]:
    root_text = str(SETTINGS.get("external_root", "")).strip()
    worker_path = Path(root_text).expanduser() / "locateanything_worker.py" if root_text else None
    try:
        cuda_available = bool(_torch().cuda.is_available())
    except RuntimeError:
        cuda_available = False
    return {
        "ok": True,
        "service": "locateanything",
        "model": str(SETTINGS["model"]),
        "cache_dir": str(SETTINGS["cache_dir"]),
        "external_root": root_text,
        "worker_available": bool(worker_path and worker_path.is_file()),
        "device": str(SETTINGS["default_device"]),
        "dtype": str(SETTINGS["default_dtype"]),
        "model_loaded": bool(WORKERS),
        "loaded_workers": [{"device": device, "dtype": dtype} for device, dtype in sorted(WORKERS)],
        "cuda_available": cuda_available,
        "allowed_roots": [str(root) for root in SETTINGS.get("allowed_roots", [])],
    }


@router.get("/health")
async def health() -> dict[str, Any]:
    return health_payload()


@router.post("/jobs")
async def create_job(req: LocateAnythingVideoReq) -> dict[str, Any]:
    if not req.prompt.strip() and not req.question.strip() and not req.categories:
        raise HTTPException(400, "prompt, categories, or question is required")
    video_path = _resolve_video_path(req.video_path)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "message": "Queued", "video_path": str(video_path),
        "prompt": req.prompt, "categories": req.categories,
    }
    asyncio.create_task(_run_job(job_id, req, video_path))
    return {"ok": True, "job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    public = dict(job)
    public.pop("result_zip_path", None)
    return public


@router.get("/jobs/{job_id}/yolo-zip")
async def get_yolo_zip(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(400, f"Job is not done: {job.get('status')}")
    zip_path = Path(job["result_zip_path"])
    if not zip_path.exists():
        raise HTTPException(404, "Result zip not found")
    return FileResponse(str(zip_path), media_type="application/zip", filename="locateanything_yolo.zip")


def configure(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model: str = DEFAULT_MODEL,
    external_root: str = DEFAULT_EXTERNAL_ROOT,
    device: str = os.environ.get("LOCANY_DEVICE", "cuda"),
    dtype: str = os.environ.get("LOCANY_DTYPE", "bf16"),
    allowed_roots: Optional[list[Path]] = None,
) -> None:
    SETTINGS["cache_dir"] = cache_dir.expanduser().resolve()
    SETTINGS["model"] = model
    SETTINGS["external_root"] = external_root
    SETTINGS["default_device"] = device
    SETTINGS["default_dtype"] = dtype
    if allowed_roots is not None:
        SETTINGS["allowed_roots"] = [path.expanduser().resolve() for path in allowed_roots]
    SETTINGS["cache_dir"].mkdir(parents=True, exist_ok=True)
    WORKERS.clear()
