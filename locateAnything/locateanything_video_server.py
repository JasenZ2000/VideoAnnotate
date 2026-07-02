from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

import cv2
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from batch_yolo_infer import box_to_yolo_line, write_text_atomic
from debug_infer import extract_items, parse_dtype, resize_for_inference
from locateanything_worker import LocateAnythingWorker


DEFAULT_CACHE_DIR = Path(os.environ.get("LOCANY_CACHE_DIR", "/tmp/object-reid-locateanything"))
DEFAULT_MODEL = os.environ.get("LOCANY_MODEL", "nvidia/LocateAnything-3B")

app = FastAPI(title="LocateAnything Video Job Server")
JOBS: dict[str, dict[str, Any]] = {}
WORKER: Optional[LocateAnythingWorker] = None
JOB_RUN_LOCK = threading.Lock()
SETTINGS: dict[str, Any] = {
    "cache_dir": DEFAULT_CACHE_DIR,
    "model": DEFAULT_MODEL,
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
            "Received a local Windows path. Configure path mapping or SFTP upload on the annotator side.",
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(400, f"Video does not exist: {path}")
    allowed_roots: list[Path] = SETTINGS.get("allowed_roots", [])
    if allowed_roots and not any(_is_relative_to(path, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise HTTPException(403, f"Video path is outside allowed roots: {roots}")
    return path


def _ensure_worker(req: LocateAnythingVideoReq) -> LocateAnythingWorker:
    global WORKER
    if WORKER is None:
        dtype = parse_dtype(req.dtype or str(SETTINGS["default_dtype"]))
        WORKER = LocateAnythingWorker(
            str(SETTINGS["model"]),
            device=req.device or str(SETTINGS["default_device"]),
            dtype=dtype,
        )
    return WORKER


def _run_model(worker: LocateAnythingWorker, image: Image.Image, req: LocateAnythingVideoReq) -> dict[str, Any]:
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
        categories = [item.strip() for item in req.categories if item.strip()]
        if not categories:
            categories = [req.prompt]
        return worker.detect(image, categories, **common)
    if req.task == "ground_single":
        return worker.ground_single(image, req.prompt, **common)
    if req.task == "ground_multi":
        return worker.ground_multi(image, req.prompt, **common)
    raise RuntimeError(f"Unsupported task: {req.task}")


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _normalized_class_map(req: LocateAnythingVideoReq) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for label, class_id in req.class_map.items():
        normalized = _normalize_label(str(label))
        if normalized:
            mapped[normalized] = int(class_id)
    return mapped


def _class_id_for_item(item: dict[str, Any], req: LocateAnythingVideoReq, class_map: dict[str, int]) -> int:
    label = _normalize_label(str(item.get("label", "")))
    if label and label in class_map:
        return class_map[label]
    prompt = _normalize_label(req.prompt)
    if prompt and prompt in class_map:
        return class_map[prompt]
    return int(req.class_id)


def _frame_to_image(frame: Any) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _write_zip(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path.resolve():
                zf.write(path, path.relative_to(source_dir))


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
        try:
            worker = _ensure_worker(req)
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
                "video_path": str(video_path),
                "prompt": req.prompt,
                "categories": req.categories,
                "class_map": req.class_map,
                "task": req.task,
                "question": req.question,
                "class_id": req.class_id,
                "score": req.score,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_step": frame_step,
                "frame_offset": req.frame_offset,
                "file_prefix": prefix,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            class_map = _normalized_class_map(req)
            processed = 0
            failed = 0
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
                    write_text_atomic(txt_path, "")
                    continue

                try:
                    original = _frame_to_image(frame)
                    inference_image, resize_ratio = resize_for_inference(
                        original,
                        long_edge=req.resize_long_edge,
                        scale=req.resize_scale,
                    )
                    result = _run_model(worker, inference_image, req)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    answer = str(result.get("answer", ""))
                    items = extract_items(answer, width, height)
                    lines = []
                    boxes = []
                    for item in items:
                        if item.get("type") != "box":
                            continue
                        box = [float(value) for value in item["bbox_xyxy"]]
                        mapped_class_id = _class_id_for_item(item, req, class_map)
                        line = box_to_yolo_line(box, width, height, mapped_class_id, req.score)
                        if line is None:
                            continue
                        lines.append(line)
                        boxes.append({
                            "label": item.get("label", ""),
                            "class_id": mapped_class_id,
                            "bbox_xyxy": box,
                        })
                    write_text_atomic(txt_path, "\n".join(lines) + ("\n" if lines else ""))
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "frame_idx": frame_idx,
                            "frame_id": frame_id,
                            "txt": txt_path.name,
                            "answer": answer,
                            "num_boxes": len(lines),
                            "boxes": boxes,
                            "inference_image_size": [inference_image.width, inference_image.height],
                            "inference_resize_ratio": resize_ratio,
                        }, ensure_ascii=False) + "\n")
                    processed += 1
                except Exception as exc:
                    failed += 1
                    write_text_atomic(txt_path, "")
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "frame_idx": frame_idx,
                            "frame_id": frame_id,
                            "txt": txt_path.name,
                            "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if processed and processed % 10 == 0:
                    elapsed = max(0.001, time.perf_counter() - started)
                    job["message"] = f"Processed {processed} frames ({processed / elapsed:.3f} fps)"
                    job["processed_frames"] = processed
                    job["failed_frames"] = failed

            _write_zip(out_dir, zip_path)
            job["status"] = "done"
            job["message"] = f"Done: processed {processed} frames, failed {failed}"
            job["processed_frames"] = processed
            job["failed_frames"] = failed
            job["result_zip_path"] = str(zip_path)
            job["labels_dir"] = str(labels_dir)
            job["metadata_path"] = str(metadata_path)
        except Exception as exc:
            job["status"] = "failed"
            job["message"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
        JOB_RUN_LOCK.release()


async def _run_job(job_id: str, req: LocateAnythingVideoReq, video_path: Path) -> None:
    await asyncio.to_thread(_run_job_sync, job_id, req, video_path)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": str(SETTINGS["model"]),
        "cache_dir": str(SETTINGS["cache_dir"]),
        "device": str(SETTINGS["default_device"]),
        "dtype": str(SETTINGS["default_dtype"]),
        "model_loaded": WORKER is not None,
        "cuda_available": torch.cuda.is_available(),
        "allowed_roots": [str(root) for root in SETTINGS.get("allowed_roots", [])],
    }


@app.post("/api/jobs")
async def create_job(req: LocateAnythingVideoReq) -> dict[str, Any]:
    if not req.prompt.strip() and not req.question.strip() and not req.categories:
        raise HTTPException(400, "prompt, categories, or question is required")
    video_path = _resolve_video_path(req.video_path)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "message": "Queued",
        "video_path": str(video_path),
        "prompt": req.prompt,
        "categories": req.categories,
    }
    asyncio.create_task(_run_job(job_id, req, video_path))
    return {"ok": True, "job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    public = dict(job)
    public.pop("result_zip_path", None)
    return public


@app.get("/api/jobs/{job_id}/yolo-zip")
async def get_yolo_zip(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(400, f"Job is not done: {job.get('status')}")
    zip_path = Path(job["result_zip_path"])
    if not zip_path.exists():
        raise HTTPException(404, "Result zip not found")
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename="locateanything_yolo.zip",
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the LocateAnything video job server.")
    parser.add_argument("--host", default=os.environ.get("LOCANY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCANY_PORT", "9011")))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=os.environ.get("LOCANY_DEVICE", "cuda"))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=os.environ.get("LOCANY_DTYPE", "bf16"))
    parser.add_argument("--allowed-root", action="append", type=Path, default=[])
    args = parser.parse_args(argv)

    SETTINGS["cache_dir"] = args.cache_dir.expanduser().resolve()
    SETTINGS["model"] = args.model
    SETTINGS["default_device"] = args.device
    SETTINGS["default_dtype"] = args.dtype
    if args.allowed_root:
        SETTINGS["allowed_roots"] = [path.expanduser().resolve() for path in args.allowed_root]
    SETTINGS["cache_dir"].mkdir(parents=True, exist_ok=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
