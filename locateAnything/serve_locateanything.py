from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from debug_infer import extract_items, parse_dtype
from locateanything_worker import LocateAnythingWorker


app = FastAPI(title="LocateAnything Debug Server")
WORKER: Optional[LocateAnythingWorker] = None
SETTINGS: dict[str, Any] = {}


class GroundReq(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    task: str = "ground_multi"
    prompt: str = ""
    categories: list[str] = []
    question: str = ""
    output_type: str = "box"
    generation_mode: str = "hybrid"
    max_new_tokens: int = 1024
    temperature: float = 0.0


def decode_image(req: GroundReq) -> Image.Image:
    if req.image_path:
        path = Path(req.image_path).expanduser().resolve()
        allowed_roots = SETTINGS.get("allowed_roots") or []
        if allowed_roots and not any(is_relative_to(path, root) for root in allowed_roots):
            raise HTTPException(403, "image_path is outside allowed roots")
        if not path.is_file():
            raise HTTPException(400, f"image_path does not exist: {path}")
        return Image.open(path).convert("RGB")
    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(400, f"invalid image_base64: {exc}") from exc
    raise HTTPException(400, "Provide image_path or image_base64")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_worker() -> LocateAnythingWorker:
    if WORKER is None:
        raise HTTPException(503, "Model is not loaded")
    return WORKER


def run_request(worker: LocateAnythingWorker, image: Image.Image, req: GroundReq) -> dict[str, Any]:
    common = {
        "generation_mode": req.generation_mode,
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "verbose": True,
    }
    if req.question:
        return worker.predict(image, req.question, **common)
    if req.task == "detect":
        categories = req.categories or [item.strip() for item in req.prompt.split(",") if item.strip()]
        if not categories:
            raise HTTPException(400, "categories or comma-separated prompt is required for detect")
        return worker.detect(image, categories, **common)
    if req.task == "ground_single":
        return worker.ground_single(image, req.prompt, **common)
    if req.task == "ground_multi":
        return worker.ground_multi(image, req.prompt, **common)
    if req.task == "ground_text":
        return worker.ground_text(image, req.prompt, **common)
    if req.task == "detect_text":
        return worker.detect_text(image, **common)
    if req.task == "ground_gui":
        return worker.ground_gui(image, req.prompt, output_type=req.output_type, **common)
    if req.task == "point":
        return worker.point(image, req.prompt, **common)
    raise HTTPException(400, f"Unsupported task: {req.task}")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": WORKER is not None,
        "model": SETTINGS.get("model"),
        "device": SETTINGS.get("device"),
        "dtype": SETTINGS.get("dtype"),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "allowed_roots": [str(root) for root in SETTINGS.get("allowed_roots", [])],
    }


@app.post("/api/ground")
async def ground(req: GroundReq) -> dict[str, Any]:
    worker = ensure_worker()
    image = decode_image(req)
    start = time.perf_counter()
    result = run_request(worker, image, req)
    if str(SETTINGS.get("device", "")).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    answer = str(result.get("answer", ""))
    return {
        "ok": True,
        "image_size": [image.width, image.height],
        "answer": answer,
        "items": extract_items(answer, image.width, image.height),
        "stats": result.get("stats"),
        "infer_seconds": elapsed,
    }


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "LocateAnything remote processor code uses Python 3.10+ type syntax. "
            f"Current Python is {sys.version.split()[0]}; please use Python 3.10 or newer."
        )

    parser = argparse.ArgumentParser(description="Run a simple LocateAnything HTTP debug server.")
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9011)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--allowed-root", action="append", type=Path, default=[])
    args = parser.parse_args()

    global WORKER
    SETTINGS.update(
        {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "allowed_roots": [root.expanduser().resolve() for root in args.allowed_root],
        }
    )
    WORKER = LocateAnythingWorker(args.model, device=args.device, dtype=parse_dtype(args.dtype))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
