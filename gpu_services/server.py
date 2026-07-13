from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI

from gpu_services import locateanything, sam31


app = FastAPI(title="Video Annotation GPU Service")
app.include_router(sam31.router)
app.include_router(locateanything.router)


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "service": "video-annotation-gpu",
        "sam31": sam31.health_payload(),
        "locateanything": locateanything.health_payload(),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the unified video-annotation GPU service.")
    parser.add_argument("--host", default=os.environ.get("GPU_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GPU_SERVICE_PORT", "9010")))
    parser.add_argument("--sam31-cache-dir", type=Path, default=sam31.DEFAULT_CACHE_DIR)
    parser.add_argument("--sam31-comfy-root", type=Path, default=sam31.DEFAULT_COMFY_ROOT)
    parser.add_argument("--sam31-checkpoint", type=Path, default=sam31.DEFAULT_CHECKPOINT)
    parser.add_argument("--sam31-runner", type=Path, default=sam31.DEFAULT_RUNNER)
    parser.add_argument("--sam31-python", default=os.environ.get("SAM31_PYTHON", os.sys.executable))
    parser.add_argument("--sam31-device", default=os.environ.get("SAM31_DEVICE", "cuda"))
    parser.add_argument("--sam31-dtype", choices=("fp16", "bf16", "fp32"), default=os.environ.get("SAM31_DTYPE", "fp16"))
    parser.add_argument("--sam31-allowed-root", action="append", type=Path)
    parser.add_argument("--locany-cache-dir", type=Path, default=locateanything.DEFAULT_CACHE_DIR)
    parser.add_argument("--locany-model", default=locateanything.DEFAULT_MODEL)
    parser.add_argument("--locateanything-root", default=os.environ.get("LOCATEANYTHING_ROOT", ""))
    parser.add_argument("--locany-device", default=os.environ.get("LOCANY_DEVICE", "cuda"))
    parser.add_argument("--locany-dtype", choices=("bf16", "fp16", "fp32"), default=os.environ.get("LOCANY_DTYPE", "bf16"))
    parser.add_argument("--locany-allowed-root", action="append", type=Path)
    parser.add_argument("--locany-output-allowed-root", action="append", type=Path)
    args = parser.parse_args(argv)

    sam31.configure(
        cache_dir=args.sam31_cache_dir,
        comfy_root=args.sam31_comfy_root,
        checkpoint=args.sam31_checkpoint,
        runner_path=args.sam31_runner,
        runner_python=args.sam31_python,
        device=args.sam31_device,
        dtype=args.sam31_dtype,
        allowed_roots=args.sam31_allowed_root,
    )
    locateanything.configure(
        cache_dir=args.locany_cache_dir,
        model=args.locany_model,
        external_root=args.locateanything_root,
        device=args.locany_device,
        dtype=args.locany_dtype,
        allowed_roots=args.locany_allowed_root,
        output_allowed_roots=args.locany_output_allowed_root,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
