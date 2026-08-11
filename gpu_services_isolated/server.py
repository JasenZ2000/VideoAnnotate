from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI

from gpu_services_isolated import locateanything
from gpu_services_isolated.worker_process import WORKER_MANAGER


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("LOCANY_PRELOAD_WORKERS", "1") == "1":
        await asyncio.to_thread(locateanything.preload_workers)
    try:
        yield
    finally:
        await asyncio.to_thread(WORKER_MANAGER.close_all)


app = FastAPI(title="Process-Isolated LocateAnything GPU Service", lifespan=lifespan)
app.include_router(locateanything.router)


@app.get("/api/health")
async def health() -> dict[str, object]:
    locate_health = locateanything.health_payload()
    return {
        "ok": bool(locate_health["ok"]),
        "service": "video-annotation-gpu-isolated",
        "locateanything": locate_health,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run LocateAnything with one independent model process per CUDA device."
    )
    parser.add_argument("--host", default=os.environ.get("GPU_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GPU_SERVICE_PORT", "10115")))
    parser.add_argument("--locany-cache-dir", type=Path, default=locateanything.DEFAULT_CACHE_DIR)
    parser.add_argument("--locany-model", default=locateanything.DEFAULT_MODEL)
    parser.add_argument("--locateanything-root", default=os.environ.get("LOCATEANYTHING_ROOT", ""))
    parser.add_argument(
        "--locany-devices", "--locany-device", dest="locany_devices",
        default=os.environ.get("LOCANY_DEVICES", os.environ.get("LOCANY_DEVICE", "cuda")),
        help="Comma-separated CUDA devices. Each receives its own spawned model process.",
    )
    parser.add_argument(
        "--locany-dtype", choices=("bf16", "fp16", "fp32"),
        default=os.environ.get("LOCANY_DTYPE", "bf16"),
    )
    parser.add_argument("--locany-allowed-root", action="append", type=Path)
    parser.add_argument("--locany-output-allowed-root", action="append", type=Path)
    parser.add_argument(
        "--no-preload", action="store_true",
        help="Start HTTP immediately and load each GPU lazily on its first job.",
    )
    args = parser.parse_args(argv)

    if args.no_preload:
        os.environ["LOCANY_PRELOAD_WORKERS"] = "0"
    locateanything.configure(
        cache_dir=args.locany_cache_dir,
        model=args.locany_model,
        external_root=args.locateanything_root,
        device=args.locany_devices,
        dtype=args.locany_dtype,
        allowed_roots=args.locany_allowed_root,
        output_allowed_roots=args.locany_output_allowed_root,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
