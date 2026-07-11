from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import threading
import webbrowser
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from utils.annotator.server import app as annotator_app
from utils.frame_sampler.server import app as frame_sampler_app


app = FastAPI(title="Video Annotation Workbench")


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/annotator/")


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "service": "video-annotation-workbench",
        "apps": {
            "annotator": "/annotator/",
            "frame_sampler": "/sampler/",
        },
    }


app.mount("/annotator", annotator_app)
app.mount("/sampler", frame_sampler_app)


def _config_path() -> Path:
    base_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
    return base_dir / "local_workbench.env"


def _load_local_config() -> Path:
    config_path = _config_path()
    if not config_path.is_file():
        return config_path
    for raw_line in config_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"LOCAL_WORKBENCH_HOST", "LOCAL_WORKBENCH_PORT"}:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
    return config_path


def _ensure_port_available(host: str, port: int, config_path: Path) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as exc:
        raise SystemExit(
            f"Cannot start: {host}:{port} is already in use.\n"
            f"Set LOCAL_WORKBENCH_PORT to another port in:\n{config_path}\n"
            "For example: LOCAL_WORKBENCH_PORT=17860"
        ) from exc


def main(argv: Optional[list[str]] = None) -> None:
    config_path = _load_local_config()
    parser = argparse.ArgumentParser(description="Run the local video annotation workbench.")
    parser.add_argument("--host", default=os.environ.get("LOCAL_WORKBENCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_WORKBENCH_PORT", "7860")))
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=getattr(sys, "frozen", False),
        help="Open the annotator page after startup (enabled by default in the packaged executable).",
    )
    args = parser.parse_args(argv)
    _ensure_port_available(args.host, args.port, config_path)
    if args.open_browser:
        threading.Timer(
            0.75,
            lambda: webbrowser.open(f"http://{args.host}:{args.port}/annotator/"),
        ).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
