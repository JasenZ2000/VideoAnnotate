from __future__ import annotations

import argparse
import os
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


def main(argv: Optional[list[str]] = None) -> None:
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
    if args.open_browser:
        threading.Timer(
            0.75,
            lambda: webbrowser.open(f"http://{args.host}:{args.port}/annotator/"),
        ).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
