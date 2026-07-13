from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="LocateAnything Batch Tool")


def _load_local_env() -> None:
    path = APP_DIR.parent / ".env.local"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_local_env()


class ConnectionReq(BaseModel):
    server_url: str
    mode: str = "sftp"
    sftp_host: str = ""
    sftp_port: int = 22
    sftp_username: str = ""
    sftp_password: str = ""
    sftp_key_path: str = ""
    sftp_remote_dir: str = ""


class BatchReq(ConnectionReq):
    input_path: str
    output_path: str
    cuda_device: int = 0
    dtype: str = "bf16"
    prompt: str = "person"
    categories: list[str] = Field(default_factory=lambda: ["person"])
    class_map: dict[str, int] = Field(default_factory=lambda: {"person": 0})
    task: str = "ground_multi"
    recursive: bool = False
    reuse_uploads: bool = True
    frame_step: int = 1
    max_frames: int = 0


def _json_request(method: str, url: str, payload: Optional[dict[str, Any]] = None, timeout: float = 30) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def _connect_sftp(req: ConnectionReq):
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("SFTP requires paramiko") from exc
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict[str, Any] = {
        "hostname": req.sftp_host, "port": req.sftp_port,
        "username": req.sftp_username, "timeout": 20,
    }
    password = req.sftp_password or os.environ.get("LOCANY_SFTP_PASSWORD", "")
    if password:
        kwargs["password"] = password
    if req.sftp_key_path:
        kwargs["key_filename"] = req.sftp_key_path
    client.connect(**kwargs)
    return client


def _mkdir_p(sftp: Any, remote_dir: str) -> None:
    current = "/" if remote_dir.startswith("/") else "."
    for part in [item for item in remote_dir.replace("\\", "/").split("/") if item]:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def _videos(input_path: str, recursive: bool) -> list[Path]:
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video: {source}")
        return [source]
    if not source.is_dir():
        raise RuntimeError(f"Input path does not exist: {source}")
    iterator = source.rglob("*") if recursive else source.iterdir()
    videos = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise RuntimeError(f"No videos found in: {source}")
    return videos


def _remote_videos(server_url: str, input_path: str, recursive: bool) -> list[str]:
    query = urllib.parse.urlencode({"path": input_path, "recursive": str(recursive).lower()})
    try:
        payload = _json_request("GET", f"{server_url.rstrip('/')}/api/locateanything/videos?{query}")
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            raise RuntimeError(
                "GPU Services 版本过旧：缺少 /api/locateanything/videos。"
                "请更新服务器上的 gpu_services 代码并重启服务。"
            ) from exc
        raise
    videos = [str(path) for path in payload.get("videos", [])]
    if not videos:
        raise RuntimeError(f"GPU server found no videos in: {input_path}")
    return videos


def _check_direct_capabilities(server_url: str, health: dict[str, Any]) -> None:
    openapi = _json_request("GET", f"{server_url.rstrip('/')}/openapi.json")
    paths = openapi.get("paths", {})
    if "/api/locateanything/videos" not in paths:
        raise RuntimeError(
            "GPU Services 版本过旧：直连模式需要 /api/locateanything/videos。"
            "请更新服务器代码并重启 GPU Services。"
        )
    output_roots = health.get("output_allowed_roots", [])
    if not output_roots:
        raise RuntimeError(
            "GPU Services 未配置直连输出目录。请设置 LOCANY_OUTPUT_ALLOWED_ROOTS 并重启服务。"
        )


def _remote_name(video: Path) -> str:
    stat = video.stat()
    digest = hashlib.sha1(f"{video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:12]
    return f"{video.stem}_{digest}{video.suffix.lower()}"


def _upload(video: Path, req: BatchReq) -> str:
    client = _connect_sftp(req)
    try:
        sftp = client.open_sftp()
        try:
            _mkdir_p(sftp, req.sftp_remote_dir)
            remote_path = posixpath.join(req.sftp_remote_dir.rstrip("/"), _remote_name(video))
            if req.reuse_uploads:
                try:
                    if sftp.stat(remote_path).st_size == video.stat().st_size:
                        return remote_path
                except OSError:
                    pass
            sftp.put(str(video), remote_path)
            return remote_path
        finally:
            sftp.close()
    finally:
        client.close()


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response, open(output, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _run_batch(job_id: str, req: BatchReq) -> None:
    job = JOBS[job_id]
    try:
        videos: list[Any]
        if req.mode == "sftp":
            videos = _videos(req.input_path, req.recursive)
        elif req.mode == "direct":
            videos = _remote_videos(req.server_url, req.input_path, req.recursive)
        else:
            raise RuntimeError("mode must be sftp or direct")
        job.update(status="running", total=len(videos), completed=0, items=[])
        base = req.server_url.rstrip("/")
        output_root = Path(req.output_path).expanduser().resolve() if req.mode == "sftp" else None
        for index, video in enumerate(videos, 1):
            video_name = video.name if isinstance(video, Path) else PurePosixPath(video).name
            video_stem = video.stem if isinstance(video, Path) else PurePosixPath(video).stem
            item: dict[str, Any] = {"video": str(video), "status": "preparing"}
            job["items"].append(item)
            job["message"] = f"[{index}/{len(videos)}] Preparing {video_name}"
            if req.mode == "sftp":
                remote_video = _upload(video, req)
                direct_output = None
            elif req.mode == "direct":
                remote_video = str(video)
                direct_output = posixpath.join(req.output_path.rstrip("/"), video_stem)
            payload = {
                "video_path": remote_video, "prompt": req.prompt,
                "categories": req.categories, "class_map": req.class_map,
                "task": req.task, "class_id": next(iter(req.class_map.values()), 0),
                "device": f"cuda:{req.cuda_device}", "dtype": req.dtype,
                "frame_step": max(1, req.frame_step), "max_frames": max(0, req.max_frames),
                "file_prefix": video_stem, "output_dir": direct_output,
            }
            created = _json_request("POST", f"{base}/api/locateanything/jobs", payload)
            remote_job_id = created["job_id"]
            item.update(status="running", remote_job_id=remote_job_id, remote_video=remote_video)
            while True:
                remote = _json_request("GET", f"{base}/api/locateanything/jobs/{remote_job_id}")
                item["message"] = remote.get("message", "")
                job["message"] = f"[{index}/{len(videos)}] {video.name}: {item['message']}"
                if remote.get("status") not in {"queued", "running"}:
                    break
                time.sleep(2)
            if remote.get("status") != "done":
                raise RuntimeError(f"{video_name}: {remote.get('message', 'remote job failed')}")
            if req.mode == "sftp":
                zip_path = output_root / f"{video_stem}_yolo.zip"  # type: ignore[operator]
                _download(f"{base}/api/locateanything/jobs/{remote_job_id}/yolo-zip", zip_path)
                item["output"] = str(zip_path)
            else:
                item["output"] = remote.get("direct_output_dir", direct_output)
            item["status"] = "done"
            job["completed"] = index
        job.update(status="done", message=f"Completed {len(videos)} video(s)")
    except Exception as exc:
        job.update(status="failed", message=str(exc))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "locateanything-batch-tool"}


@app.post("/api/test")
async def test_connection(req: ConnectionReq) -> dict[str, Any]:
    try:
        gpu = _json_request("GET", f"{req.server_url.rstrip('/')}/api/locateanything/health")
        result: dict[str, Any] = {"ok": True, "gpu": gpu}
        if req.mode == "sftp":
            client = _connect_sftp(req)
            try:
                sftp = client.open_sftp()
                try:
                    result["sftp"] = {"ok": True, "remote_dir_exists": bool(sftp.stat(req.sftp_remote_dir))}
                finally:
                    sftp.close()
            finally:
                client.close()
        else:
            _check_direct_capabilities(req.server_url, gpu)
            result["direct"] = {"ok": True, "output_allowed_roots": gpu.get("output_allowed_roots", [])}
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/jobs")
async def create_batch(req: BatchReq) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued", "completed": 0, "total": 0, "items": []}
    threading.Thread(target=_run_batch, args=(job_id, req), daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_batch(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the standalone LocateAnything batch tool")
    parser.add_argument("--host", default=os.environ.get("LOCANY_TOOL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCANY_TOOL_PORT", "7870")))
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{args.host}:{args.port}/")).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
