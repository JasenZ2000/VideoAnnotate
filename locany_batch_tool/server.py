from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import posixpath
import queue
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCKS: dict[str, threading.RLock] = {}

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
    task_kind: Literal["video", "images"] = "video"


class BatchReq(ConnectionReq):
    input_path: str
    output_path: str
    cuda_devices: list[int] = Field(default_factory=list)
    cuda_device: Optional[int] = None
    dtype: str = "bf16"
    prompt: str = "person"
    categories: list[str] = Field(default_factory=lambda: ["person"])
    class_map: dict[str, int] = Field(default_factory=lambda: {"person": 0})
    task: str = "ground_multi"
    recursive: bool = False
    reuse_uploads: bool = True
    frame_step: int = 1
    max_frames: int = 0
    max_images: int = 0
    copy_images: bool = False


class MotFilterReq(BaseModel):
    input_dir: str
    output_dir: str
    tracking_method: str = "sparse_track"
    iou_match: float = Field(default=0.3, ge=0.0, le=1.0)
    max_missed: int = Field(default=15, ge=0)
    fusion_method: str = "bidirectional_iou_all_pairs"
    iou_fuse: float = Field(default=0.5, ge=0.0, le=1.0)
    min_track_len: int = Field(default=10, ge=1)
    output_voc: bool = False
    voc_output_dir: str = ""
    metadata_path: str = ""


class AdaptiveSampleReq(BaseModel):
    dataset_dir: str
    output_dir: str = ""
    mode: str = "auto"
    intervals: tuple[int, int, int, int] = (20, 10, 5, 2)
    window_frames: int = Field(default=60, gt=1)
    stride_frames: int = Field(default=15, gt=0)
    threshold_quantiles: tuple[float, float, float] = (0.50, 0.75, 0.90)
    motion_fusion_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    dry_run: bool = False


class FrameExtractReq(BaseModel):
    video_path: str
    output_dir: str = ""
    frame_interval: int = Field(default=1, ge=1)
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    overwrite: bool = False


class WorkflowDefaultsReq(BaseModel):
    workspace_dir: str


def run_mot_filter(req: MotFilterReq) -> dict[str, Any]:
    from utils.mot_pipeline.config import DEFAULT_CONFIG, deep_update
    from utils.mot_pipeline.yolo_filter import filter_yolo_annotations

    config = deep_update(DEFAULT_CONFIG, {
        "tracking": {
            "method": req.tracking_method,
            "iou_match": req.iou_match,
            "max_missed": req.max_missed,
        },
        "fusion": {
            "method": req.fusion_method,
            "iou_fuse": req.iou_fuse,
            "min_track_len": req.min_track_len,
        },
    })
    return filter_yolo_annotations(
        req.input_dir,
        req.output_dir,
        config,
        output_voc=req.output_voc,
        voc_output_dir=req.voc_output_dir or None,
        metadata_path=req.metadata_path or None,
    )


def run_adaptive_sample(req: AdaptiveSampleReq) -> dict[str, object]:
    from utils.adaptive_frame_sampler import AdaptiveSamplingConfig, sample_dataset

    config = AdaptiveSamplingConfig(
        mode=req.mode,
        intervals=req.intervals,
        window_frames=req.window_frames,
        stride_frames=req.stride_frames,
        threshold_quantiles=req.threshold_quantiles,
        motion_fusion_weight=req.motion_fusion_weight,
    )
    cache_path = Path(req.dataset_dir).expanduser().resolve() / ".adaptive_phash_cache.npz"
    return sample_dataset(
        dataset_dir=req.dataset_dir,
        output_dir=req.output_dir or None,
        config=config,
        dry_run=req.dry_run,
        phash_cache_path=cache_path,
    )


def run_frame_extract(req: FrameExtractReq) -> dict[str, object]:
    from locany_batch_tool.frame_extract import extract_video_frames

    return extract_video_frames(
        video_path=req.video_path,
        frame_interval=req.frame_interval,
        output_dir=req.output_dir or None,
        jpeg_quality=req.jpeg_quality,
        overwrite=req.overwrite,
    )


def run_workflow_defaults(req: WorkflowDefaultsReq) -> dict[str, str]:
    from locany_batch_tool.workflow_defaults import build_workflow_defaults

    return build_workflow_defaults(req.workspace_dir)


def parse_cuda_devices(value: str) -> list[int]:
    devices: list[int] = []
    for raw in value.replace("，", ",").split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item.startswith("cuda:"):
            item = item[5:]
        if not item.isdigit():
            raise ValueError(f"无效的 CUDA 设备号：{raw.strip()}")
        device = int(item)
        if device < 0 or device > 1024:
            raise ValueError(f"CUDA 设备号超出范围：{device}")
        if device not in devices:
            devices.append(device)
    if not devices:
        raise ValueError("请至少填写一个 CUDA 设备号，例如 0 或 0,1")
    return devices


def _selected_cuda_devices(req: BatchReq) -> list[int]:
    values = req.cuda_devices or ([req.cuda_device] if req.cuda_device is not None else [0])
    return parse_cuda_devices(",".join(str(value) for value in values))


def _job_snapshot(job_id: str) -> dict[str, Any]:
    lock = JOB_LOCKS.setdefault(job_id, threading.RLock())
    with lock:
        return copy.deepcopy(JOBS[job_id])


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
            try:
                sftp.mkdir(current)
            except OSError:
                # Concurrent upload workers may create the same directory between stat and mkdir.
                sftp.stat(current)


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


def _images(input_path: str, recursive: bool) -> list[Path]:
    source = Path(input_path).expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"Image input must be a directory: {source}")
    iterator = source.rglob("*") if recursive else source.iterdir()
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No supported images found in: {source}")
    return images


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


def _check_task_api(server_url: str, task_kind: str = "video") -> None:
    openapi = _json_request("GET", f"{server_url.rstrip('/')}/openapi.json")
    required_path = (
        "/api/locateanything/image-jobs"
        if task_kind == "images"
        else "/api/locateanything/jobs"
    )
    if required_path not in openapi.get("paths", {}):
        raise RuntimeError(
            f"GPU Services is too old for this mode: missing {required_path}. "
            "Update the server repository and restart GPU Services."
        )


def _check_inference_profile(health: dict[str, Any]) -> None:
    actual = (
        health.get("runtime"),
        health.get("generation_mode"),
        health.get("batch_size"),
    )
    if actual != ("batch", "hybrid", 4):
        raise RuntimeError(
            "GPU Services must run the batch-hybrid-4 LocateAnything profile; "
            f"reported runtime={actual[0]!r}, generation_mode={actual[1]!r}, "
            f"batch_size={actual[2]!r}. Update and restart gpu_services."
        )
    if health.get("batch_runtime_supported") is not True:
        raise RuntimeError(
            "GPU Services LocateAnything worker does not support batch runtime. "
            "Deploy the updated locateanything_worker.py and restart gpu_services."
        )
    if health.get("batch_utils_available") is not True or health.get("kernel_utils_available") is not True:
        raise RuntimeError(
            "GPU Services model directory must contain batch_utils and kernel_utils for batch-hybrid-4."
        )


def _check_direct_capabilities(
    server_url: str, health: dict[str, Any], task_kind: str = "video",
) -> None:
    _check_task_api(server_url, task_kind)
    openapi = _json_request("GET", f"{server_url.rstrip('/')}/openapi.json")
    paths = openapi.get("paths", {})
    required_path = (
        "/api/locateanything/image-jobs"
        if task_kind == "images"
        else "/api/locateanything/videos"
    )
    if required_path not in paths:
        raise RuntimeError(
            f"GPU Services is too old for this mode: missing {required_path}. "
            "Update the server repository and restart GPU Services."
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


def _image_remote_root(source: Path, images: list[Path]) -> str:
    manifest = [str(source.resolve())]
    for image in images:
        stat = image.stat()
        manifest.append(f"{image.relative_to(source).as_posix()}|{stat.st_size}|{int(stat.st_mtime)}")
    digest = hashlib.sha1("\n".join(manifest).encode("utf-8")).hexdigest()[:12]
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source.name)
    return f"images_{safe_name or 'input'}_{digest}"


def _upload_image_directory(source: Path, images: list[Path], req: BatchReq) -> str:
    remote_root = posixpath.join(req.sftp_remote_dir.rstrip("/"), _image_remote_root(source, images))
    client = _connect_sftp(req)
    try:
        sftp = client.open_sftp()
        try:
            _mkdir_p(sftp, remote_root)
            for image in images:
                relative = image.relative_to(source).as_posix()
                remote_path = posixpath.join(remote_root, relative)
                _mkdir_p(sftp, posixpath.dirname(remote_path))
                if req.reuse_uploads:
                    try:
                        if sftp.stat(remote_path).st_size == image.stat().st_size:
                            continue
                    except OSError:
                        pass
                sftp.put(str(image), remote_path)
            return remote_root
        finally:
            sftp.close()
    finally:
        client.close()


def _run_video_item(
    job: dict[str, Any], req: BatchReq, item: dict[str, Any], video: Any,
    device: int, output_root: Optional[Path], update_lock: Any,
) -> None:
    base = req.server_url.rstrip("/")
    video_name = video.name if isinstance(video, Path) else PurePosixPath(video).name
    video_stem = video.stem if isinstance(video, Path) else PurePosixPath(video).stem
    with update_lock:
        item.update(status="preparing", message=f"Preparing on cuda:{device}")
    if req.mode == "sftp":
        remote_video = _upload(video, req)
        direct_output = None
    else:
        remote_video = str(video)
        direct_output = posixpath.join(req.output_path.rstrip("/"), video_stem)
    payload = {
        "video_path": remote_video, "prompt": req.prompt,
        "categories": req.categories, "class_map": req.class_map,
        "task": req.task, "class_id": next(iter(req.class_map.values()), 0),
        "device": f"cuda:{device}", "dtype": req.dtype,
        "frame_step": max(1, req.frame_step), "max_frames": max(0, req.max_frames),
        "file_prefix": video_stem, "output_dir": direct_output,
    }
    created = _json_request("POST", f"{base}/api/locateanything/jobs", payload)
    remote_job_id = created["job_id"]
    with update_lock:
        item.update(status="running", remote_job_id=remote_job_id, remote_video=remote_video)
    while True:
        remote = _json_request("GET", f"{base}/api/locateanything/jobs/{remote_job_id}")
        with update_lock:
            item["message"] = remote.get("message", "")
            item["assigned_device"] = remote.get("assigned_device", f"cuda:{device}")
            job["message"] = (
                f"已完成 {job['completed']}/{job['total']}，"
                f"{video_name} 正在 {item['assigned_device']} 运行：{item['message']}"
            )
        if remote.get("status") not in {"queued", "running"}:
            break
        time.sleep(2)
    if remote.get("status") != "done":
        raise RuntimeError(remote.get("message", "remote job failed"))
    if req.mode == "sftp":
        zip_path = output_root / f"{video_stem}_yolo.zip"  # type: ignore[operator]
        _download(f"{base}/api/locateanything/jobs/{remote_job_id}/yolo-zip", zip_path)
        with update_lock:
            item["output"] = str(zip_path)
    else:
        with update_lock:
            item["output"] = remote.get("direct_output_dir", direct_output)
    with update_lock:
        item["status"] = "done"


def _run_image_batch(job_id: str, req: BatchReq) -> None:
    job = JOBS[job_id]
    update_lock = JOB_LOCKS.setdefault(job_id, threading.RLock())
    devices = _selected_cuda_devices(req)
    if len(devices) != 1:
        raise RuntimeError("Image-directory inference currently requires exactly one CUDA device.")
    device = devices[0]
    base = req.server_url.rstrip("/")
    health = _json_request("GET", f"{base}/api/locateanything/health")
    _check_inference_profile(health)
    enabled = [str(value) for value in health.get("devices", [])]
    if enabled and f"cuda:{device}" not in enabled:
        raise RuntimeError(f"GPU Services has not enabled cuda:{device}; available devices: {', '.join(enabled)}")

    if req.mode == "sftp":
        source = Path(req.input_path).expanduser().resolve()
        images = _images(str(source), req.recursive)
        with update_lock:
            job.update(
                status="running", total=1, completed=0, finished=0,
                cuda_devices=devices,
                items=[{"input": str(source), "status": "preparing", "message": f"Uploading {len(images)} images"}],
                message=f"Uploading {len(images)} images to GPU Services",
            )
        remote_input = _upload_image_directory(source, images, req)
        direct_output = None
        local_output = Path(req.output_path).expanduser().resolve()
        result_name = f"{source.name}_images.zip"
    elif req.mode == "direct":
        _check_direct_capabilities(req.server_url, health, "images")
        remote_input = req.input_path
        direct_output = req.output_path
        local_output = None
        result_name = ""
        with update_lock:
            job.update(
                status="running", total=1, completed=0, finished=0,
                cuda_devices=devices,
                items=[{"input": remote_input, "status": "preparing", "message": "Preparing image-directory job"}],
                message="Preparing image-directory inference",
            )
    else:
        raise RuntimeError("mode must be sftp or direct")

    item = job["items"][0]
    payload = {
        "input_dir": remote_input,
        "prompt": req.prompt,
        "categories": req.categories,
        "class_map": req.class_map,
        "task": req.task,
        "class_id": next(iter(req.class_map.values()), 0),
        "device": f"cuda:{device}",
        "dtype": req.dtype,
        "recursive": req.recursive,
        "max_images": max(0, req.max_images),
        "copy_images": req.copy_images,
        "output_dir": direct_output,
    }
    created = _json_request("POST", f"{base}/api/locateanything/image-jobs", payload)
    remote_job_id = created["job_id"]
    with update_lock:
        item.update(status="running", remote_job_id=remote_job_id, requested_device=f"cuda:{device}")
    while True:
        remote = _json_request("GET", f"{base}/api/locateanything/image-jobs/{remote_job_id}")
        with update_lock:
            item["message"] = remote.get("message", "")
            item["assigned_device"] = remote.get("assigned_device", f"cuda:{device}")
            job["message"] = item["message"] or "Running image-directory inference"
        if remote.get("status") not in {"queued", "running"}:
            break
        time.sleep(2)
    if remote.get("status") != "done":
        raise RuntimeError(remote.get("message", "remote image job failed"))
    if req.mode == "sftp":
        zip_path = local_output / result_name  # type: ignore[operator]
        _download(f"{base}/api/locateanything/image-jobs/{remote_job_id}/annotations-zip", zip_path)
        output = str(zip_path)
    else:
        output = remote.get("direct_output_dir", direct_output)
    with update_lock:
        item.update(status="done", output=output)
        job.update(status="done", completed=1, finished=1, message="Image-directory inference completed")


def _run_batch(job_id: str, req: BatchReq) -> None:
    job = JOBS[job_id]
    update_lock = JOB_LOCKS.setdefault(job_id, threading.RLock())
    try:
        if req.task_kind == "images":
            _run_image_batch(job_id, req)
            return
        if req.mode == "sftp":
            videos: list[Any] = _videos(req.input_path, req.recursive)
        elif req.mode == "direct":
            videos = _remote_videos(req.server_url, req.input_path, req.recursive)
        else:
            raise RuntimeError("mode must be sftp or direct")
        devices = _selected_cuda_devices(req)
        health = _json_request("GET", f"{req.server_url.rstrip('/')}/api/locateanything/health")
        if len(devices) > 1 and not health.get("parallel_jobs", False):
            raise RuntimeError(
                "GPU Services 版本过旧，不支持多 GPU 并行调度。"
                "请更新服务器仓库并重启 gpu_services；旧服务会显示 "
                "'Waiting for previous LocateAnything job' 并强制串行。"
            )
        _check_inference_profile(health)
        enabled = [str(device) for device in health.get("devices", [])]
        if enabled:
            requested = [f"cuda:{device}" for device in devices]
            missing = [device for device in requested if device not in enabled]
            if missing:
                raise RuntimeError(
                    f"GPU Services 未启用 {', '.join(missing)}；服务端可用设备：{', '.join(enabled)}"
                )
        items = [
            {"video": str(video), "status": "queued", "message": "等待空闲 GPU"}
            for video in videos
        ]
        with update_lock:
            job.update(
                status="running", total=len(videos), completed=0, finished=0, items=items,
                cuda_devices=devices,
                message=f"使用 {len(devices)} 张 GPU 并行处理 {len(videos)} 个视频",
            )
        output_root = Path(req.output_path).expanduser().resolve() if req.mode == "sftp" else None
        work: queue.Queue[tuple[int, Any]] = queue.Queue()
        for index, video in enumerate(videos):
            work.put((index, video))
        errors: list[str] = []

        def consume(device: int) -> None:
            while True:
                try:
                    index, video = work.get_nowait()
                except queue.Empty:
                    return
                item = items[index]
                with update_lock:
                    item["requested_device"] = f"cuda:{device}"
                try:
                    _run_video_item(job, req, item, video, device, output_root, update_lock)
                    with update_lock:
                        job["completed"] += 1
                except Exception as exc:
                    with update_lock:
                        item.update(status="failed", message=str(exc))
                        errors.append(f"{PurePosixPath(str(video).replace(chr(92), '/')).name}: {exc}")
                finally:
                    with update_lock:
                        job["finished"] += 1
                        job["message"] = f"已处理 {job['finished']}/{job['total']}，成功 {job['completed']}"
                    work.task_done()

        workers = [threading.Thread(target=consume, args=(device,), daemon=True) for device in devices]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            with update_lock:
                job.update(status="failed", message=f"成功 {job['completed']}，失败 {len(errors)}：{errors[0]}")
        else:
            with update_lock:
                job.update(status="done", message=f"已用 {len(devices)} 张 GPU 完成 {len(videos)} 个视频")
    except Exception as exc:
        with update_lock:
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
        _check_inference_profile(gpu)
        result: dict[str, Any] = {"ok": True, "gpu": gpu}
        if req.mode == "sftp":
            _check_task_api(req.server_url, req.task_kind)
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
            _check_direct_capabilities(req.server_url, gpu, req.task_kind)
            result["direct"] = {"ok": True, "output_allowed_roots": gpu.get("output_allowed_roots", [])}
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/jobs")
async def create_batch(req: BatchReq) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued", "completed": 0, "total": 0, "items": []}
    JOB_LOCKS[job_id] = threading.RLock()
    threading.Thread(target=_run_batch, args=(job_id, req), daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.post("/api/mot-filter")
async def mot_filter(req: MotFilterReq) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(run_mot_filter, req)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/adaptive-sample")
async def adaptive_sample(req: AdaptiveSampleReq) -> dict[str, object]:
    try:
        return await asyncio.to_thread(run_adaptive_sample, req)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/extract-frames")
async def extract_frames(req: FrameExtractReq) -> dict[str, object]:
    try:
        return await asyncio.to_thread(run_frame_extract, req)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/workflow-defaults")
async def workflow_defaults(req: WorkflowDefaultsReq) -> dict[str, str]:
    try:
        return await asyncio.to_thread(run_workflow_defaults, req)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/jobs/{job_id}")
async def get_batch(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return _job_snapshot(job_id)


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
