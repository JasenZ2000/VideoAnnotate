from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import hmac
import json
import os
import posixpath
import secrets
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.annotator.state import AnnotationState
from utils.mot_pipeline.config import deep_update, load_config
from utils.mot_pipeline.pipeline import run_pipeline
from utils.mot_pipeline.utils.converters import export_tracking_results_to_yolo
from workflow_platform.database import DEFAULT_STAGE_NAMES, PlatformDatabase

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PROJECT_ROOT = APP_DIR.parent
DEFAULT_TASKS_DIR = PROJECT_ROOT / "platform_tasks"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
API_SCHEMA_VERSION = 4
SESSION_COOKIE_NAME = "annotation_platform_session"
SESSION_TTL_DAYS = int(os.environ.get("ANNOTATION_PLATFORM_SESSION_DAYS", "7"))
PASSWORD_ITERATIONS = 200_000

app = FastAPI(title="Annotation Workflow Platform")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

TASK_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
SETTINGS: dict[str, Any] = {
    "tasks_dir": Path(os.environ.get("ANNOTATION_PLATFORM_TASKS_DIR", DEFAULT_TASKS_DIR)).resolve(),
    "project_config": Path(
        os.environ.get("ANNOTATION_PLATFORM_CONFIG", PROJECT_ROOT / "config.json")
    ).resolve(),
    "database_path": os.environ.get("ANNOTATION_PLATFORM_DB", ""),
}
_DATABASE: Optional[PlatformDatabase] = None
_DATABASE_KEY = ""
_DATABASE_LOCK = threading.Lock()
CURRENT_USER: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar("workflow_platform_current_user", default=None)
REQUEST_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("workflow_platform_request_active", default=False)


@app.middleware("http")
async def auth_session_middleware(request: Request, call_next):
    request_token = REQUEST_ACTIVE.set(True)
    user_token = CURRENT_USER.set(None)
    try:
        raw_token = request.cookies.get(SESSION_COOKIE_NAME, "")
        if raw_token:
            session_user = database().get_session_user(session_token_hash(raw_token), now_iso())
            if session_user is not None:
                CURRENT_USER.set(auth_public_user(session_user))
        path = request.url.path
        if (
            path.startswith("/api/")
            and path not in {"/api/health", "/api/auth/me", "/api/auth/login", "/api/auth/bootstrap-admin"}
            and not path.startswith("/api/auth/")
            and CURRENT_USER.get() is None
        ):
            return JSONResponse(
                {"detail": "Authentication required", "bootstrap_required": database().user_count() == 0},
                status_code=401,
            )
        response = await call_next(request)
        return response
    finally:
        CURRENT_USER.reset(user_token)
        REQUEST_ACTIVE.reset(request_token)


class CreateTaskReq(BaseModel):
    name: str
    task_type: str = "general"  # general | video_detection
    publisher: str = ""
    manager: str = ""
    annotators: str = ""
    instructions: str = ""
    part_count: int = 0
    part_name_prefix: str = "Part"
    prelabel_source: str = "none"  # none | yolo | locateanything
    prompt: str = "person"
    classes: str = "0 person"
    assignee: str = ""
    notes: str = ""


class RunLocateAnythingReq(BaseModel):
    actor: str = ""
    prompt: Optional[str] = None
    video_id: Optional[str] = None
    max_frames: int = 0
    start_frame: int = 0
    frame_step: int = 1
    cuda_device: Optional[int] = None
    server_url: Optional[str] = None
    video_transfer: Optional[str] = None
    local_path_prefix: Optional[str] = None
    remote_path_prefix: Optional[str] = None
    sftp_host: Optional[str] = None
    sftp_port: Optional[int] = None
    sftp_username: Optional[str] = None
    sftp_password: Optional[str] = None
    sftp_password_env: Optional[str] = None
    sftp_key_path: Optional[str] = None
    sftp_remote_dir: Optional[str] = None
    sftp_reuse_existing: Optional[bool] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    request_timeout: Optional[float] = None
    download_timeout: Optional[float] = None
    poll_interval: Optional[float] = None


class LocateAnythingSettingsReq(BaseModel):
    actor: str = ""
    prompt: Optional[str] = None
    classes: Optional[str] = None
    server_url: Optional[str] = None
    video_transfer: Optional[str] = None
    local_path_prefix: Optional[str] = None
    remote_path_prefix: Optional[str] = None
    sftp_host: Optional[str] = None
    sftp_port: Optional[int] = None
    sftp_username: Optional[str] = None
    sftp_password: Optional[str] = None
    sftp_password_env: Optional[str] = None
    sftp_key_path: Optional[str] = None
    sftp_remote_dir: Optional[str] = None
    sftp_reuse_existing: Optional[bool] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    request_timeout: Optional[float] = None
    download_timeout: Optional[float] = None
    poll_interval: Optional[float] = None


class RunTrackingReq(BaseModel):
    actor: str = ""
    label_source: str = "auto"  # auto | input | locateanything
    short_gap_max: int = 5


class PackageReq(BaseModel):
    actor: str = ""
    include_video: bool = True


class UpdateTaskReq(BaseModel):
    actor: str = ""
    name: Optional[str] = None
    assignee: Optional[str] = None
    notes: Optional[str] = None
    prompt: Optional[str] = None
    classes: Optional[str] = None
    task_type: Optional[str] = None
    publisher: Optional[str] = None
    manager: Optional[str] = None
    annotators: Optional[str] = None
    instructions: Optional[str] = None


class SplitVideoReq(BaseModel):
    actor: str = ""
    video_id: Optional[str] = None
    segment_length: int
    label_source: str = "input"  # input | locateanything | none


class CreatePartsReq(BaseModel):
    actor: str
    count: int
    name_prefix: str = "Part"
    instructions: str = ""


class ActorReq(BaseModel):
    actor: str


class SubmitPartReq(BaseModel):
    actor: str
    note: str = ""


class ReviewPartReq(BaseModel):
    actor: str
    action: str  # approve | rework
    note: str = ""


class CreateIssueReq(BaseModel):
    actor: str
    part_id: Optional[int] = None
    title: str
    description: str = ""
    severity: str = "normal"


class ResolveIssueReq(BaseModel):
    actor: str
    resolution: str


class BootstrapAdminReq(BaseModel):
    username: str
    password: str
    display_name: str = ""


class LoginReq(BaseModel):
    username: str
    password: str


class CreateUserReq(BaseModel):
    username: str
    password: str
    role: str = "user"
    display_name: str = ""
    is_active: bool = True


class UpdateUserReq(BaseModel):
    role: Optional[str] = None
    display_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class ChangePasswordReq(BaseModel):
    old_password: str
    new_password: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def session_expiry_iso(days: int = SESSION_TTL_DAYS) -> str:
    return (datetime.now(timezone.utc).astimezone() + timedelta(days=max(1, days))).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    safe = []
    for ch in value.strip():
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        elif ch in " .":
            safe.append("_")
    result = "".join(safe).strip("_")
    return result or "task"


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def parse_people(raw: str) -> list[str]:
    people: list[str] = []
    seen: set[str] = set()
    for item in raw.replace(";", "\n").replace(",", "\n").splitlines():
        username = item.strip()
        if username and username not in seen:
            seen.add(username)
            people.append(username)
    return people


def normalize_username(username: str) -> str:
    value = username.strip()
    if not value:
        raise HTTPException(400, "username is required")
    if any(ch.isspace() for ch in value):
        raise HTTPException(400, "username must not contain whitespace")
    return value


def validate_user_role(role: str) -> str:
    value = role.strip().lower()
    if value not in {"admin", "user"}:
        raise HTTPException(400, "role must be admin or user")
    return value


def validate_password(password: str) -> str:
    if len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    return password


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations_raw, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), iterations).hex()
    return hmac.compare_digest(actual, expected)


def session_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def auth_public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "username": str(user["username"]),
        "role": str(user.get("role", "user")),
        "display_name": str(user.get("display_name", "")),
        "is_active": bool(user.get("is_active", True)),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
        "last_login_at": user.get("last_login_at"),
    }


def current_user_optional() -> Optional[dict[str, Any]]:
    return CURRENT_USER.get()


def is_request_active() -> bool:
    return REQUEST_ACTIVE.get()


def require_authenticated_user() -> Optional[dict[str, Any]]:
    user = current_user_optional()
    if user is not None:
        return user
    if is_request_active():
        raise HTTPException(401, "Authentication required")
    return None


def require_admin_user() -> Optional[dict[str, Any]]:
    user = require_authenticated_user()
    if user is not None and str(user.get("role", "user")) != "admin":
        raise HTTPException(403, "Admin privileges are required")
    return user


def resolve_actor(actor: str = "") -> str:
    user = current_user_optional()
    if user is not None:
        return str(user["username"])
    return actor.strip()


def current_user_is_admin() -> bool:
    user = current_user_optional()
    return bool(user and str(user.get("role", "user")) == "admin")


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=max(1, SESSION_TTL_DAYS) * 24 * 60 * 60,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


def require_task_participant(task_id: str, actor: str) -> None:
    username = resolve_actor(actor)
    if current_user_is_admin():
        return
    if not database().is_participant(task_id, username):
        raise HTTPException(403, "当前操作人不是该任务的参与者")


def require_task_manager(task_id: str, actor: str) -> None:
    username = resolve_actor(actor)
    if current_user_is_admin():
        return
    people = database().task_people(task_id)
    orphaned = not people["publisher"] and not people["manager"]
    if not username or (not orphaned and not database().is_manager(task_id, username)):
        raise HTTPException(403, "只有发布人或负责人可以执行该操作")


def require_video_manager(task_id: str, actor: str) -> None:
    task = database().load_task(task_id)
    if task.get("task_type") != "video_detection":
        raise HTTPException(400, "该任务不是视频目标检测标注任务")
    require_task_manager(task_id, actor)


def parse_classes_text(raw: str, default_prompt: str = "person") -> list[dict[str, Any]]:
    classes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_labels: set[str] = set()
    text = raw.strip()
    if not text:
        text = f"0 {default_prompt or 'person'}"
    for line in text.replace(",", "\n").splitlines():
        item = line.strip()
        if not item:
            continue
        class_id: Optional[int] = None
        label = ""
        if "=" in item:
            left, right = item.split("=", 1)
            if right.strip().lstrip("-").isdigit():
                label, class_id = left.strip(), int(right.strip())
            elif left.strip().lstrip("-").isdigit():
                class_id, label = int(left.strip()), right.strip()
        elif ":" in item:
            left, right = item.split(":", 1)
            if left.strip().lstrip("-").isdigit():
                class_id, label = int(left.strip()), right.strip()
            elif right.strip().lstrip("-").isdigit():
                label, class_id = left.strip(), int(right.strip())
        else:
            parts = item.split(maxsplit=1)
            if len(parts) == 2 and parts[0].lstrip("-").isdigit():
                class_id, label = int(parts[0]), parts[1].strip()
            elif len(parts) == 2 and parts[1].lstrip("-").isdigit():
                label, class_id = parts[0].strip(), int(parts[1])
        normalized = normalize_label(label)
        if class_id is None or class_id < 0 or not normalized:
            raise HTTPException(400, f"Invalid class mapping: {item}")
        if class_id in seen_ids:
            raise HTTPException(400, f"Duplicate class id: {class_id}")
        if normalized in seen_labels:
            raise HTTPException(400, f"Duplicate class label: {label}")
        seen_ids.add(class_id)
        seen_labels.add(normalized)
        classes.append({"id": class_id, "name": label.strip()})
    if not classes:
        classes.append({"id": 0, "name": default_prompt or "person"})
    classes.sort(key=lambda item: int(item["id"]))
    return classes


def classes_to_text(classes: list[dict[str, Any]]) -> str:
    return "\n".join(f"{int(item['id'])} {item['name']}" for item in classes)


def task_class_payload(task: dict[str, Any], cfg: dict[str, Any], prompt: str) -> tuple[list[str], dict[str, int], int]:
    classes = task.get("classes") or []
    categories: list[str] = []
    class_map: dict[str, int] = {}
    for item in classes:
        label = str(item.get("name", "")).strip()
        if not label:
            continue
        class_id = int(item.get("id", 0))
        categories.append(label)
        class_map[label] = class_id
    fallback_class_id = int(cfg.get("class_id", 0))
    if not class_map:
        label = prompt or str(cfg.get("prompt", "person"))
        categories = [label]
        class_map = {label: fallback_class_id}
    return categories, class_map, fallback_class_id


def tasks_dir() -> Path:
    path = Path(SETTINGS["tasks_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    configured = str(SETTINGS.get("database_path", "")).strip()
    return Path(configured).expanduser().resolve() if configured else tasks_dir() / "platform.sqlite3"


def database() -> PlatformDatabase:
    global _DATABASE, _DATABASE_KEY
    path = database_path()
    key = str(path)
    if _DATABASE is not None and _DATABASE_KEY == key:
        return _DATABASE
    with _DATABASE_LOCK:
        if _DATABASE is None or _DATABASE_KEY != key:
            store = PlatformDatabase(path)
            store.initialize()
            SETTINGS["legacy_migration"] = store.migrate_legacy_directory(tasks_dir())
            _DATABASE = store
            _DATABASE_KEY = key
    return _DATABASE


def task_dir(task_id: str) -> Path:
    path = tasks_dir() / task_id
    if not path.is_dir() or not database().task_exists(task_id):
        raise HTTPException(404, f"Task not found: {task_id}")
    return path


def load_task(path: Path) -> dict[str, Any]:
    try:
        return database().load_task(path.name)
    except KeyError as exc:
        raise HTTPException(404, f"Task not found: {path.name}") from exc


def save_task(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    database().save_task(payload)


def append_event(path: Path, message: str, level: str = "info") -> None:
    database().add_event(path.name, now_iso(), level, message)


def set_stage(path: Path, stage: str, status: str, message: str = "") -> None:
    with TASK_LOCK:
        task = load_task(path)
        task.setdefault("stages", {}).setdefault(stage, {})
        task["stages"][stage].update({
            "status": status,
            "message": message,
            "updated_at": now_iso(),
        })
        if stage in {"locateanything", "tracking", "package", "export"}:
            task["status"] = f"{stage}_{status}"
        save_task(path, task)
    append_event(path, f"{stage}: {status}{' - ' + message if message else ''}")


def ensure_task_dirs(path: Path) -> None:
    for name in (
        "raw",
        "videos",
        "input_labels",
        "locany_labels",
        "tracking",
        "package",
        "reviewed",
        "exports",
        "attachments",
        "logs",
    ):
        (path / name).mkdir(parents=True, exist_ok=True)


def ensure_task_shape(task: dict[str, Any]) -> dict[str, Any]:
    task.setdefault("deleted", False)
    task.setdefault("videos", [])
    task.setdefault("notes", "")
    task.setdefault("task_type", "video_detection" if task.get("videos") else "general")
    task.setdefault("publisher", task.get("assignee", ""))
    task.setdefault("manager", task.get("assignee", ""))
    task.setdefault("annotators", [])
    task.setdefault("instructions", "")
    task["assignee"] = task.get("manager", "")
    if not task.get("classes"):
        task["classes"] = [{"id": 0, "name": task.get("prompt") or "person"}]
    stages = task.setdefault("stages", {})
    for stage in DEFAULT_STAGE_NAMES:
        stages.setdefault(stage, {"status": "pending", "message": ""})
    task["classes_text"] = classes_to_text(task.get("classes", []))
    return task


def task_with_workflow(task: dict[str, Any], include_detail: bool = False) -> dict[str, Any]:
    task = ensure_task_shape(task)
    task_id = str(task["task_id"])
    task["part_summary"] = database().part_summary(task_id)
    if include_detail:
        task["parts"] = database().list_parts(task_id, now=now_iso())
        attachments = database().list_attachments(task_id)
        for attachment in attachments:
            attachment.pop("stored_path", None)
        task["attachments"] = attachments
        task["issues"] = database().list_issues(task_id)
        task["events"] = database().list_events(task_id, limit=100)
    return task


def locany_settings_payload(path: Path, task: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    task = ensure_task_shape(task or load_task(path))
    effective = task_effective_config(path).get("locateanything", {})
    raw = read_task_local_config(path).get("locateanything", {})
    return {
        "task_id": task["task_id"],
        "prompt": task.get("prompt") or "person",
        "classes": task.get("classes_text") or classes_to_text(task.get("classes", [])),
        "settings": public_locany_settings(effective),
        "overrides": public_locany_settings(raw),
    }


def update_task_locany_overrides(path: Path, req: LocateAnythingSettingsReq) -> None:
    raw = read_task_local_config(path)
    section = dict(raw.get("locateanything", {}))
    for key in (
        "server_url",
        "video_transfer",
        "local_path_prefix",
        "remote_path_prefix",
        "sftp_host",
        "sftp_username",
        "sftp_password_env",
        "sftp_key_path",
        "sftp_remote_dir",
        "device",
        "dtype",
    ):
        value = getattr(req, key)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            section[key] = stripped
        else:
            section.pop(key, None)
    for key in ("sftp_port", "request_timeout", "download_timeout", "poll_interval"):
        value = getattr(req, key)
        if value is not None:
            section[key] = value
    for key in ("sftp_reuse_existing",):
        value = getattr(req, key)
        if value is not None:
            section[key] = bool(value)
    if section:
        raw["locateanything"] = section
    else:
        raw.pop("locateanything", None)
    write_task_local_config(path, raw)


def video_record(task: dict[str, Any], video_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    videos = task.get("videos", [])
    if not videos:
        return None
    target_id = video_id or task.get("current_video_id")
    if target_id:
        for video in videos:
            if video.get("video_id") == target_id:
                return video
    return videos[-1]


def video_root(path: Path, video_id: str) -> Path:
    return path / "videos" / video_id


def current_video_root(path: Path, task: Optional[dict[str, Any]] = None) -> Optional[Path]:
    task = task or load_task(path)
    video = video_record(task)
    if not video:
        return None
    return video_root(path, str(video["video_id"]))


def find_video(path: Path, video_id: Optional[str] = None) -> Optional[Path]:
    if database().task_exists(path.name):
        task = load_task(path)
        video = video_record(task, video_id)
        if video:
            candidate = Path(video.get("path", ""))
            if candidate.is_file():
                return candidate
    raw = path / "raw"
    for item in sorted(raw.iterdir()) if raw.is_dir() else []:
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
            return item
    return None


def video_metadata(video: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"width": width, "height": height, "frame_count": frame_count, "fps": fps}


def label_index(path: Path) -> Optional[int]:
    import re

    match = re.search(r"_([0-9]+)\.txt$", path.name)
    return int(match.group(1)) if match else None


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (output_dir / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}") from exc
        zf.extractall(output_dir)


def write_zip(source_paths: list[tuple[Path, Path]], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in source_paths:
            if src.is_file():
                zf.write(src, arc)
            elif src.is_dir():
                for item in sorted(src.rglob("*")):
                    if item.is_file():
                        zf.write(item, arc / item.relative_to(src))


def project_config() -> dict[str, Any]:
    config_path = Path(SETTINGS["project_config"])
    return load_config(str(config_path)) if config_path.exists() else load_config(None)


def task_effective_config(path: Path) -> dict[str, Any]:
    cfg = project_config()
    local_config = path / "config.json"
    if local_config.exists():
        with open(local_config, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, json.load(f))
    return cfg


def task_config_path(path: Path) -> Path:
    return path / "config.json"


def read_task_local_config(path: Path) -> dict[str, Any]:
    config_path = task_config_path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def write_task_local_config(path: Path, payload: dict[str, Any]) -> None:
    config_path = task_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def locany_request_config(cfg: dict[str, Any], req: Any) -> dict[str, Any]:
    merged = dict(cfg or {})
    for key in (
        "server_url",
        "video_transfer",
        "local_path_prefix",
        "remote_path_prefix",
        "sftp_host",
        "sftp_port",
        "sftp_username",
        "sftp_password",
        "sftp_password_env",
        "sftp_key_path",
        "sftp_remote_dir",
        "sftp_reuse_existing",
        "device",
        "dtype",
        "request_timeout",
        "download_timeout",
        "poll_interval",
    ):
        value = getattr(req, key, None)
        if value is not None:
            merged[key] = value.strip() if isinstance(value, str) else value
    return merged


def public_locany_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    password_env = str(cfg.get("sftp_password_env", "")).strip()
    return {
        "server_url": str(cfg.get("server_url", "")).strip(),
        "video_transfer": str(cfg.get("video_transfer", "sftp")).strip() or "sftp",
        "local_path_prefix": str(cfg.get("local_path_prefix", "")).strip(),
        "remote_path_prefix": str(cfg.get("remote_path_prefix", "")).strip(),
        "sftp_host": str(cfg.get("sftp_host", "")).strip(),
        "sftp_port": int(cfg.get("sftp_port", 22)),
        "sftp_username": str(cfg.get("sftp_username", "")).strip(),
        "sftp_password_env": password_env,
        "sftp_password_configured": bool(password_env and os.environ.get(password_env)),
        "sftp_key_path": str(cfg.get("sftp_key_path", "")).strip(),
        "sftp_remote_dir": str(cfg.get("sftp_remote_dir", "")).strip(),
        "sftp_reuse_existing": bool(cfg.get("sftp_reuse_existing", True)),
        "device": str(cfg.get("device", "cuda")).strip() or "cuda",
        "dtype": str(cfg.get("dtype", "bf16")).strip() or "bf16",
        "request_timeout": float(cfg.get("request_timeout", 30)),
        "download_timeout": float(cfg.get("download_timeout", 600)),
        "poll_interval": float(cfg.get("poll_interval", 5)),
    }


def validate_locany_connection_config(cfg: dict[str, Any]) -> None:
    from urllib.parse import urlparse

    server_url = str(cfg.get("server_url", "")).strip()
    parsed = urlparse(server_url)
    if not server_url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("locateanything.server_url must be a valid HTTP or HTTPS URL")
    transfer = str(cfg.get("video_transfer", "sftp")).strip().lower()
    if transfer not in {"sftp", "path"}:
        raise ValueError("locateanything.video_transfer must be 'sftp' or 'path'")
    if transfer == "sftp":
        missing = [
            key
            for key in ("sftp_host", "sftp_username", "sftp_remote_dir")
            if not str(cfg.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(f"LocateAnything SFTP settings are missing: {', '.join(missing)}")
    if transfer == "path":
        if not str(cfg.get("local_path_prefix", "")).strip() or not str(cfg.get("remote_path_prefix", "")).strip():
            raise ValueError("LocateAnything path transfer requires local_path_prefix and remote_path_prefix")
    port = int(cfg.get("sftp_port", 22))
    if port < 1 or port > 65535:
        raise ValueError("LocateAnything sftp_port must be between 1 and 65535")


def test_locany_sftp(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("SFTP transfer requires paramiko. Install it with: pip install paramiko") from exc

    host = str(cfg.get("sftp_host", "")).strip()
    username = str(cfg.get("sftp_username", "")).strip()
    remote_dir = str(cfg.get("sftp_remote_dir", "")).strip()
    port = int(cfg.get("sftp_port", 22))
    password = str(cfg.get("sftp_password", "")).strip() or None
    password_env = str(cfg.get("sftp_password_env", "LOCANY_SFTP_PASSWORD")).strip()
    if password is None and password_env:
        password = os.environ.get(password_env)
    key_path = str(cfg.get("sftp_key_path", "")).strip()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": 30,
    }
    if key_path:
        connect_kwargs["key_filename"] = key_path
    if password:
        connect_kwargs["password"] = password
    ssh.connect(**connect_kwargs)
    try:
        sftp = ssh.open_sftp()
        try:
            remote_dir_exists = False
            remote_dir_stat_error = ""
            try:
                sftp.stat(remote_dir)
                remote_dir_exists = True
            except OSError as exc:
                remote_dir_stat_error = str(exc)
            return {
                "ok": True,
                "host": host,
                "username": username,
                "remote_dir": remote_dir,
                "remote_dir_exists": remote_dir_exists,
                "remote_dir_stat_error": remote_dir_stat_error,
            }
        finally:
            sftp.close()
    finally:
        ssh.close()


def json_http_request(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
    service_name: str = "remote service",
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_name} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{service_name} request failed: {exc}") from exc


def download_binary(url: str, output_path: Path, timeout: float, service_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(output_path, "wb") as f:
                shutil.copyfileobj(response, f)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_name} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{service_name} request failed: {exc}") from exc


def remote_video_name(video: Path) -> str:
    stat = video.stat()
    key = f"{video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    digest = hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{video.stem}_{digest}{video.suffix}"


def sftp_mkdir_p(sftp: Any, remote_dir: str) -> None:
    parts = [part for part in remote_dir.replace("\\", "/").split("/") if part]
    current = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def upload_video_via_sftp(video: Path, cfg: dict[str, Any], task_path: Path) -> str:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("SFTP transfer requires paramiko. Install it with: pip install paramiko") from exc

    host = str(cfg.get("sftp_host", "")).strip()
    username = str(cfg.get("sftp_username", "")).strip()
    remote_dir = str(cfg.get("sftp_remote_dir", "")).strip()
    if not host or not username or not remote_dir:
        raise RuntimeError("locateanything.sftp_host, sftp_username, and sftp_remote_dir are required")

    port = int(cfg.get("sftp_port", 22))
    password_env = str(cfg.get("sftp_password_env", "LOCANY_SFTP_PASSWORD")).strip()
    password = str(cfg.get("sftp_password", "")).strip() or None
    if password is None and password_env:
        password = os.environ.get(password_env)
    key_path = str(cfg.get("sftp_key_path", "")).strip()
    reuse_existing = bool(cfg.get("sftp_reuse_existing", True))
    remote_path = posixpath.join(remote_dir.rstrip("/"), remote_video_name(video))

    append_event(task_path, f"Uploading video to {host}:{remote_path}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": 30,
    }
    if key_path:
        connect_kwargs["key_filename"] = key_path
    if password:
        connect_kwargs["password"] = password
    ssh.connect(**connect_kwargs)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp_mkdir_p(sftp, remote_dir)
            if reuse_existing:
                try:
                    if sftp.stat(remote_path).st_size == video.stat().st_size:
                        append_event(task_path, f"Reusing uploaded video {remote_path}")
                        return remote_path
                except OSError:
                    pass
            sftp.put(str(video), remote_path)
            return remote_path
        finally:
            sftp.close()
    finally:
        ssh.close()


def locateanything_remote_video_path(video: Path, cfg: dict[str, Any], task_path: Path) -> str:
    transfer = str(cfg.get("video_transfer", "path")).lower()
    if transfer == "sftp":
        return upload_video_via_sftp(video, cfg, task_path)
    if transfer == "path":
        local_prefix = str(cfg.get("local_path_prefix", "")).strip()
        remote_prefix = str(cfg.get("remote_path_prefix", "")).strip().replace("\\", "/")
        if not local_prefix or not remote_prefix:
            raise RuntimeError(
                "LocateAnything path transfer requires local_path_prefix and remote_path_prefix"
            )
        local_root = Path(local_prefix).expanduser().resolve()
        try:
            rel_path = video.resolve().relative_to(local_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Video path {video.resolve()} is outside locateanything.local_path_prefix {local_root}"
            ) from exc
        return posixpath.join(remote_prefix.rstrip("/"), rel_path.as_posix())
    raise RuntimeError(f"Unsupported locateanything.video_transfer: {transfer}")


def build_locany_payload(
    task: dict[str, Any],
    cfg: dict[str, Any],
    req: RunLocateAnythingReq,
    video: Path,
    remote_video_path: str,
    file_prefix: str,
) -> tuple[dict[str, Any], str]:
    prompt = req.prompt or task.get("prompt") or str(cfg.get("prompt", "person"))
    categories, class_map, fallback_class_id = task_class_payload(task, cfg, prompt)
    locany_task = str(cfg.get("task", "ground_multi"))
    if len(categories) > 1 and not str(cfg.get("question", "")).strip():
        locany_task = "detect"
    device = str(cfg.get("device", "cuda"))
    if req.device is not None and str(req.device).strip():
        device = str(req.device).strip()
    if req.cuda_device is not None:
        device = f"cuda:{req.cuda_device}"
    payload = {
        "video_path": remote_video_path,
        "prompt": prompt,
        "categories": categories,
        "class_map": class_map,
        "task": locany_task,
        "question": str(cfg.get("question", "")),
        "class_id": fallback_class_id,
        "score": float(cfg.get("score", 1.0)),
        "start_frame": max(0, req.start_frame),
        "max_frames": max(0, req.max_frames),
        "frame_step": max(1, req.frame_step),
        "frame_offset": int(cfg.get("frame_offset", 1)),
        "file_prefix": file_prefix or video.stem,
        "resize_long_edge": int(cfg.get("resize_long_edge", 1024)),
        "generation_mode": str(cfg.get("generation_mode", "slow")),
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "use_cache": bool(cfg.get("use_cache", True)),
        "device": device,
        "dtype": str(req.dtype).strip() if req.dtype is not None and str(req.dtype).strip() else str(cfg.get("dtype", "bf16")),
    }
    return payload, prompt


def run_remote_locany_video(
    task_path: Path,
    task: dict[str, Any],
    cfg: dict[str, Any],
    req: RunLocateAnythingReq,
    video: Path,
    final_dir: Path,
    status_prefix: str,
) -> tuple[Path, str]:
    server_url = str(cfg.get("server_url", "")).rstrip("/")
    if not server_url:
        raise RuntimeError("locateanything.server_url is not configured")
    remote_video_path = locateanything_remote_video_path(video, cfg, task_path)
    payload, prompt = build_locany_payload(task, cfg, req, video, remote_video_path, video.stem)
    timeout = float(cfg.get("request_timeout", 30))
    poll_interval = float(cfg.get("poll_interval", 5))
    set_stage(task_path, "locateanything", "running", f"{status_prefix}: submitting remote job")
    remote = json_http_request(
        "POST",
        f"{server_url}/api/locateanything/jobs",
        payload,
        timeout,
        "LocateAnything",
    )
    remote_job_id = remote["job_id"]

    while True:
        status = json_http_request(
            "GET",
            f"{server_url}/api/locateanything/jobs/{remote_job_id}",
            None,
            timeout,
            "LocateAnything",
        )
        message = status.get("message", status.get("status", ""))
        set_stage(task_path, "locateanything", "running", f"{status_prefix}: {message}")
        if status.get("status") == "done":
            break
        if status.get("status") == "failed":
            raise RuntimeError(message or "Remote LocateAnything failed")
        time.sleep(poll_interval)

    zip_path = final_dir.parent / "locateanything_yolo.zip"
    extract_dir = final_dir.parent / "_extract"
    download_binary(
        f"{server_url}/api/locateanything/jobs/{remote_job_id}/yolo-zip",
        zip_path,
        float(cfg.get("download_timeout", 600)),
        "LocateAnything",
    )
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    safe_extract_zip(zip_path, extract_dir)
    labels_src = extract_dir / "labels"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    if labels_src.is_dir():
        shutil.move(str(labels_src), str(final_dir))
    else:
        final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir, prompt


def run_locany_job(task_id: str, req: RunLocateAnythingReq) -> None:
    path = tasks_dir() / task_id
    with JOB_LOCK:
        try:
            set_stage(path, "locateanything", "running", "Preparing video")
            video = find_video(path, req.video_id)
            if video is None:
                raise RuntimeError("No video uploaded")
            task = ensure_task_shape(load_task(path))
            cfg = locany_request_config(task_effective_config(path).get("locateanything", {}), req)
            validate_locany_connection_config(cfg)
            record = video_record(task, req.video_id)
            current_root = video_root(path, str(record["video_id"])) if record else current_video_root(path, task)
            if current_root is None:
                raise RuntimeError("No current video")
            final_dir = current_root / "locany_labels" / video.stem
            final_dir, prompt = run_remote_locany_video(path, task, cfg, req, video, final_dir, "full video")
            set_stage(path, "locateanything", "done", f"Labels saved to {final_dir}")
            with TASK_LOCK:
                task = ensure_task_shape(load_task(path))
                task["prelabel_source"] = "locateanything"
                task["prelabel_dir"] = str(final_dir)
                task["prompt"] = prompt
                record = video_record(task, req.video_id)
                if record is not None:
                    record["locany_label_dir"] = str(final_dir)
                    record["updated_at"] = now_iso()
                save_task(path, task)
        except Exception as exc:
            set_stage(path, "locateanything", "failed", str(exc))


def run_segment_locany_job(task_id: str, req: RunLocateAnythingReq) -> None:
    path = tasks_dir() / task_id
    with JOB_LOCK:
        try:
            set_stage(path, "locateanything", "running", "Starting segment LocateAnything")
            task = ensure_task_shape(load_task(path))
            record = video_record(task, req.video_id)
            if record is None:
                raise RuntimeError("No current video")
            segments = record.get("segments", [])
            if not segments:
                raise RuntimeError("Current video has no segments")
            cfg = locany_request_config(task_effective_config(path).get("locateanything", {}), req)
            validate_locany_connection_config(cfg)
            completed = 0
            failed = 0
            for segment in segments:
                seg_video = Path(segment["video_path"])
                seg_dir = seg_video.parents[1]
                final_dir = seg_dir / "locany_labels" / seg_video.stem
                try:
                    final_dir, prompt = run_remote_locany_video(
                        path,
                        task,
                        cfg,
                        req,
                        seg_video,
                        final_dir,
                        segment["segment_id"],
                    )
                    segment["locateanything"] = {
                        "status": "done",
                        "label_dir": str(final_dir),
                        "prompt": prompt,
                        "updated_at": now_iso(),
                    }
                    segment["locany_label_dir"] = str(final_dir)
                    completed += 1
                except Exception as exc:
                    segment["locateanything"] = {
                        "status": "failed",
                        "message": str(exc),
                        "updated_at": now_iso(),
                    }
                    failed += 1
                with TASK_LOCK:
                    latest = ensure_task_shape(load_task(path))
                    latest_record = video_record(latest, str(record["video_id"]))
                    if latest_record is not None:
                        latest_record["segments"] = segments
                    latest["prelabel_source"] = "locateanything"
                    save_task(path, latest)
                set_stage(path, "locateanything", "running", f"Segments done={completed}, failed={failed}")
            set_stage(path, "locateanything", "done", f"Segment LocateAnything done={completed}, failed={failed}")
        except Exception as exc:
            set_stage(path, "locateanything", "failed", str(exc))


def select_label_dir(path: Path, source: str) -> Path:
    video = find_video(path)
    if video is None:
        raise RuntimeError("No video uploaded")
    task = ensure_task_shape(load_task(path))
    current_root = current_video_root(path, task)
    candidates = []
    if source in {"auto", "locateanything"}:
        if current_root is not None:
            candidates.append(current_root / "locany_labels" / video.stem)
        candidates.append(path / "locany_labels" / video.stem)
    if source in {"auto", "input"}:
        if current_root is not None:
            candidates.append(current_root / "input_labels" / video.stem)
        candidates.append(path / "input_labels" / video.stem)
        candidates.append(path / "input_labels")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    raise RuntimeError(f"No usable label directory for source={source}")


def build_label_map(label_dir: Optional[Path]) -> tuple[dict[int, Path], int]:
    if label_dir is None or not label_dir.is_dir():
        return {}, 1
    indexed: list[tuple[int, Path]] = []
    for txt in label_dir.glob("*.txt"):
        idx = label_index(txt)
        if idx is not None:
            indexed.append((idx, txt))
    if not indexed:
        return {}, 1
    indexed.sort(key=lambda item: item[0])
    frame_offset = 1 if indexed[0][0] == 1 else 0
    return {idx - frame_offset: path for idx, path in indexed}, frame_offset


def split_video_job(task_id: str, req: SplitVideoReq) -> None:
    path = tasks_dir() / task_id
    try:
        task = ensure_task_shape(load_task(path))
        record = video_record(task, req.video_id)
        if record is None:
            raise RuntimeError("No video available to split")
        video_id = str(record["video_id"])
        video_path = Path(record["path"])
        if not video_path.is_file():
            raise RuntimeError(f"Video file missing: {video_path}")
        segment_length = max(1, int(req.segment_length))
        vroot = video_root(path, video_id)
        segments_root = vroot / "segments"
        if segments_root.exists():
            shutil.rmtree(segments_root)
        segments_root.mkdir(parents=True, exist_ok=True)

        label_dir: Optional[Path] = None
        if req.label_source == "input":
            raw_label_dir = record.get("input_label_dir")
            label_dir = Path(raw_label_dir) if raw_label_dir else None
        elif req.label_source == "locateanything":
            raw_label_dir = record.get("locany_label_dir")
            label_dir = Path(raw_label_dir) if raw_label_dir else None
        label_map, label_frame_offset = build_label_map(label_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        segments = []

        for seg_idx, seg_start in enumerate(range(0, frame_count, segment_length)):
            seg_end = min(frame_count, seg_start + segment_length)
            seg_id = f"seg_{seg_idx:04d}"
            seg_dir = segments_root / seg_id
            raw_dir = seg_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            seg_video_path = raw_dir / video_path.name
            writer = cv2.VideoWriter(str(seg_video_path), fourcc, fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            written = 0
            for _ in range(seg_start, seg_end):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
            writer.release()

            seg_label_dir = None
            copied_labels = 0
            if label_map:
                seg_label_dir = seg_dir / "input_labels" / video_path.stem
                seg_label_dir.mkdir(parents=True, exist_ok=True)
                for global_idx in range(seg_start, seg_start + written):
                    src = label_map.get(global_idx)
                    local_frame_id = global_idx - seg_start + label_frame_offset
                    dst = seg_label_dir / f"{video_path.stem}_{local_frame_id}.txt"
                    if src is not None:
                        shutil.copy2(str(src), str(dst))
                        copied_labels += 1
                    else:
                        dst.write_text("", encoding="utf-8")

            segments.append({
                "segment_id": seg_id,
                "start_frame": seg_start,
                "end_frame": seg_start + written - 1,
                "frame_count": written,
                "video_path": str(seg_video_path),
                "input_label_dir": str(seg_label_dir) if seg_label_dir else "",
                "labels_copied": copied_labels,
                "status": "ready",
                "created_at": now_iso(),
            })

        cap.release()
        with TASK_LOCK:
            task = ensure_task_shape(load_task(path))
            record = video_record(task, video_id)
            if record is None:
                raise RuntimeError("Video record disappeared during split")
            record["segments"] = segments
            record["split"] = {
                "status": "done",
                "segment_length": segment_length,
                "label_source": req.label_source,
                "segments": len(segments),
                "updated_at": now_iso(),
            }
            task["status"] = "segmented"
            save_task(path, task)
        append_event(path, f"Video {video_id} split into {len(segments)} segments")
    except Exception as exc:
        with TASK_LOCK:
            task = ensure_task_shape(load_task(path))
            record = video_record(task, req.video_id)
            if record is not None:
                record["split"] = {"status": "failed", "message": str(exc), "updated_at": now_iso()}
            task["status"] = "split_failed"
            save_task(path, task)
        append_event(path, f"Split failed: {exc}", level="error")


def run_tracking_job(task_id: str, req: RunTrackingReq) -> None:
    path = tasks_dir() / task_id
    with JOB_LOCK:
        try:
            set_stage(path, "tracking", "running", "Starting tracking pipeline")
            video = find_video(path)
            if video is None:
                raise RuntimeError("No video uploaded")
            label_dir = select_label_dir(path, req.label_source)
            out_dir = path / "tracking"
            cfg = task_effective_config(path)
            run_pipeline(video_path=video, ann_dir=label_dir, out_dir=out_dir, config=cfg)
            results = out_dir / "tracking_results.json"
            if results.exists() and req.short_gap_max > 0:
                state = AnnotationState()
                state.import_tracking_results(str(results))
                gap_result = state.interpolate_short_gaps(req.short_gap_max)
                with open(results, "w", encoding="utf-8") as f:
                    json.dump(state.export_tracking_results(), f, ensure_ascii=False, indent=2)
                    f.write("\n")
                export_tracking_results_to_yolo(results, out_dir / cfg["exports"]["yolo_dirname"])
                append_event(path, f"Filled short gaps: {gap_result}")
            set_stage(path, "tracking", "done", f"Tracking results saved to {out_dir}")
            with TASK_LOCK:
                task = load_task(path)
                task["tracking_results"] = str(results)
                task["status"] = "needs_review"
                save_task(path, task)
        except Exception as exc:
            set_stage(path, "tracking", "failed", str(exc))


def run_segment_tracking_job(task_id: str, req: RunTrackingReq) -> None:
    path = tasks_dir() / task_id
    with JOB_LOCK:
        try:
            set_stage(path, "tracking", "running", "Starting segment tracking")
            task = ensure_task_shape(load_task(path))
            record = video_record(task)
            if record is None:
                raise RuntimeError("No current video")
            segments = record.get("segments", [])
            if not segments:
                raise RuntimeError("Current video has no segments")
            cfg = task_effective_config(path)
            completed = 0
            failed = 0
            for segment in segments:
                seg_dir = Path(segment["video_path"]).parents[1]
                seg_video = Path(segment["video_path"])
                label_dir = Path(segment.get("input_label_dir", ""))
                if not label_dir.is_dir() and req.label_source in {"auto", "locateanything"}:
                    label_dir = Path(segment.get("locany_label_dir", "")) if segment.get("locany_label_dir") else seg_dir / "locany_labels" / seg_video.stem
                if not label_dir.is_dir() or not any(label_dir.glob("*.txt")):
                    segment["tracking"] = {
                        "status": "skipped",
                        "message": "No segment labels",
                        "updated_at": now_iso(),
                    }
                    continue
                out_dir = seg_dir / "tracking"
                try:
                    run_pipeline(video_path=seg_video, ann_dir=label_dir, out_dir=out_dir, config=cfg)
                    results = out_dir / "tracking_results.json"
                    if results.exists() and req.short_gap_max > 0:
                        state = AnnotationState()
                        state.import_tracking_results(str(results))
                        gap_result = state.interpolate_short_gaps(req.short_gap_max)
                        with open(results, "w", encoding="utf-8") as f:
                            json.dump(state.export_tracking_results(), f, ensure_ascii=False, indent=2)
                            f.write("\n")
                        export_tracking_results_to_yolo(results, out_dir / cfg["exports"]["yolo_dirname"])
                        append_event(path, f"{segment['segment_id']} filled short gaps: {gap_result}")
                    segment["tracking"] = {
                        "status": "done",
                        "results": str(results),
                        "updated_at": now_iso(),
                    }
                    completed += 1
                except Exception as exc:
                    segment["tracking"] = {
                        "status": "failed",
                        "message": str(exc),
                        "updated_at": now_iso(),
                    }
                    failed += 1
                with TASK_LOCK:
                    latest = ensure_task_shape(load_task(path))
                    latest_record = video_record(latest, str(record["video_id"]))
                    if latest_record is not None:
                        latest_record["segments"] = segments
                    save_task(path, latest)
                set_stage(path, "tracking", "running", f"Segments done={completed}, failed={failed}")
            set_stage(path, "tracking", "done", f"Segment tracking done={completed}, failed={failed}")
        except Exception as exc:
            set_stage(path, "tracking", "failed", str(exc))


def build_package(task_id: str, include_video: bool) -> Path:
    path = task_dir(task_id)
    video = find_video(path)
    tracking_json = path / "tracking" / "tracking_results.json"
    if not tracking_json.exists():
        raise HTTPException(400, "tracking_results.json does not exist")
    package_path = path / "package" / f"{task_id}_annotation_package.zip"
    task_snapshot = path / "package" / "task.json"
    task_snapshot.write_text(
        json.dumps(ensure_task_shape(load_task(path)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    items: list[tuple[Path, Path]] = [
        (tracking_json, Path("tracking_results.json")),
        (task_snapshot, Path("task.json")),
    ]
    overview = path / "tracking" / "tracking_overview.mp4"
    if overview.exists():
        items.append((overview, Path("tracking_overview.mp4")))
    if include_video and video is not None:
        items.append((video, Path("raw") / video.name))
    write_zip(items, package_path)
    set_stage(path, "package", "done", f"Package saved to {package_path}")
    return package_path


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    db_health = database().health()
    auth_user = current_user_optional()
    return {
        "ok": True,
        "service": "annotation-platform",
        "api_schema_version": API_SCHEMA_VERSION,
        "tasks_dir": str(SETTINGS["tasks_dir"]),
        "config": str(SETTINGS["project_config"]),
        "database": db_health,
        "auth": {
            "enabled": True,
            "bootstrap_required": database().user_count() == 0,
            "authenticated": auth_user is not None,
            "user": auth_public_user(auth_user) if auth_user is not None else None,
        },
        "legacy_migration": SETTINGS.get("legacy_migration", {"tasks": 0, "events": 0, "failed": 0}),
    }


@app.get("/api/auth/me")
async def auth_me():
    user = current_user_optional()
    if user is None:
        return {"authenticated": False, "bootstrap_required": database().user_count() == 0}
    return {
        "authenticated": True,
        "bootstrap_required": False,
        "user": auth_public_user(user),
    }


@app.post("/api/auth/bootstrap-admin")
async def bootstrap_admin(req: BootstrapAdminReq, response: Response):
    if database().user_count() > 0:
        raise HTTPException(409, "Bootstrap is already complete")
    username = normalize_username(req.username)
    password = validate_password(req.password)
    now = now_iso()
    user = database().create_user(
        username,
        hash_password(password),
        "admin",
        req.display_name.strip(),
        now,
        True,
    )
    raw_token = secrets.token_urlsafe(32)
    database().create_session(str(uuid.uuid4()), username, session_token_hash(raw_token), session_expiry_iso(), now)
    database().touch_user_login(username, now)
    set_session_cookie(response, raw_token)
    return {"ok": True, "user": auth_public_user(user)}


@app.post("/api/auth/login")
async def login(req: LoginReq, response: Response):
    username = normalize_username(req.username)
    password = req.password
    try:
        user = database().get_user_with_password(username)
    except KeyError as exc:
        raise HTTPException(401, "Invalid username or password") from exc
    if not bool(user.get("is_active", True)):
        raise HTTPException(403, "This account is disabled")
    if not verify_password(password, str(user.get("password_hash", ""))):
        raise HTTPException(401, "Invalid username or password")
    now = now_iso()
    raw_token = secrets.token_urlsafe(32)
    database().create_session(str(uuid.uuid4()), username, session_token_hash(raw_token), session_expiry_iso(), now)
    database().touch_user_login(username, now)
    set_session_cookie(response, raw_token)
    return {"ok": True, "user": auth_public_user(database().get_user(username))}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    raw_token = request.cookies.get(SESSION_COOKIE_NAME, "")
    if raw_token:
        database().delete_session(session_token_hash(raw_token))
    clear_session_cookie(response)
    return {"ok": True}


@app.post("/api/auth/change-password")
async def change_password(req: ChangePasswordReq):
    user = require_authenticated_user()
    if user is None:
        raise HTTPException(401, "Authentication required")
    current = database().get_user_with_password(str(user["username"]))
    if not verify_password(req.old_password, str(current.get("password_hash", ""))):
        raise HTTPException(401, "Current password is incorrect")
    validate_password(req.new_password)
    database().update_user(str(user["username"]), password_hash=hash_password(req.new_password), now=now_iso())
    return {"ok": True}


@app.get("/api/users")
async def list_users():
    require_admin_user()
    return {"users": database().list_users()}


@app.post("/api/users")
async def create_user(req: CreateUserReq):
    require_admin_user()
    username = normalize_username(req.username)
    password = validate_password(req.password)
    role = validate_user_role(req.role)
    try:
        user = database().create_user(
            username,
            hash_password(password),
            role,
            req.display_name.strip(),
            now_iso(),
            req.is_active,
        )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(409, "Username already exists") from exc
        raise
    return {"ok": True, "user": user}


@app.patch("/api/users/{username}")
async def update_user(username: str, req: UpdateUserReq):
    require_admin_user()
    normalized = normalize_username(username)
    try:
        before = database().get_user(normalized)
    except KeyError as exc:
        raise HTTPException(404, "User not found") from exc
    next_role = validate_user_role(req.role) if req.role is not None else str(before.get("role", "user"))
    next_active = bool(req.is_active) if req.is_active is not None else bool(before.get("is_active", True))
    if before.get("role") == "admin" and database().active_admin_count() == 1 and (next_role != "admin" or not next_active):
        raise HTTPException(400, "At least one active admin account must remain")
    password_hash = hash_password(validate_password(req.password)) if req.password is not None else None
    user = database().update_user(
        normalized,
        password_hash=password_hash,
        role=req.role.strip().lower() if req.role is not None else None,
        display_name=req.display_name.strip() if req.display_name is not None else None,
        is_active=req.is_active,
        now=now_iso(),
    )
    if password_hash is not None or req.is_active is False:
        database().delete_sessions_for_user(normalized)
    return {"ok": True, "user": user}


@app.get("/api/tasks")
async def list_tasks(include_deleted: bool = False):
    require_authenticated_user()
    rows = [task_with_workflow(task) for task in database().list_tasks(include_deleted)]
    return {"tasks": rows}


@app.post("/api/tasks")
async def create_task(req: CreateTaskReq):
    if req.task_type not in {"general", "video_detection"}:
        raise HTTPException(400, "task_type must be general or video_detection")
    if req.part_count < 0 or req.part_count > 10000:
        raise HTTPException(400, "part_count must be between 0 and 10000")
    manager = req.manager.strip() or req.assignee.strip() or req.publisher.strip()
    publisher = req.publisher.strip() or manager
    if not publisher or not manager:
        raise HTTPException(400, "发布人和负责人不能为空")
    task_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(req.name)}_{uuid.uuid4().hex[:6]}"
    path = tasks_dir() / task_id
    path.mkdir(parents=True, exist_ok=False)
    ensure_task_dirs(path)
    classes = parse_classes_text(req.classes, req.prompt)
    task = {
        "task_id": task_id,
        "name": req.name,
        "task_type": req.task_type,
        "publisher": publisher,
        "manager": manager,
        "assignee": manager,
        "annotators": parse_people(req.annotators),
        "instructions": req.instructions,
        "notes": req.notes,
        "prelabel_source": req.prelabel_source,
        "prompt": req.prompt,
        "classes": classes,
        "classes_text": classes_to_text(classes),
        "status": "created",
        "deleted": False,
        "videos": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "stages": {
            "video": {"status": "pending", "message": ""},
            "prelabel": {"status": "pending", "message": ""},
            "locateanything": {"status": "pending", "message": ""},
            "tracking": {"status": "pending", "message": ""},
            "package": {"status": "pending", "message": ""},
            "review": {"status": "pending", "message": ""},
            "export": {"status": "pending", "message": ""},
        },
    }
    save_task(path, task)
    append_event(path, "Task created")
    if req.part_count:
        database().create_parts(
            task_id,
            req.part_count,
            req.part_name_prefix,
            req.instructions,
            now_iso(),
        )
        append_event(path, f"Created {req.part_count} parts")
    return {"ok": True, "task": task_with_workflow(load_task(path), include_detail=True)}


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, req: UpdateTaskReq):
    path = task_dir(task_id)
    require_task_manager(task_id, req.actor)
    with TASK_LOCK:
        task = ensure_task_shape(load_task(path))
        for key in ("name", "assignee", "notes", "prompt"):
            value = getattr(req, key)
            if value is not None:
                task[key] = value
        for key in ("task_type", "publisher", "manager", "instructions"):
            value = getattr(req, key)
            if value is not None:
                task[key] = value
        if req.annotators is not None:
            task["annotators"] = parse_people(req.annotators)
        if req.manager is not None:
            task["assignee"] = req.manager
        elif req.assignee is not None:
            task["manager"] = req.assignee
        if req.classes is not None:
            task["classes"] = parse_classes_text(req.classes, task.get("prompt") or "person")
            task["classes_text"] = classes_to_text(task["classes"])
        save_task(path, task)
    append_event(path, "Task metadata updated")
    return {"ok": True, "task": task_with_workflow(task, include_detail=True)}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str, actor: str):
    path = task_dir(task_id)
    require_task_manager(task_id, actor)
    with TASK_LOCK:
        task = ensure_task_shape(load_task(path))
        task["deleted"] = True
        task["deleted_at"] = now_iso()
        task["status"] = "deleted"
        save_task(path, task)
    append_event(path, "Task soft-deleted")
    return {"ok": True}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    path = task_dir(task_id)
    return task_with_workflow(load_task(path), include_detail=True)


@app.get("/api/tasks/{task_id}/locateanything-settings")
async def get_task_locany_settings(task_id: str):
    path = task_dir(task_id)
    task = load_task(path)
    if task.get("task_type") != "video_detection":
        raise HTTPException(400, "Only video detection tasks support LocateAnything settings")
    return locany_settings_payload(path, task)


@app.patch("/api/tasks/{task_id}/locateanything-settings")
async def save_task_locany_settings(task_id: str, req: LocateAnythingSettingsReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    with TASK_LOCK:
        task = ensure_task_shape(load_task(path))
        if req.prompt is not None:
            task["prompt"] = req.prompt.strip() or task.get("prompt") or "person"
        if req.classes is not None:
            task["classes"] = parse_classes_text(req.classes, task.get("prompt") or "person")
            task["classes_text"] = classes_to_text(task["classes"])
        update_task_locany_overrides(path, req)
        save_task(path, task)
    append_event(path, f"{req.actor} updated LocateAnything settings")
    return {"ok": True, **locany_settings_payload(path)}


@app.delete("/api/tasks/{task_id}/locateanything-settings")
async def reset_task_locany_settings(task_id: str, actor: str):
    path = task_dir(task_id)
    require_video_manager(task_id, actor)
    with TASK_LOCK:
        raw = read_task_local_config(path)
        raw.pop("locateanything", None)
        write_task_local_config(path, raw)
    append_event(path, f"{actor} reset LocateAnything overrides")
    return {"ok": True, **locany_settings_payload(path)}


@app.post("/api/tasks/{task_id}/locateanything/test-connection")
async def test_task_locany_connection(task_id: str, req: LocateAnythingSettingsReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    cfg = locany_request_config(task_effective_config(path).get("locateanything", {}), req)
    try:
        validate_locany_connection_config(cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    server_url = str(cfg.get("server_url", "")).rstrip("/")
    try:
        server = await asyncio.to_thread(
            json_http_request,
            "GET",
            f"{server_url}/api/health",
            None,
            float(cfg.get("request_timeout", 30)),
            "LocateAnything",
        )
        payload: dict[str, Any] = {
            "ok": True,
            "server": {
                "service": server.get("service", ""),
                "ok": server.get("ok", False),
                "cache_dir": server.get("cache_dir"),
                "loaded_workers": server.get("loaded_workers", []),
            },
            "video_transfer": str(cfg.get("video_transfer", "sftp")).lower(),
        }
        if payload["video_transfer"] == "sftp":
            payload["sftp"] = await asyncio.to_thread(test_locany_sftp, cfg)
        else:
            payload["path_mapping"] = {
                "local_path_prefix": str(cfg.get("local_path_prefix", "")).strip(),
                "remote_path_prefix": str(cfg.get("remote_path_prefix", "")).strip(),
            }
        return payload
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/tasks/{task_id}/parts")
async def create_parts(task_id: str, req: CreatePartsReq):
    path = task_dir(task_id)
    require_task_manager(task_id, req.actor)
    try:
        parts = database().create_parts(
            task_id,
            req.count,
            req.name_prefix,
            req.instructions,
            now_iso(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    append_event(path, f"{req.actor} created {req.count} parts")
    return {"ok": True, "parts": parts, "summary": database().part_summary(task_id)}


@app.post("/api/tasks/{task_id}/parts/claim-next")
async def claim_next_part(task_id: str, req: ActorReq):
    path = task_dir(task_id)
    try:
        part = database().claim_next_part(task_id, req.actor.strip(), now_iso())
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if part is None:
        raise HTTPException(409, "没有可领取的 Part")
    append_event(path, f"{req.actor} claimed {part['name']}")
    return {"ok": True, "part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/start")
async def start_part(task_id: str, part_id: int, req: ActorReq):
    path = task_dir(task_id)
    try:
        part = database().start_part(task_id, part_id, req.actor.strip(), now_iso())
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    append_event(path, f"{req.actor} started {part['name']}")
    return {"ok": True, "part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/submit")
async def submit_part(task_id: str, part_id: int, req: SubmitPartReq):
    path = task_dir(task_id)
    try:
        part = database().submit_part(task_id, part_id, req.actor.strip(), req.note, now_iso())
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "Part 不存在") from exc
    append_event(path, f"{req.actor} submitted {part['name']}")
    return {"ok": True, "part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/release")
async def release_part(task_id: str, part_id: int, req: ActorReq):
    path = task_dir(task_id)
    try:
        part = database().release_part(task_id, part_id, req.actor.strip(), now_iso())
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "Part 不存在") from exc
    append_event(path, f"{req.actor} released {part['name']}")
    return {"ok": True, "part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/review")
async def review_part(task_id: str, part_id: int, req: ReviewPartReq):
    path = task_dir(task_id)
    try:
        part = database().review_part(
            task_id, part_id, req.actor.strip(), req.action, req.note, now_iso()
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    append_event(path, f"{req.actor} reviewed {part['name']}: {req.action}")
    return {"ok": True, "part": part}


@app.post("/api/tasks/{task_id}/attachments")
async def upload_attachment(
    task_id: str,
    file: UploadFile = File(...),
    actor: str = Form(...),
    part_id: Optional[int] = Form(None),
):
    path = task_dir(task_id)
    require_task_participant(task_id, actor)
    if part_id is not None:
        try:
            database().get_part(task_id, part_id)
        except KeyError as exc:
            raise HTTPException(404, "Part 不存在") from exc
    filename = Path(file.filename or "attachment.bin").name
    attachment_id = uuid.uuid4().hex
    stored = path / "attachments" / f"{attachment_id}_{filename}"
    digest = hashlib.sha256()
    size = 0
    with open(stored, "wb") as output:
        while chunk := file.file.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    record = database().add_attachment({
        "attachment_id": attachment_id,
        "task_id": task_id,
        "part_id": part_id,
        "filename": filename,
        "stored_path": str(stored),
        "media_type": file.content_type or "application/octet-stream",
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "uploaded_by": actor,
        "created_at": now_iso(),
    })
    record.pop("stored_path", None)
    append_event(path, f"{actor} uploaded attachment: {filename}")
    return {"ok": True, "attachment": record}


@app.get("/api/tasks/{task_id}/attachments/{attachment_id}/download")
async def download_attachment(task_id: str, attachment_id: str):
    task_dir(task_id)
    try:
        attachment = database().get_attachment(task_id, attachment_id)
    except KeyError as exc:
        raise HTTPException(404, "附件不存在") from exc
    stored = Path(attachment["stored_path"])
    if not stored.is_file():
        raise HTTPException(404, "附件文件已丢失")
    return FileResponse(
        str(stored),
        filename=attachment["filename"],
        media_type=attachment["media_type"],
    )


@app.post("/api/tasks/{task_id}/issues")
async def create_issue(task_id: str, req: CreateIssueReq):
    path = task_dir(task_id)
    require_task_participant(task_id, req.actor)
    if req.severity not in {"low", "normal", "high", "blocking"}:
        raise HTTPException(400, "无效的问题严重程度")
    if req.part_id is not None:
        try:
            database().get_part(task_id, req.part_id)
        except KeyError as exc:
            raise HTTPException(404, "Part 不存在") from exc
    task = load_task(path)
    issue = database().create_issue({
        "task_id": task_id,
        "part_id": req.part_id,
        "reported_by": req.actor,
        "assigned_to": task.get("manager", ""),
        "severity": req.severity,
        "title": req.title.strip(),
        "description": req.description,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    })
    append_event(path, f"{req.actor} reported issue: {req.title}", level="warning")
    return {"ok": True, "issue": issue}


@app.post("/api/tasks/{task_id}/issues/{issue_id}/resolve")
async def resolve_issue(task_id: str, issue_id: int, req: ResolveIssueReq):
    path = task_dir(task_id)
    try:
        issue = database().resolve_issue(
            task_id, issue_id, req.actor.strip(), req.resolution, now_iso()
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "问题单不存在") from exc
    append_event(path, f"{req.actor} resolved issue #{issue_id}")
    return {"ok": True, "issue": issue}


@app.post("/api/tasks/{task_id}/video")
async def upload_video(task_id: str, file: UploadFile = File(...), actor: str = Form(...)):
    path = task_dir(task_id)
    require_video_manager(task_id, actor)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported video extension: {suffix}")
    original_name = Path(file.filename or f"video{suffix}").name
    video_id = f"{slugify(Path(original_name).stem)}_{uuid.uuid4().hex[:6]}"
    video_dir = video_root(path, video_id)
    raw_dir = video_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / original_name
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    meta = video_metadata(dst)
    with TASK_LOCK:
        task = ensure_task_shape(load_task(path))
        video = {
            "video_id": video_id,
            "name": original_name,
            "path": str(dst),
            "metadata": meta,
            "status": "uploaded",
            "segments": [],
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        task["videos"].append(video)
        task["current_video_id"] = video_id
        task["video"] = str(dst)
        task["video_metadata"] = meta
        task["stages"]["video"] = {"status": "done", "message": dst.name, "updated_at": now_iso()}
        task["status"] = "video_uploaded"
        save_task(path, task)
    append_event(path, f"Video uploaded: {dst.name} ({video_id})")
    return {"ok": True, "video_id": video_id, "video": str(dst), "metadata": meta}


@app.post("/api/tasks/{task_id}/labels-zip")
async def upload_labels_zip(
    task_id: str,
    file: UploadFile = File(...),
    actor: str = Form(...),
):
    path = task_dir(task_id)
    require_video_manager(task_id, actor)
    video = find_video(path)
    if video is None:
        raise HTTPException(400, "Upload video before labels")
    task = ensure_task_shape(load_task(path))
    current_root = current_video_root(path, task)
    if current_root is None:
        raise HTTPException(400, "No current video")
    zip_path = current_root / "input_labels" / "uploaded_labels.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    extract_dir = current_root / "input_labels" / "_extract"
    final_dir = current_root / "input_labels" / video.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    safe_extract_zip(zip_path, extract_dir)
    txt_files = list(extract_dir.rglob("*.txt"))
    if not txt_files:
        raise HTTPException(400, "Zip does not contain txt labels")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for txt in txt_files:
        shutil.copy2(str(txt), str(final_dir / txt.name))
    with TASK_LOCK:
        task = ensure_task_shape(load_task(path))
        task["prelabel_source"] = "yolo"
        task["prelabel_dir"] = str(final_dir)
        record = video_record(task)
        if record is not None:
            record["input_label_dir"] = str(final_dir)
            record["updated_at"] = now_iso()
        task["stages"]["prelabel"] = {"status": "done", "message": f"{len(txt_files)} txt files", "updated_at": now_iso()}
        task["status"] = "prelabel_ready"
        save_task(path, task)
    append_event(path, f"YOLO labels uploaded: {len(txt_files)} txt files")
    return {"ok": True, "label_dir": str(final_dir), "txt_count": len(txt_files)}


@app.post("/api/tasks/{task_id}/split-video")
async def split_video(task_id: str, req: SplitVideoReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    if req.segment_length < 1:
        raise HTTPException(400, "segment_length must be >= 1")
    with TASK_LOCK:
        task = ensure_task_shape(load_task(path))
        record = video_record(task, req.video_id)
        if record is None:
            raise HTTPException(400, "No video available")
        record["split"] = {
            "status": "queued",
            "segment_length": req.segment_length,
            "label_source": req.label_source,
            "updated_at": now_iso(),
        }
        task["status"] = "split_queued"
        save_task(path, task)
    append_event(path, f"Queued split: {record['video_id']} every {req.segment_length} frames")
    threading.Thread(target=split_video_job, args=(task_id, req), daemon=True).start()
    return {"ok": True}


@app.post("/api/tasks/{task_id}/run-locateanything")
async def start_locany(task_id: str, req: RunLocateAnythingReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    threading.Thread(target=run_locany_job, args=(task_id, req), daemon=True).start()
    set_stage(path, "locateanything", "queued", "Queued")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/run-segment-locateanything")
async def start_segment_locany(task_id: str, req: RunLocateAnythingReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    threading.Thread(target=run_segment_locany_job, args=(task_id, req), daemon=True).start()
    set_stage(path, "locateanything", "queued", "Segment LocateAnything queued")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/run-tracking")
async def start_tracking(task_id: str, req: RunTrackingReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    threading.Thread(target=run_tracking_job, args=(task_id, req), daemon=True).start()
    set_stage(path, "tracking", "queued", "Queued")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/run-segment-tracking")
async def start_segment_tracking(task_id: str, req: RunTrackingReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    threading.Thread(target=run_segment_tracking_job, args=(task_id, req), daemon=True).start()
    set_stage(path, "tracking", "queued", "Segment tracking queued")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/package")
async def create_package(task_id: str, req: PackageReq):
    require_video_manager(task_id, req.actor)
    package = build_package(task_id, req.include_video)
    return {"ok": True, "package": str(package)}


@app.get("/api/tasks/{task_id}/package/download")
async def download_package(task_id: str, actor: str):
    path = task_dir(task_id)
    require_video_manager(task_id, actor)
    packages = sorted((path / "package").glob("*_annotation_package.zip"))
    if not packages:
        package = build_package(task_id, include_video=True)
    else:
        package = packages[-1]
    return FileResponse(str(package), filename=package.name, media_type="application/zip")


@app.post("/api/tasks/{task_id}/reviewed")
async def upload_reviewed(
    task_id: str,
    file: UploadFile = File(...),
    actor: str = Form(...),
):
    path = task_dir(task_id)
    require_task_participant(task_id, actor)
    dst = path / "reviewed" / "tracking_results.reviewed.json"
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with TASK_LOCK:
        task = load_task(path)
        task["reviewed_tracking_results"] = str(dst)
        task["stages"]["review"] = {"status": "done", "message": dst.name, "updated_at": now_iso()}
        task["status"] = "reviewed"
        save_task(path, task)
    append_event(path, "Reviewed tracking_results uploaded")
    return {"ok": True, "path": str(dst)}


@app.post("/api/tasks/{task_id}/export-yolo")
async def export_yolo(task_id: str, req: ActorReq):
    path = task_dir(task_id)
    require_video_manager(task_id, req.actor)
    reviewed = path / "reviewed" / "tracking_results.reviewed.json"
    source = reviewed if reviewed.exists() else path / "tracking" / "tracking_results.json"
    if not source.exists():
        raise HTTPException(400, "No tracking results to export")
    out = export_tracking_results_to_yolo(source, path / "exports" / "yolo")
    set_stage(path, "export", "done", f"YOLO exported to {out}")
    return {"ok": True, "output_dir": str(out)}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the annotation workflow platform.")
    parser.add_argument("--host", default=os.environ.get("ANNOTATION_PLATFORM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANNOTATION_PLATFORM_PORT", "8088")))
    parser.add_argument("--tasks-dir", type=Path, default=Path(os.environ.get("ANNOTATION_PLATFORM_TASKS_DIR", DEFAULT_TASKS_DIR)))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("ANNOTATION_PLATFORM_CONFIG", PROJECT_ROOT / "config.json")),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.environ["ANNOTATION_PLATFORM_DB"]) if os.environ.get("ANNOTATION_PLATFORM_DB") else None,
        help="SQLite database path. Defaults to <tasks-dir>/platform.sqlite3.",
    )
    args = parser.parse_args(argv)

    SETTINGS["tasks_dir"] = args.tasks_dir.expanduser().resolve()
    SETTINGS["project_config"] = args.config.expanduser().resolve()
    SETTINGS["database_path"] = str(args.database.expanduser().resolve()) if args.database else ""
    tasks_dir()
    database()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
