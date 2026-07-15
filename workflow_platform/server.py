from __future__ import annotations

import argparse
import contextvars
import csv
import hashlib
import hmac
import io
import os
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from workflow_platform.database import PlatformDatabase, TASK_FIELDS


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PROJECT_ROOT = APP_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "platform_tasks"
API_SCHEMA_VERSION = 6
SESSION_COOKIE_NAME = "annotation_platform_session"
SESSION_TTL_DAYS = int(os.environ.get("ANNOTATION_PLATFORM_SESSION_DAYS", "7"))
PASSWORD_ITERATIONS = 200_000

SPREADSHEET_HEADERS = (
    "申请日期", "申请人", "项目", "标注内容", "数据集溯源", "每小时可标",
    "数据量", "预计工时/单人", "数据路径", "标注说明书路径",
)
HEADER_TO_FIELD = dict(zip(SPREADSHEET_HEADERS, TASK_FIELDS))

app = FastAPI(title="多人标注协作平台")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SETTINGS: dict[str, Any] = {
    "data_dir": Path(os.environ.get("ANNOTATION_PLATFORM_TASKS_DIR", DEFAULT_DATA_DIR)).resolve(),
    "database_path": os.environ.get("ANNOTATION_PLATFORM_DB", ""),
}
_DATABASE: Optional[PlatformDatabase] = None
_DATABASE_KEY = ""
_DATABASE_LOCK = threading.Lock()
CURRENT_USER: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "workflow_platform_current_user", default=None
)


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


class PublishTasksReq(BaseModel):
    clipboard_text: str
    product_tag: str
    part_count: int
    part_prefix: str = ""


class AddPartsReq(BaseModel):
    count: int


class UpdateTaskReq(BaseModel):
    product_tag: Optional[str] = None
    part_prefix: Optional[str] = None
    application_date: Optional[str] = None
    applicant: Optional[str] = None
    project: Optional[str] = None
    annotation_content: Optional[str] = None
    dataset_source: Optional[str] = None
    hourly_capacity: Optional[str] = None
    data_amount: Optional[str] = None
    estimated_hours: Optional[str] = None
    data_path: Optional[str] = None
    guide_path: Optional[str] = None


class SubmitPartReq(BaseModel):
    note: str = ""


class ReviewPartReq(BaseModel):
    action: str
    note: str = ""


class CommentReq(BaseModel):
    content: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def database_path() -> Path:
    configured = str(SETTINGS.get("database_path", "")).strip()
    return Path(configured).expanduser().resolve() if configured else Path(SETTINGS["data_dir"]) / "metadata.sqlite3"


def database() -> PlatformDatabase:
    global _DATABASE, _DATABASE_KEY
    path = database_path()
    key = str(path)
    with _DATABASE_LOCK:
        if _DATABASE is None or _DATABASE_KEY != key:
            _DATABASE = PlatformDatabase(path)
            _DATABASE.initialize()
            _DATABASE_KEY = key
        return _DATABASE


def normalize_username(username: str) -> str:
    value = username.strip()
    if not value or any(ch.isspace() for ch in value):
        raise HTTPException(400, "用户名不能为空且不能包含空格")
    return value


def validate_password(password: str) -> str:
    if len(password) < 8:
        raise HTTPException(400, "密码至少需要 8 个字符")
    return password


def validate_role(role: str) -> str:
    value = role.strip().lower()
    if value not in {"admin", "user"}:
        raise HTTPException(400, "角色必须是 admin 或 user")
    return value


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("ascii"), PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("ascii"), int(iterations)
        ).hex()
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def session_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {key: user.get(key) for key in (
        "username", "role", "display_name", "is_active", "created_at", "updated_at", "last_login_at"
    )}


def require_user() -> dict[str, Any]:
    user = CURRENT_USER.get()
    if user is None:
        raise HTTPException(401, "请先登录")
    return user


def require_admin() -> dict[str, Any]:
    user = require_user()
    if user.get("role") != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user


def _raise_database_error(exc: BaseException) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(404, "任务或 Part 不存在") from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(403, str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(400, str(exc)) from exc
    raise exc


def parse_spreadsheet_rows(text: str) -> list[dict[str, str]]:
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(text.strip()), delimiter="\t")
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        raise ValueError("请粘贴任务表格中的至少一行")

    first = rows[0]
    recognized = sum(1 for cell in first if cell in HEADER_TO_FIELD)
    if recognized >= 3:
        mapping = [HEADER_TO_FIELD.get(cell) for cell in first]
        data_rows = rows[1:]
    else:
        mapping = list(TASK_FIELDS)
        data_rows = rows
    if not data_rows:
        raise ValueError("只检测到表头，没有任务数据")

    result = []
    for row in data_rows:
        record = {field: "" for field in TASK_FIELDS}
        for index, value in enumerate(row):
            if index < len(mapping) and mapping[index]:
                record[str(mapping[index])] = value
        if not record["project"] and not record["annotation_content"]:
            continue
        result.append(record)
    if not result:
        raise ValueError("没有识别到有效任务；请使用制表符分隔的十列格式")
    return result


@app.middleware("http")
async def authentication(request: Request, call_next):
    token = CURRENT_USER.set(None)
    try:
        raw = request.cookies.get(SESSION_COOKIE_NAME, "")
        if raw:
            user = database().get_session_user(session_token_hash(raw), now_iso())
            if user:
                CURRENT_USER.set(public_user(user))
        path = request.url.path
        public_api = path in {
            "/api/health", "/api/auth/me", "/api/auth/login", "/api/auth/bootstrap-admin"
        }
        if path.startswith("/api/") and not public_api and CURRENT_USER.get() is None:
            return JSONResponse({"detail": "请先登录"}, status_code=401)
        return await call_next(request)
    finally:
        CURRENT_USER.reset(token)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "service": "annotation-collaboration-platform",
        "api_schema_version": API_SCHEMA_VERSION,
        "database": database().health(),
    }


@app.get("/api/auth/me")
async def auth_me():
    return {
        "authenticated": CURRENT_USER.get() is not None,
        "user": CURRENT_USER.get(),
        "bootstrap_required": database().user_count() == 0,
    }


def _set_session(response: Response, username: str) -> dict[str, Any]:
    raw = secrets.token_urlsafe(32)
    now = now_iso()
    expiry = (datetime.now(timezone.utc).astimezone() + timedelta(days=max(1, SESSION_TTL_DAYS))).isoformat(timespec="seconds")
    database().create_session(uuid.uuid4().hex, username, session_token_hash(raw), expiry, now)
    database().touch_user_login(username, now)
    response.set_cookie(
        SESSION_COOKIE_NAME, raw, httponly=True, samesite="lax",
        secure=os.environ.get("ANNOTATION_PLATFORM_SECURE_COOKIE", "0") == "1",
        max_age=SESSION_TTL_DAYS * 86400,
    )
    return public_user(database().get_user(username))


@app.post("/api/auth/bootstrap-admin")
async def bootstrap_admin(req: BootstrapAdminReq, response: Response):
    if database().user_count() != 0:
        raise HTTPException(409, "管理员已经初始化")
    username = normalize_username(req.username)
    password = validate_password(req.password)
    database().create_user(username, hash_password(password), "admin", req.display_name.strip(), now_iso())
    return {"user": _set_session(response, username)}


@app.post("/api/auth/login")
async def login(req: LoginReq, response: Response):
    try:
        user = database().get_user(normalize_username(req.username), include_password=True)
    except KeyError as exc:
        raise HTTPException(401, "用户名或密码错误") from exc
    if not user["is_active"] or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    return {"user": _set_session(response, user["username"])}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    raw = request.cookies.get(SESSION_COOKIE_NAME, "")
    if raw:
        database().delete_session(session_token_hash(raw))
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.post("/api/auth/change-password")
async def change_password(req: ChangePasswordReq):
    user = require_user()
    stored = database().get_user(user["username"], include_password=True)
    if not verify_password(req.old_password, stored["password_hash"]):
        raise HTTPException(400, "当前密码错误")
    database().update_user(
        user["username"], password_hash=hash_password(validate_password(req.new_password)), now=now_iso()
    )
    return {"ok": True}


@app.get("/api/users")
async def list_users():
    require_admin()
    return {"users": database().list_users()}


@app.post("/api/users")
async def create_user(req: CreateUserReq):
    require_admin()
    try:
        user = database().create_user(
            normalize_username(req.username), hash_password(validate_password(req.password)),
            validate_role(req.role), req.display_name.strip(), now_iso(), req.is_active,
        )
    except Exception as exc:
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(409, "用户名已存在") from exc
        raise
    return {"user": user}


@app.patch("/api/users/{username}")
async def update_user(username: str, req: UpdateUserReq):
    actor = require_admin()
    target = database().get_user(username)
    role = validate_role(req.role) if req.role is not None else None
    if target["role"] == "admin" and target["is_active"] and (
        role == "user" or req.is_active is False
    ) and database().active_admin_count() <= 1:
        raise HTTPException(400, "不能停用或降级最后一个管理员")
    if username == actor["username"] and req.is_active is False:
        raise HTTPException(400, "不能停用当前账号")
    user = database().update_user(
        username, role=role, display_name=req.display_name,
        password_hash=hash_password(validate_password(req.password)) if req.password else None,
        is_active=req.is_active, now=now_iso(),
    )
    return {"user": user}


@app.post("/api/tasks/preview")
async def preview_tasks(req: PublishTasksReq):
    require_user()
    try:
        rows = parse_spreadsheet_rows(req.clipboard_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"rows": rows, "count": len(rows)}


@app.post("/api/tasks")
async def publish_tasks(req: PublishTasksReq):
    actor = require_user()["username"]
    tag = req.product_tag.strip()
    if not tag:
        raise HTTPException(400, "请填写产品大标签")
    if req.part_count < 1 or req.part_count > 10000:
        raise HTTPException(400, "Part 数量必须在 1 到 10000 之间")
    try:
        rows = parse_spreadsheet_rows(req.clipboard_text)
        created = []
        for row in rows:
            now = now_iso()
            project = row["project"] or "未命名项目"
            content = row["annotation_content"] or "标注任务"
            task = {
                "task_id": f"task-{uuid.uuid4().hex[:12]}",
                "name": f"{project} · {content}",
                "status": "ready",
                "publisher": actor,
                "manager": actor,
                "product_tag": tag,
                "part_prefix": req.part_prefix.strip(),
                **row,
                "created_at": now,
                "updated_at": now,
            }
            created.append(database().create_task(task, req.part_count, now))
    except BaseException as exc:
        _raise_database_error(exc)
    return {"tasks": created, "count": len(created)}


@app.get("/api/tasks")
async def list_tasks():
    actor = require_user()["username"]
    tasks = database().list_tasks(now_iso())
    for task in tasks:
        task["is_publisher"] = task["publisher"] == actor
        mine = database().list_parts(task["task_id"], now_iso(), actor)
        task["my_parts"] = len(mine)
        task["my_rework"] = sum(1 for part in mine if part["status"] == "rework")
        task["my_active"] = sum(1 for part in mine if part["status"] == "in_progress")
    return {"tasks": tasks}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    actor = require_user()["username"]
    try:
        task = database().get_task(task_id, now_iso())
    except KeyError as exc:
        raise HTTPException(404, "任务不存在") from exc
    publisher = task["publisher"] == actor
    task["is_publisher"] = publisher
    task["available_parts"] = task["part_summary"]["pending"]
    if publisher:
        task["parts"] = database().list_parts(task_id, now_iso())
        task["statistics"] = database().annotator_statistics(task_id, now_iso())
    else:
        task["parts"] = database().list_parts(task_id, now_iso(), actor)
    return task


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, req: UpdateTaskReq):
    actor = require_user()["username"]
    changes = {
        field: value for field, value in req.model_dump(exclude_unset=True).items()
        if value is not None
    }
    if not changes:
        raise HTTPException(400, "请至少修改一项任务信息")
    if "product_tag" in changes and not changes["product_tag"].strip():
        raise HTTPException(400, "产品大标签不能为空")
    try:
        task = database().update_task(task_id, actor, changes, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"task": task}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    actor = require_user()["username"]
    try:
        database().delete_task(task_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/parts")
async def add_parts(task_id: str, req: AddPartsReq):
    actor = require_user()["username"]
    try:
        parts = database().add_parts(task_id, req.count, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"parts": parts, "summary": database().part_summary(task_id)}


@app.post("/api/tasks/{task_id}/parts/claim-next")
async def claim_next_part(task_id: str):
    actor = require_user()["username"]
    try:
        part = database().claim_next_part(task_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    if part is None:
        raise HTTPException(409, "当前没有可领取的 Part")
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/start-rework")
async def start_rework(task_id: str, part_id: int):
    actor = require_user()["username"]
    try:
        part = database().start_rework(task_id, part_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/submit")
async def submit_part(task_id: str, part_id: int, req: SubmitPartReq):
    actor = require_user()["username"]
    try:
        part = database().submit_part(task_id, part_id, actor, req.note, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/comments")
async def add_comment(task_id: str, part_id: int, req: CommentReq):
    actor = require_user()["username"]
    try:
        part = database().add_part_comment(task_id, part_id, actor, req.content, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/review")
async def review_part(task_id: str, part_id: int, req: ReviewPartReq):
    actor = require_user()["username"]
    try:
        part = database().review_part(task_id, part_id, actor, req.action, req.note, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the annotation collaboration platform.")
    parser.add_argument("--host", default=os.environ.get("ANNOTATION_PLATFORM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANNOTATION_PLATFORM_PORT", "8000")))
    parser.add_argument("--tasks-dir", default=str(SETTINGS["data_dir"]))
    parser.add_argument("--database", default=str(database_path()))
    args = parser.parse_args(argv)
    SETTINGS["data_dir"] = Path(args.tasks_dir).expanduser().resolve()
    SETTINGS["database_path"] = str(Path(args.database).expanduser().resolve())
    database()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
