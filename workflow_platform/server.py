from __future__ import annotations

import argparse
import contextvars
import csv
import hashlib
import hmac
import io
import os
import re
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
from workflow_platform.tls import discover_tls_hosts, ensure_self_signed_certificate, normalize_tls_hosts


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PROJECT_ROOT = APP_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "platform_tasks"
API_SCHEMA_VERSION = 13
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
    "secure_cookie": os.environ.get("ANNOTATION_PLATFORM_SECURE_COOKIE", "0") == "1",
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
    part_count: int = 0
    part_prefix: str = ""
    part_manifest: str = ""


class AddPartsReq(BaseModel):
    count: int


class UpdateTaskReq(BaseModel):
    manager: Optional[str] = None
    product_tag: Optional[str] = None
    part_prefix: Optional[str] = None
    expected_part_seconds: Optional[float] = None
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


class UpdateTaskOrderingReq(BaseModel):
    rank: Optional[int] = None
    priority: Optional[str] = None


class SubmitPartReq(BaseModel):
    note: str = ""


class ReviewPartReq(BaseModel):
    action: str
    note: str = ""


class CommentReq(BaseModel):
    content: str


class ReturnPartReq(BaseModel):
    note: str = ""


class TimeReviewReq(BaseModel):
    decision: str
    note: str = ""


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


def _work_path(data_root: str, manifest_path: str) -> str:
    path = manifest_path.strip()
    if path.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", path):
        return path
    root = data_root.strip()
    if not root:
        raise ValueError("清单使用相对路径时，任务表中的数据路径不能为空")
    if path in {".", ".\\", "./"}:
        return root.rstrip("/\\")
    separator = "\\" if "\\" in root and "/" not in root else "/"
    return root.rstrip("/\\") + separator + path.strip("/\\").replace("/", separator).replace("\\", separator)


def parse_part_manifest(text: str, data_root: str) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), 1):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        if len(cells) > 2:
            raise ValueError(f"Part 清单第 {line_number} 行最多只能有两列：显示名和路径")
        display_name, raw_path = ("", cells[0]) if len(cells) == 1 else (cells[0], cells[1])
        if not raw_path:
            raise ValueError(f"Part 清单第 {line_number} 行缺少工作目录")
        work_path = _work_path(data_root, raw_path)
        key = work_path.replace("\\", "/").rstrip("/").casefold()
        if key in seen:
            raise ValueError(f"Part 清单包含重复工作目录：{work_path}")
        seen.add(key)
        if not display_name:
            if raw_path.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", raw_path):
                display_name = raw_path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
            else:
                display_name = (
                    "数据集根目录" if raw_path in {".", ".\\", "./"} else
                    " / ".join(part for part in re.split(r"[\\/]", raw_path.strip("/\\")) if part)
                )
        specs.append({"name": display_name, "work_path": work_path})
    if not specs:
        raise ValueError("Part 工作目录清单为空")
    if len(specs) > 10000:
        raise ValueError("Part 工作目录清单不能超过 10000 行")
    return specs


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
        secure=bool(SETTINGS.get("secure_cookie")),
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


@app.get("/api/user-options")
async def list_user_options():
    require_user()
    return {
        "users": [
            {"username": item["username"], "display_name": item["display_name"]}
            for item in database().list_users() if item["is_active"]
        ]
    }


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


@app.delete("/api/users/{username}")
async def delete_user(username: str):
    actor = require_admin()
    target_username = normalize_username(username)
    if target_username == actor["username"]:
        raise HTTPException(400, "不能删除当前登录账号")
    try:
        target = database().get_user(target_username)
    except KeyError as exc:
        raise HTTPException(404, "用户不存在") from exc
    if (
        target["role"] == "admin"
        and target["is_active"]
        and database().active_admin_count() <= 1
    ):
        raise HTTPException(400, "不能删除最后一个管理员")
    summary = database().delete_user(target_username, actor["username"], now_iso())
    return {"ok": True, "summary": summary}


@app.post("/api/tasks/preview")
async def preview_tasks(req: PublishTasksReq):
    require_user()
    try:
        rows = parse_spreadsheet_rows(req.clipboard_text)
        part_specs = None
        if req.part_manifest.strip():
            if len(rows) != 1:
                raise ValueError("使用 Part 工作目录清单时，一次只能发布一行任务")
            part_specs = parse_part_manifest(req.part_manifest, rows[0]["data_path"])
        elif req.part_count < 1 or req.part_count > 10000:
            raise ValueError("Part 数量必须在 1 到 10000 之间")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "rows": rows, "count": len(rows),
        "part_count": len(part_specs) if part_specs is not None else req.part_count,
        "part_preview": (part_specs or [])[:20],
    }


@app.post("/api/tasks")
async def publish_tasks(req: PublishTasksReq):
    actor = require_user()["username"]
    tag = req.product_tag.strip()
    if not tag:
        raise HTTPException(400, "请填写产品大标签")
    if not req.part_manifest.strip() and (req.part_count < 1 or req.part_count > 10000):
        raise HTTPException(400, "Part 数量必须在 1 到 10000 之间")
    try:
        rows = parse_spreadsheet_rows(req.clipboard_text)
        if req.part_manifest.strip() and len(rows) != 1:
            raise ValueError("使用 Part 工作目录清单时，一次只能发布一行任务")
        created = []
        for row in rows:
            part_specs = (
                parse_part_manifest(req.part_manifest, row["data_path"])
                if req.part_manifest.strip() else None
            )
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
            created.append(database().create_task(
                task, req.part_count, now, part_specs=part_specs, return_details=False,
            ))
    except BaseException as exc:
        _raise_database_error(exc)
    return {"tasks": [{key: item[key] for key in
                        ("task_id", "name", "project", "annotation_content", "part_prefix")}
                      for item in created], "count": len(created)}


@app.get("/api/tasks")
async def list_tasks():
    user = require_user()
    actor = user["username"]
    tasks = database().list_tasks(now_iso())
    for task in tasks:
        task["is_publisher"] = task["publisher"] == actor
        task["is_manager"] = bool(task.get("manager")) and task["manager"] == actor and not task["is_publisher"]
        task["can_review"] = task["is_publisher"] or task["is_manager"] or user.get("role") == "admin"
        task["can_view_all"] = task["can_review"] or task["is_manager"]
        task["can_manage_ordering"] = user.get("role") == "admin"
        mine = database().actor_part_summary(task["task_id"], actor)
        task["my_parts"] = mine["total"]
        task["my_rework"] = mine["rework"]
        task["my_active"] = mine["in_progress"]
    return {"tasks": tasks}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, part_status: str = "", part_query: str = "",
                   part_page: int = 1, part_page_size: int = 50):
    user = require_user()
    actor = user["username"]
    try:
        task = database().get_task(task_id, now_iso())
    except KeyError as exc:
        raise HTTPException(404, "任务不存在") from exc
    publisher = task["publisher"] == actor
    manager = bool(task.get("manager")) and task["manager"] == actor and not publisher
    can_review = publisher or manager or user.get("role") == "admin"
    can_view_all = can_review or manager
    task["is_publisher"] = publisher
    task["is_manager"] = manager
    task["can_review"] = can_review
    task["can_view_all"] = can_view_all
    task["can_manage_ordering"] = user.get("role") == "admin"
    task["available_parts"] = task["part_summary"]["pending"]
    if can_view_all:
        try:
            parts_page = database().list_parts_page(
                task_id, now_iso(), status=part_status, query=part_query,
                page=part_page, page_size=part_page_size,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        task["parts"] = parts_page.pop("items")
        task["parts_page"] = parts_page
    if publisher or manager:
        task["statistics"] = database().annotator_statistics(task_id, now_iso())
    elif not can_view_all:
        task["parts"] = database().list_parts(task_id, now_iso(), actor)
    task["audit_logs"] = database().list_task_audit_logs(task_id)
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
    if "expected_part_seconds" in changes:
        seconds = float(changes["expected_part_seconds"])
        if seconds < 0 or seconds > 31 * 86400:
            raise HTTPException(400, "每个 Part 预计耗时必须在 0 到 31 天之间")
        changes["expected_part_seconds"] = seconds
    if "manager" in changes:
        manager = str(changes["manager"]).strip()
        if manager:
            try:
                target = database().get_user(manager)
            except KeyError as exc:
                raise HTTPException(400, "指定的协同查看人不存在") from exc
            if not target["is_active"]:
                raise HTTPException(400, "指定的协同查看人已停用")
        changes["manager"] = manager
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


@app.patch("/api/tasks/{task_id}/ordering")
async def update_task_ordering(task_id: str, req: UpdateTaskOrderingReq):
    user = require_admin()
    try:
        task = database().update_task_ordering(
            task_id, user["username"], rank=req.rank, priority=req.priority,
            now=now_iso(), is_admin=True,
        )
    except BaseException as exc:
        _raise_database_error(exc)
    return {"task": task}


@app.post("/api/tasks/{task_id}/parts")
async def add_parts(task_id: str, req: AddPartsReq):
    actor = require_user()["username"]
    try:
        database().add_parts(task_id, req.count, actor, now_iso(), return_parts=False)
    except BaseException as exc:
        _raise_database_error(exc)
    return {"added": req.count, "summary": database().part_summary(task_id)}


@app.delete("/api/tasks/{task_id}/parts/{part_id}")
async def delete_part(task_id: str, part_id: int):
    actor = require_user()["username"]
    try:
        deleted = database().delete_part(task_id, part_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"deleted": deleted, "summary": database().part_summary(task_id)}


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


@app.post("/api/tasks/{task_id}/parts/{part_id}/pause")
async def pause_part(task_id: str, part_id: int):
    actor = require_user()["username"]
    try:
        part = database().pause_part(task_id, part_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/resume")
async def resume_part(task_id: str, part_id: int):
    actor = require_user()["username"]
    try:
        part = database().resume_part(task_id, part_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/return")
async def return_part(task_id: str, part_id: int, req: ReturnPartReq):
    actor = require_user()["username"]
    try:
        part = database().return_part(task_id, part_id, actor, req.note, now_iso())
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
    user = require_user()
    actor = user["username"]
    try:
        part = database().review_part(
            task_id, part_id, actor, req.action, req.note, now_iso(),
            is_admin=user.get("role") == "admin",
        )
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/{part_id}/time-review")
async def review_part_time(task_id: str, part_id: int, req: TimeReviewReq):
    actor = require_user()["username"]
    try:
        part = database().review_part_time(
            task_id, part_id, actor, req.decision, req.note, now_iso()
        )
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the annotation collaboration platform.")
    parser.add_argument("--host", default=os.environ.get("ANNOTATION_PLATFORM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANNOTATION_PLATFORM_PORT", "8000")))
    parser.add_argument("--tasks-dir", default=str(SETTINGS["data_dir"]))
    parser.add_argument("--database", default=str(database_path()))
    parser.add_argument(
        "--ssl-certfile", default=os.environ.get("ANNOTATION_PLATFORM_SSL_CERTFILE", ""),
        help="PEM certificate chain used to serve HTTPS.",
    )
    parser.add_argument(
        "--ssl-keyfile", default=os.environ.get("ANNOTATION_PLATFORM_SSL_KEYFILE", ""),
        help="PEM private key used to serve HTTPS.",
    )
    parser.add_argument(
        "--auto-https", action="store_true",
        default=os.environ.get("ANNOTATION_PLATFORM_AUTO_HTTPS", "0") == "1",
        help="Generate and reuse a self-signed certificate when explicit PEM files are absent.",
    )
    parser.add_argument(
        "--tls-hosts", default=os.environ.get("ANNOTATION_PLATFORM_TLS_HOSTS", ""),
        help="Comma-separated DNS names and IP addresses added to an auto-generated certificate.",
    )
    parser.add_argument(
        "--tls-cert-dir", default=os.environ.get("ANNOTATION_PLATFORM_TLS_CERT_DIR", ""),
        help="Directory for the auto-generated certificate; defaults to <tasks-dir>/tls.",
    )
    args = parser.parse_args(argv)
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("--ssl-certfile and --ssl-keyfile must be provided together")
    ssl_certfile = str(Path(args.ssl_certfile).expanduser().resolve()) if args.ssl_certfile else None
    ssl_keyfile = str(Path(args.ssl_keyfile).expanduser().resolve()) if args.ssl_keyfile else None
    if args.auto_https and not ssl_certfile:
        certificate_dir = (
            Path(args.tls_cert_dir).expanduser().resolve()
            if args.tls_cert_dir else Path(args.tasks_dir).expanduser().resolve() / "tls"
        )
        requested_hosts = normalize_tls_hosts(args.tls_hosts.split(","))
        hosts = normalize_tls_hosts([*discover_tls_hosts(args.host), *requested_hosts])
        try:
            generated_cert, generated_key = ensure_self_signed_certificate(
                certificate_dir / "selfsigned-cert.pem",
                certificate_dir / "selfsigned-key.pem",
                hosts,
            )
        except ValueError as exc:
            parser.error(str(exc))
        ssl_certfile, ssl_keyfile = str(generated_cert), str(generated_key)
        print(f"Auto HTTPS certificate: {ssl_certfile}")
        print(f"Auto HTTPS hosts: {', '.join(hosts)}")
    if ssl_certfile and not Path(ssl_certfile).is_file():
        parser.error(f"SSL certificate file does not exist: {ssl_certfile}")
    if ssl_keyfile and not Path(ssl_keyfile).is_file():
        parser.error(f"SSL private key file does not exist: {ssl_keyfile}")
    SETTINGS["data_dir"] = Path(args.tasks_dir).expanduser().resolve()
    SETTINGS["database_path"] = str(Path(args.database).expanduser().resolve())
    SETTINGS["secure_cookie"] = bool(ssl_certfile) or (
        os.environ.get("ANNOTATION_PLATFORM_SECURE_COOKIE", "0") == "1"
    )
    database()
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    main()
