from __future__ import annotations

import argparse
import asyncio
import logging
import base64
import contextvars
import csv
import hashlib
import hmac
import io
import math
import os
import re
import secrets
import string
import threading
import uuid
import urllib.parse
import urllib.request
import json
from contextlib import asynccontextmanager, suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workflow_platform.database import PlatformDatabase, TASK_FIELDS
from workflow_platform.tls import discover_tls_hosts, ensure_self_signed_certificate, normalize_tls_hosts


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PROJECT_ROOT = APP_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "platform_tasks"
API_SCHEMA_VERSION = 18
SESSION_COOKIE_NAME = "annotation_platform_session"
SESSION_TTL_DAYS = int(os.environ.get("ANNOTATION_PLATFORM_SESSION_DAYS", "7"))
PASSWORD_ITERATIONS = 200_000

SPREADSHEET_HEADERS = (
    "申请日期", "申请人", "项目", "标注内容", "数据集溯源", "每小时可标",
    "数据量", "预计工时/单人", "数据路径", "标注说明书路径",
)
HEADER_TO_FIELD = dict(zip(SPREADSHEET_HEADERS, TASK_FIELDS))

async def automatic_report_loop():
    while True:
        try:
            await asyncio.to_thread(run_automatic_report, now_iso())
            await asyncio.to_thread(process_reminder_events)
        except Exception:
            logging.getLogger(__name__).exception("Automatic daily report check failed")
        await asyncio.sleep(30)


@asynccontextmanager
async def platform_lifespan(app):
    worker = asyncio.create_task(automatic_report_loop())
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


app = FastAPI(title="多人标注协作平台", lifespan=platform_lifespan)
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
    phone: str = ""
    is_active: bool = True


class UpdateUserReq(BaseModel):
    role: Optional[str] = None
    display_name: Optional[str] = None
    phone: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class ChangePasswordReq(BaseModel):
    old_password: str
    new_password: str


class PublishTasksReq(BaseModel):
    clipboard_text: str = ""
    product_tag: str
    part_count: int = 0
    part_prefix: str = ""
    part_manifest: str = ""
    annotation_tool: str = ""
    input_mode: str = "table"
    task: Optional[dict[str, str]] = None


class AddPartsReq(BaseModel):
    count: int = 0
    part_manifest: str = ""


class UpdateTaskReq(BaseModel):
    annotation_tool: Optional[str] = None
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


class AttendanceSettingsReq(BaseModel):
    work_start: str
    work_end: str


class DingRobotReq(BaseModel):
    name: str
    webhook_url: str
    secret: str = ""


class DingRecipientReq(BaseModel):
    name: str
    phone: str
    username: str = ""


class DingTestReq(BaseModel):
    robot_id: int
    content: str = "测试通知"
    recipient_ids: list[int] = []
    usernames: list[str] = Field(default_factory=list)


class TaskNotificationReq(BaseModel):
    kind: str = "task"
    task_id: str = ""
    robot_id: int
    usernames: list[str] = Field(default_factory=list)
    note: str = ""


class AutomaticReportSettingsReq(BaseModel):
    enabled: bool = False
    robot_id: Optional[int] = None
    usernames: list[str] = Field(default_factory=list)
    note: str = ""


class EventReminderRule(BaseModel):
    enabled: bool = False
    robot_id: Optional[int] = None
    usernames: list[str] = Field(default_factory=list)
    notify_publisher: bool = False


class ReminderManagementReq(BaseModel):
    rules: dict[str, EventReminderRule]
    task_template: str
    daily_template: str

class UpdatePartReq(BaseModel):
    name: Optional[str] = None
    work_path: Optional[str] = None
    image_count: Optional[int] = None
    progress_count: Optional[int] = None
    bbox_count: Optional[int] = Field(default=None, strict=True, ge=0, le=2**53 - 1)
    work_seconds: Optional[float] = Field(default=None, strict=True, ge=0, le=31 * 86400)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def working_day_completion(start: date, work_days: int) -> date:
    current = start
    remaining = work_days
    while True:
        if current.weekday() < 5:
            remaining -= 1
            if remaining <= 0:
                return current
        current += timedelta(days=1)


def statistics_period_bounds(period: str, anchor: str = "", *, now: Optional[str] = None) -> tuple[str, str]:
    """Resolve a day/week/month report window in the platform's local timezone."""
    value = period.strip().lower()
    if value not in {"day", "week", "month"}:
        raise ValueError("统计周期必须是 day、week 或 month")
    current = datetime.fromisoformat(now) if now else datetime.now().astimezone()
    tz = current.tzinfo or timezone.utc
    try:
        selected = date.fromisoformat(anchor) if anchor.strip() else current.date()
    except ValueError as exc:
        raise ValueError("统计日期必须使用 YYYY-MM-DD 格式") from exc
    start_date = selected
    if value == "week":
        start_date = selected - timedelta(days=selected.weekday())
    elif value == "month":
        start_date = selected.replace(day=1)
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
    if value == "day":
        end = start + timedelta(days=1)
    elif value == "week":
        end = start + timedelta(days=7)
    else:
        next_month = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = datetime.combine(next_month, datetime.min.time(), tzinfo=tz)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


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
        "username", "role", "display_name", "phone", "is_active", "created_at", "updated_at", "last_login_at"
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


def parse_part_manifest(text: str, data_root: str) -> list[dict[str, Any]]:
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), 1):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        if len(cells) > 3:
            raise ValueError(f"Part 清单第 {line_number} 行最多只能有三列：显示名、路径和图像数量")
        if len(cells) == 1:
            display_name, raw_path, image_count = "", cells[0], None
        elif len(cells) == 2:
            if cells[1].isdigit():
                display_name, raw_path, image_count = "", cells[0], int(cells[1])
            else:
                display_name, raw_path, image_count = cells[0], cells[1], None
        else:
            display_name, raw_path = cells[0], cells[1]
            try:
                image_count = int(cells[2])
                if image_count < 0: raise ValueError
            except ValueError as exc:
                raise ValueError(f"Part 清单第 {line_number} 行图像数量必须是非负整数") from exc
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
        spec: dict[str, Any] = {"name": display_name, "work_path": work_path}
        if image_count is not None:
            spec["image_count"] = image_count
        specs.append(spec)
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


@app.get("/overview")
async def overview_page():
    require_user()
    return FileResponse(STATIC_DIR / "overview.html")


@app.get("/users")
async def users_page():
    require_admin()
    return FileResponse(STATIC_DIR / "users.html")


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


@app.get("/api/attendance")
async def attendance_status():
    user = require_user()
    today = datetime.now().astimezone().date().isoformat()
    return database().attendance_status(user["username"], today, now_iso())

@app.post("/api/attendance/check-in")
async def attendance_check_in():
    user = require_user()
    today = datetime.now().astimezone().date().isoformat()
    return database().check_in(user["username"], today, now_iso())


@app.get("/api/attendance/settlements")
async def attendance_settlements(days: int = 30, date: str = ""):
    require_user()
    selected = date or datetime.now().astimezone().date().isoformat()
    try:
        datetime.strptime(selected, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(400, "日期必须使用 YYYY-MM-DD 格式") from exc
    return database().attendance_settlements(selected, days)

@app.get("/api/admin/dingtalk")
async def dingtalk_config():
    require_admin()
    return {"robots": database().list_ding_robots(), "recipients": database().list_ding_recipients()}

@app.post("/api/admin/dingtalk/robots")
async def add_dingtalk_robot(req: DingRobotReq):
    require_admin()
    if not req.name.strip() or not req.webhook_url.strip():
        raise HTTPException(400, "机器人名称和 Webhook 地址不能为空")
    robot_id = database().save_ding_robot(req.name.strip(), req.webhook_url.strip(), req.secret, now_iso())
    return {"robot_id": robot_id}

@app.delete("/api/admin/dingtalk/robots/{robot_id}")
async def remove_dingtalk_robot(robot_id: int):
    require_admin()
    try:
        database().delete_ding_robot(robot_id)
    except KeyError as exc:
        raise HTTPException(404, "机器人不存在") from exc
    return {"ok": True}

@app.post("/api/admin/dingtalk/recipients")
async def add_dingtalk_recipient(req: DingRecipientReq):
    require_admin()
    if not req.name.strip() or not req.phone.strip():
        raise HTTPException(400, "通知人姓名和手机号不能为空")
    try:
        recipient_id = database().save_ding_recipient(req.name.strip(), req.phone.strip(), req.username.strip(), now_iso())
    except Exception as exc:
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(409, "手机号已存在") from exc
        raise
    return {"recipient_id": recipient_id}

@app.delete("/api/admin/dingtalk/recipients/{recipient_id}")
async def remove_dingtalk_recipient(recipient_id: int):
    require_admin()
    try:
        database().delete_ding_recipient(recipient_id)
    except KeyError as exc:
        raise HTTPException(404, "通知人不存在") from exc
    return {"ok": True}

def send_dingtalk(req: DingTestReq):
    try:
        robot = database().get_ding_robot(req.robot_id)
        users = {item["username"]: item for item in database().list_users()}
        mobiles = []
        for username in req.usernames:
            target = users.get(username)
            if not target or not target["is_active"] or not target["phone"].strip():
                raise HTTPException(400, "请选择已启用且填写手机号的用户")
            if target["phone"].strip() not in mobiles:
                mobiles.append(target["phone"].strip())
        if not req.content.strip():
            raise HTTPException(400, "通知内容不能为空")
        timestamp = str(int(datetime.now().timestamp() * 1000))
        sign = base64.b64encode(hmac.new(robot["secret"].encode(), f"{timestamp}\n{robot['secret']}".encode(), hashlib.sha256).digest()).decode()
        separator = "&" if "?" in robot["webhook_url"] else "?"
        url = robot["webhook_url"] + separator + urllib.parse.urlencode({"timestamp": timestamp, "sign": sign})
        payload = {"msgtype": "text", "text": {"content": req.content}, "at": {"atMobiles": mobiles, "isAtAll": False}}
        request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
        if result.get("errcode", 0) != 0:
            raise HTTPException(502, result.get("errmsg", "钉钉返回失败"))
        return {"ok": True, "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"钉钉通知发送失败：{exc}") from exc


@app.post("/api/admin/dingtalk/test")
def test_dingtalk(req: DingTestReq):
    require_admin()
    return send_dingtalk(req)


def validate_notification_targets(robot_id: Optional[int], usernames: list[str]) -> None:
    try:
        robot = database().get_ding_robot(robot_id or 0)
    except KeyError as exc:
        raise HTTPException(400, "请选择可用机器人") from exc
    if not robot["is_active"]:
        raise HTTPException(400, "机器人已停用")
    users = {item["username"]: item for item in database().list_users()}
    if any(name not in users or not users[name]["is_active"] or not users[name]["phone"].strip() for name in usernames):
        raise HTTPException(400, "请选择已启用且填写手机号的用户")



DEFAULT_TASK_TEMPLATE = (
    "{event}\n标签：{product_tag}\n项目：{project}\n内容：{annotation_content}\n"
    "发布者：{publisher}\n排名：{rank}；优先级：{priority}\n"
    "状态：{status}；Part：{part_progress}\n预计工时：{expected_hours} 小时\n"
    "预计完成：{expected_completion}\n备注：{note}"
)
DEFAULT_DAILY_TEMPLATE = (
    "下班日报 · {date}\n今日报到：{attendance_count} 人\n今日完成：{completed_parts} 个 Part\n"
    "完成工时：预计 {expected_hours} 小时；实际 {actual_hours} 小时\n\n"
    "任务表格\n{task_table}\n\n备注：{note}"
)
EVENT_NAMES = {"published": "新任务发布", "priority": "任务优先级变动", "started": "任务开始", "completed": "任务完成"}
TEMPLATE_FIELDS = {"event", "content", "date", "note", "project", "product_tag", "annotation_content", "publisher", "priority", "rank", "expected_completion", "attendance_count", "completed_parts", "expected_hours", "actual_hours"}
TEMPLATE_FIELDS.update({"status", "part_progress", "task_table"})


def reminder_settings() -> dict[str, Any]:
    stored = database().reminder_config()
    task_template = stored.get("task_template", DEFAULT_TASK_TEMPLATE)
    daily_template = stored.get("daily_template", DEFAULT_DAILY_TEMPLATE)
    if task_template == "{event}\n{content}":
        task_template = DEFAULT_TASK_TEMPLATE
    if daily_template == "{content}":
        daily_template = DEFAULT_DAILY_TEMPLATE
    return {"rules": {key: stored.get("rules", {}).get(key, EventReminderRule().model_dump()) for key in EVENT_NAMES},
            "task_template": task_template,
            "daily_template": daily_template,
            "defaults": {"task_template": DEFAULT_TASK_TEMPLATE, "daily_template": DEFAULT_DAILY_TEMPLATE}}


def validate_reminder_template(template: str) -> None:
    if not template.strip() or len(template) > 8000:
        raise HTTPException(400, "模板不能为空且长度不能超过 8000 字符")
    try:
        for literal, field, specification, conversion in string.Formatter().parse(template):
            if field is not None and (field not in TEMPLATE_FIELDS or specification or conversion):
                raise ValueError()
    except ValueError as exc:
        raise HTTPException(400, "模板包含不支持的占位符或格式；使用页面列出的 {变量名}") from exc


@app.get("/reminders")
async def reminders_page():
    require_admin()
    return FileResponse(STATIC_DIR / "reminders.html")


@app.get("/api/admin/reminders")
def get_reminder_management():
    require_admin()
    return {**reminder_settings(), "history": database().reminder_history()}


@app.patch("/api/admin/reminders")
def update_reminder_management(req: ReminderManagementReq):
    require_admin()
    if set(req.rules) != set(EVENT_NAMES):
        raise HTTPException(400, "请配置发布、优先级、开始、完成四种任务事件")
    validate_reminder_template(req.task_template)
    validate_reminder_template(req.daily_template)
    for rule in req.rules.values():
        if rule.enabled:
            validate_notification_targets(rule.robot_id, rule.usernames)
    database().save_reminder_config(req.model_dump())
    return reminder_settings()


def enqueue_task_reminder(event: str, task_id: str, actor: str, event_id: str = "", note: str = "") -> None:
    try:
        rule = reminder_settings()["rules"][event]
        if not rule["enabled"]:
            return
        task = database().get_task(task_id, now_iso())
        users = {user["username"]: user for user in database().list_users()}
        names = list(rule["usernames"])
        if rule["notify_publisher"] and task["publisher"] not in names:
            names.append(task["publisher"])
        names = [name for name in names if name in users and users[name]["is_active"] and users[name]["phone"].strip()]
        request = TaskNotificationReq(kind="task", task_id=task_id, robot_id=rule["robot_id"], usernames=names, note=note)
        content = notification_content(request, now_iso(), event=EVENT_NAMES[event])
        payload = DingTestReq(robot_id=rule["robot_id"], usernames=names, content=content).model_dump()
        database().enqueue_reminder(event_id or uuid.uuid4().hex, event, task_id, payload, now_iso())
    except Exception:
        logging.getLogger(__name__).exception("Could not queue task reminder for %s", task_id)


def process_reminder_events() -> None:
    for attempt in range(20):
        event = database().claim_reminder()
        if event is None:
            return
        try:
            send_dingtalk(DingTestReq(**json.loads(event["payload"])))
            database().finish_reminder(event["event_id"], "sent", "已发送")
        except Exception:
            database().finish_reminder(event["event_id"], "failed", "发送失败，请检查机器人及用户手机号")

def notification_content(req: TaskNotificationReq, now: str, event: str = "任务提醒") -> str:
    overview = build_task_overview(now)
    statuses = {"awaiting_parts": "待补充 Part", "pending": "待领取", "in_progress": "执行中", "submitted": "待审核", "completed": "已完成", "mixed": "混合状态"}
    priorities = {"low": "低", "medium": "中", "high": "高", "urgent": "紧急"}
    if req.kind == "task":
        task = next((item for item in overview["tasks"] if item["task_id"] == req.task_id), None)
        if task is None:
            raise HTTPException(404, "任务不存在")
        summary = task["part_summary"]
        lines = ["任务提醒", f"标签：{task['product_tag']}", f"项目：{task['project']}", f"内容：{task['annotation_content']}",
                 f"排名：{task['rank']}；优先级：{priorities.get(task['priority'], task['priority'])}",
                 f"状态：{statuses[task['overview_status']]}；Part：{summary['completed']} / {summary['total']}",
                 f"预计工时：{task['estimated_completion_hours']:.2f} 小时", f"预计完成：{task['estimated_completion_date'] or '待估算'}"]
    elif req.kind == "daily":
        work_date = datetime.fromisoformat(now).date().isoformat()
        work = database().attendance_settlements(work_date, 1)["work"]["parts"]
        expected = sum(float(item["expected_hours"]) for item in work)
        actual = sum(float(item["actual_seconds"] or 0) for item in work) / 3600
        lines = [f"下班日报 · {work_date}", f"今日报到：{overview['attendance']['checked_in_count']} 人", f"今日完成：{len(work)} 个 Part",
                 f"完成工时：预计 {expected:.2f} 小时；实际 {actual:.2f} 小时", "", "任务表格", "排名 | 优先级 | 标签 | 项目 | 内容 | 状态 | Part进度 | 预计工时 | 预计完成"]
        for task in overview["tasks"]:
            values = [task["rank"], priorities.get(task["priority"], task["priority"]), task["product_tag"], task["project"], task["annotation_content"],
                      statuses[task["overview_status"]], f"{task['part_summary']['completed']}/{task['part_summary']['total']}",
                      f"{task['estimated_completion_hours']:.2f}h", task["estimated_completion_date"] or "待估算"]
            lines.append(" | ".join(str(value).replace("\n", " ").replace("\r", " ") for value in values))
    else:
        raise HTTPException(400, "不支持的通知类型")
    if req.note.strip():
        lines.extend(["", "备注：" + req.note.strip()])
    config = reminder_settings()
    template = config["task_template"] if req.kind == "task" else config["daily_template"]
    values = {field: "" for field in TEMPLATE_FIELDS}
    values.update(event=event, content="\n".join(lines), date=datetime.fromisoformat(now).date().isoformat(), note=req.note,
                  attendance_count=overview["attendance"]["checked_in_count"])
    if req.kind == "task":
        values.update({field: task.get(field, "") for field in ("project", "product_tag", "annotation_content", "publisher", "priority", "rank")})
        values.update(expected_completion=task["estimated_completion_date"] or "待估算", expected_hours=f"{task['estimated_completion_hours']:.2f}")
        values.update(priority=priorities.get(task["priority"], task["priority"]), status=statuses[task["overview_status"]],
                      part_progress=f"{task['part_summary']['completed']} / {task['part_summary']['total']}")
    else:
        values.update(completed_parts=len(work), expected_hours=f"{expected:.2f}", actual_hours=f"{actual:.2f}")
        table_start = lines.index("任务表格") + 1
        values["task_table"] = "\n".join(lines[table_start:table_start + len(overview["tasks"]) + 1])
    return template.format_map(values)


@app.get("/api/admin/notifications")
def notification_settings():
    require_admin()
    return database().notification_settings()


@app.patch("/api/admin/notifications")
def update_notification_settings(req: AutomaticReportSettingsReq):
    require_admin()
    if req.enabled:
        validate_notification_targets(req.robot_id, req.usernames)
    return database().save_notification_settings(req.model_dump(), now_iso())


@app.post("/api/admin/notifications/preview")
def preview_notification(req: TaskNotificationReq):
    require_admin()
    return {"content": notification_content(req, now_iso())}


@app.post("/api/admin/notifications/send")
def send_notification(req: TaskNotificationReq):
    require_admin()
    validate_notification_targets(req.robot_id, req.usernames)
    content = notification_content(req, now_iso())
    return send_dingtalk(DingTestReq(robot_id=req.robot_id, usernames=req.usernames, content=content))


def run_automatic_report(now: str) -> None:
    db = database()
    config = db.notification_settings()
    current = datetime.fromisoformat(now)
    work_date = current.date().isoformat()
    settings = db.attendance_settings(now)
    if not config["enabled"] or current.weekday() >= 5 or current.strftime("%H:%M") < settings["work_end"]:
        return
    if not db.claim_automatic_report(work_date, now):
        return
    try:
        validate_notification_targets(config["robot_id"], config["usernames"])
        req = TaskNotificationReq(kind="daily", robot_id=config["robot_id"], usernames=config["usernames"], note=config["note"])
        content = notification_content(req, now)
        send_dingtalk(DingTestReq(robot_id=req.robot_id, usernames=req.usernames, content=content))
        db.finish_automatic_report(work_date, "sent", "日报已发送", now)
    except Exception:
        db.finish_automatic_report(work_date, "failed", "发送失败，请检查机器人及用户手机号后手动发送", now)

@app.get("/api/admin/attendance")
async def admin_attendance():
    require_admin()
    today = datetime.now().astimezone().date().isoformat()
    return database().attendance_summary(today, now_iso())

@app.patch("/api/admin/attendance")
async def admin_update_attendance(req: AttendanceSettingsReq):
    require_admin()
    for value in (req.work_start, req.work_end):
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError as exc:
            raise HTTPException(400, "上下班时间必须使用 HH:MM 格式") from exc
    if req.work_start >= req.work_end:
        raise HTTPException(400, "上班时间必须早于下班时间")
    return database().update_attendance_settings(req.work_start, req.work_end, now_iso())

@app.get("/api/admin/statistics")
async def admin_statistics(period: str = "month", date: str = ""):
    require_admin()
    try:
        current = now_iso()
        start, end = statistics_period_bounds(period, date, now=current)
        return database().admin_annotation_statistics(start, end, current)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
            validate_role(req.role), req.display_name.strip(), now_iso(), req.is_active, req.phone.strip(),
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
        phone=req.phone.strip() if req.phone is not None else None,
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


def publication_rows(req: PublishTasksReq) -> list[dict[str, str]]:
    if req.input_mode == "table":
        return parse_spreadsheet_rows(req.clipboard_text)
    if req.input_mode != "manual":
        raise ValueError("不支持的发布方式")
    row = {field: str((req.task or {}).get(field, "")).strip() for field in TASK_FIELDS}
    if not row["project"] and not row["annotation_content"]:
        raise ValueError("手填任务至少需要项目或标注内容")
    return [row]


@app.post("/api/tasks/preview")
async def preview_tasks(req: PublishTasksReq):
    require_user()
    try:
        rows = publication_rows(req)
        part_specs = None
        if req.part_manifest.strip():
            if len(rows) != 1:
                raise ValueError("使用 Part 工作目录清单时，一次只能发布一行任务")
            part_specs = parse_part_manifest(req.part_manifest, rows[0]["data_path"])
        elif req.part_count < 0 or req.part_count > 10000:
            raise ValueError("Part 数量必须在 0 到 10000 之间")
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
    if not req.part_manifest.strip() and (req.part_count < 0 or req.part_count > 10000):
        raise HTTPException(400, "Part 数量必须在 0 到 10000 之间")
    try:
        rows = publication_rows(req)
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
                "annotation_tool": req.annotation_tool.strip(),
                **row,
                "created_at": now,
                "updated_at": now,
            }
            created.append(database().create_task(
                task, req.part_count, now, part_specs=part_specs, return_details=False,
            ))
    except BaseException as exc:
        _raise_database_error(exc)
    for item in created:
        enqueue_task_reminder("published", item["task_id"], actor, "published:" + item["task_id"])
    return {"tasks": [{key: item[key] for key in
                        ("task_id", "name", "project", "annotation_content", "part_prefix")}
                      for item in created], "count": len(created)}


@app.get("/api/directory-overview")
async def directory_overview(task_id: str = "", status: str = "", query: str = "", page: int = 1, page_size: int = 50, product_tag: str = ""):
    require_user()
    try:
        return database().directory_overview(task_id, status, query, page, page_size, product_tag)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def build_task_overview(now: str):
    tasks = database().task_overview(now)
    today = datetime.fromisoformat(now).date()
    attendance = database().attendance_summary(today.isoformat(), now)
    people = attendance["checked_in_count"]
    available_hours = people * 8
    cumulative = 0.0
    for task in sorted(tasks, key=lambda item: (int(item.get("rank") or 0) or 10**9, item.get("updated_at", ""))):
        try:
            hours = max(0.0, float(task.get("estimated_hours") or 0))
        except (TypeError, ValueError):
            hours = 0.0
        if not hours:
            try:
                amount = float(str(task.get("data_amount") or "").replace(",", ""))
                hourly = float(task.get("hourly_capacity") or 0)
                hours = amount / hourly if hourly > 0 else 0.0
            except (TypeError, ValueError):
                hours = 0.0
        task["estimated_completion_hours"] = hours
        if task["overview_status"] == "completed":
            task["estimated_completion_days"] = None
            task["estimated_completion_date"] = None
            continue
        task["estimated_completion_days"] = max(1, math.ceil((cumulative + hours) / available_hours)) if available_hours and hours else None
        task["estimated_completion_date"] = working_day_completion(today, task["estimated_completion_days"]).isoformat() if task["estimated_completion_days"] else None
        cumulative += hours
    tasks.sort(key=lambda item: (item["estimated_completion_date"] or "9999-12-31", int(item.get("rank") or 0) or 10**9, item["task_id"]))
    return {"tasks": tasks, "attendance": attendance, "available_hours": available_hours}


@app.get("/api/task-overview")
async def task_overview():
    require_user()
    return build_task_overview(now_iso())


@app.get("/api/tasks")
async def list_tasks():
    user = require_user()
    actor = user["username"]
    tasks = database().list_tasks(now_iso())
    for task in tasks:
        task["can_manage_task"] = task["publisher"] == actor or user.get("role") == "admin"
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
    now = now_iso()
    try:
        task = database().get_task(task_id, now)
    except KeyError as exc:
        raise HTTPException(404, "任务不存在") from exc
    publisher = task["publisher"] == actor
    applicant = bool(task.get("applicant")) and task["applicant"] == actor
    manager = bool(task.get("manager")) and task["manager"] == actor and not publisher
    can_review = publisher or manager or user.get("role") == "admin"
    can_view_all = can_review or manager or task.get("applicant") == actor
    task["can_manage_task"] = publisher or user.get("role") == "admin"
    task["is_publisher"] = publisher
    task["is_applicant"] = applicant
    task["is_manager"] = manager
    task["can_review"] = can_review
    task["can_view_all"] = can_view_all
    task["can_manage_ordering"] = user.get("role") == "admin"
    task["available_parts"] = task["part_summary"]["pending"]
    if can_view_all:
        try:
            parts_page = database().list_parts_page(
                task_id, now, status=part_status, query=part_query,
                page=part_page, page_size=part_page_size,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        task["parts"] = parts_page.pop("items")
        task["parts_page"] = parts_page
    if publisher or manager or applicant or user.get("role") == "admin":
        task["statistics"] = database().annotator_statistics(task_id, now)
        task["annotation_statistics"] = database().task_annotation_statistics(task_id, now)
    elif not can_view_all:
        task["parts"] = database().list_parts(task_id, now, actor)
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
        before = database().get_task(task_id, now_iso())
        task = database().update_task_ordering(
            task_id, user["username"], rank=req.rank, priority=req.priority,
            now=now_iso(), is_admin=True,
        )
    except BaseException as exc:
        _raise_database_error(exc)
    if before["priority"] != task["priority"]:
        enqueue_task_reminder("priority", task_id, user["username"], note=f"优先级：{before['priority']} → {task['priority']}")
    return {"task": task}


@app.post("/api/tasks/{task_id}/parts/preview")
async def preview_add_parts(task_id: str, req: AddPartsReq):
    user = require_user()
    try:
        task = database().get_task(task_id, now_iso())
        if task["publisher"] != user["username"] and user["role"] != "admin":
            raise HTTPException(403, "只有发布者或管理员可以追加 Part")
        specs = parse_part_manifest(req.part_manifest, task["data_path"]) if req.part_manifest.strip() else None
        count = len(specs) if specs is not None else req.count
        if not 1 <= count <= 10000:
            raise ValueError("Part 数量必须在 1 到 10000 之间")
        if specs:
            with database().connection() as connection:
                existing = {database()._path_key(row[0]) for row in connection.execute(
                    "SELECT work_path FROM parts WHERE task_id=? AND work_path!=''", (task_id,))}
            for spec in specs:
                if database()._path_key(spec["work_path"]) in existing:
                    raise ValueError(f"重复工作目录：{spec['work_path']}")
        return {"count": count, "part_preview": (specs or [])[:20]}
    except BaseException as exc:
        _raise_database_error(exc)


@app.post("/api/tasks/{task_id}/parts")
async def add_parts(task_id: str, req: AddPartsReq):
    user = require_user()
    actor = user["username"]
    try:
        task = database().get_task(task_id, now_iso())
        if task["publisher"] != actor and user["role"] != "admin":
            raise HTTPException(403, "只有发布者或管理员可以追加 Part")
        specs = parse_part_manifest(req.part_manifest, task["data_path"]) if req.part_manifest.strip() else None
        database().add_parts(task_id, req.count, actor, now_iso(), return_parts=False, part_specs=specs)
    except BaseException as exc:
        _raise_database_error(exc)
    return {"added": len(specs) if specs is not None else req.count, "summary": database().part_summary(task_id)}


@app.delete("/api/tasks/{task_id}/parts/{part_id}")
async def delete_part(task_id: str, part_id: int):
    actor = require_user()["username"]
    try:
        deleted = database().delete_part(task_id, part_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    return {"deleted": deleted, "summary": database().part_summary(task_id)}

@app.patch("/api/tasks/{task_id}/parts/{part_id}")
async def update_part(task_id: str, part_id: int, req: UpdatePartReq):
    actor = require_user()["username"]
    try:
        bbox_changes = {"bbox_count": req.bbox_count} if "bbox_count" in req.model_fields_set else {}
        time_changes = {"work_seconds": req.work_seconds} if "work_seconds" in req.model_fields_set else {}
        part = database().update_part(task_id, part_id, actor, now_iso(), name=req.name, work_path=req.work_path, image_count=req.image_count, progress_count=req.progress_count, **bbox_changes, **time_changes)
    except BaseException as exc:
        _raise_database_error(exc)
    return {"part": part}


@app.post("/api/tasks/{task_id}/parts/claim-next")
async def claim_next_part(task_id: str):
    actor = require_user()["username"]
    try:
        before = database().get_task(task_id, now_iso())
        part = database().claim_next_part(task_id, actor, now_iso())
    except BaseException as exc:
        _raise_database_error(exc)
    if part is None:
        raise HTTPException(409, "当前没有可领取的 Part")
    if before["status"] == "ready":
        enqueue_task_reminder("started", task_id, actor, "started:" + task_id)
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
        before = database().get_task(task_id, now_iso())
        part = database().review_part(
            task_id, part_id, actor, req.action, req.note, now_iso(),
            is_admin=user.get("role") == "admin",
        )
    except BaseException as exc:
        _raise_database_error(exc)
    after = database().get_task(task_id, now_iso())
    if before["status"] != "completed" and after["status"] == "completed":
        enqueue_task_reminder("completed", task_id, actor, "completed:" + task_id)
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
