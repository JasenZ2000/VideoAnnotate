from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


SCHEMA_VERSION = 5
PART_STATUSES = ("pending", "in_progress", "submitted", "rework", "completed")
TASK_FIELDS = (
    "application_date",
    "applicant",
    "project",
    "annotation_content",
    "dataset_source",
    "hourly_capacity",
    "data_amount",
    "estimated_hours",
    "data_path",
    "guide_path",
)


def _elapsed_seconds(start: Optional[str], end: Optional[str]) -> float:
    if not start or not end:
        return 0.0
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return 0.0


class PlatformDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    display_name TEXT NOT NULL DEFAULT '',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                );
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ready',
                    deleted INTEGER NOT NULL DEFAULT 0,
                    publisher TEXT NOT NULL,
                    manager TEXT NOT NULL DEFAULT '',
                    product_tag TEXT NOT NULL DEFAULT '',
                    part_prefix TEXT NOT NULL DEFAULT '',
                    application_date TEXT NOT NULL DEFAULT '',
                    applicant TEXT NOT NULL DEFAULT '',
                    project TEXT NOT NULL DEFAULT '',
                    annotation_content TEXT NOT NULL DEFAULT '',
                    dataset_source TEXT NOT NULL DEFAULT '',
                    hourly_capacity TEXT NOT NULL DEFAULT '',
                    data_amount TEXT NOT NULL DEFAULT '',
                    estimated_hours TEXT NOT NULL DEFAULT '',
                    data_path TEXT NOT NULL DEFAULT '',
                    guide_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parts (
                    part_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    part_index INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    work_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    annotator TEXT,
                    claimed_at TEXT,
                    submitted_at TEXT,
                    reviewed_at TEXT,
                    work_seconds REAL NOT NULL DEFAULT 0,
                    submission_note TEXT NOT NULL DEFAULT '',
                    review_note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(task_id, part_index)
                );
                CREATE INDEX IF NOT EXISTS idx_parts_task_status
                    ON parts(task_id, status, part_index);
                CREATE TABLE IF NOT EXISTS part_work_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_id INTEGER NOT NULL REFERENCES parts(part_id) ON DELETE CASCADE,
                    annotator TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_open_part_session
                    ON part_work_sessions(part_id) WHERE ended_at IS NULL;
                CREATE TABLE IF NOT EXISTS part_comments (
                    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_id INTEGER NOT NULL REFERENCES parts(part_id) ON DELETE CASCADE,
                    actor TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'note',
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_part_comments_part
                    ON part_comments(part_id, comment_id);
                """
            )
            task_columns = {row["name"] for row in connection.execute("PRAGMA table_info(tasks)")}
            additions = {
                "publisher": "TEXT NOT NULL DEFAULT ''",
                "manager": "TEXT NOT NULL DEFAULT ''",
                "product_tag": "TEXT NOT NULL DEFAULT ''",
                "part_prefix": "TEXT NOT NULL DEFAULT ''",
                "application_date": "TEXT NOT NULL DEFAULT ''",
                "applicant": "TEXT NOT NULL DEFAULT ''",
                "project": "TEXT NOT NULL DEFAULT ''",
                "annotation_content": "TEXT NOT NULL DEFAULT ''",
                "dataset_source": "TEXT NOT NULL DEFAULT ''",
                "hourly_capacity": "TEXT NOT NULL DEFAULT ''",
                "data_amount": "TEXT NOT NULL DEFAULT ''",
                "estimated_hours": "TEXT NOT NULL DEFAULT ''",
                "data_path": "TEXT NOT NULL DEFAULT ''",
                "guide_path": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in additions.items():
                if column not in task_columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")
            part_columns = {row["name"] for row in connection.execute("PRAGMA table_info(parts)")}
            if "work_path" not in part_columns:
                connection.execute("ALTER TABLE parts ADD COLUMN work_path TEXT NOT NULL DEFAULT ''")
            task_columns.update(additions)
            legacy_columns = {row["name"] for row in connection.execute("PRAGMA table_info(tasks)")}
            if "assignee" in legacy_columns:
                connection.execute(
                    "UPDATE tasks SET publisher=assignee WHERE publisher='' AND assignee!=''"
                )
            connection.execute(
                "UPDATE tasks SET manager=publisher WHERE manager='' AND publisher!=''"
            )
            connection.execute(
                "INSERT INTO schema_info(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def health(self) -> dict[str, Any]:
        with self.connection() as connection:
            check = connection.execute("PRAGMA quick_check").fetchone()[0]
            version = connection.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
        return {"quick_check": check, "schema_version": int(version[0]) if version else 0}

    # Users and sessions
    def user_count(self) -> int:
        with self.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def active_admin_count(self) -> int:
        with self.connection() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1"
            ).fetchone()[0])

    def create_user(self, username: str, password_hash: str, role: str, display_name: str,
                    now: str, is_active: bool = True) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO users(username,password_hash,role,display_name,is_active,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (username, password_hash, role, display_name, int(is_active), now, now),
            )
        return self.get_user(username)

    def get_user(self, username: str, include_password: bool = False) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            raise KeyError(username)
        result = dict(row)
        result["is_active"] = bool(result["is_active"])
        if not include_password:
            result.pop("password_hash", None)
        return result

    def list_users(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT username,role,display_name,is_active,created_at,updated_at,last_login_at "
                "FROM users ORDER BY role DESC, username"
            ).fetchall()
        return [{**dict(row), "is_active": bool(row["is_active"])} for row in rows]

    def update_user(self, username: str, *, role: Optional[str] = None,
                    display_name: Optional[str] = None, password_hash: Optional[str] = None,
                    is_active: Optional[bool] = None, now: str) -> dict[str, Any]:
        assignments = ["updated_at=?"]
        values: list[Any] = [now]
        for column, value in (("role", role), ("display_name", display_name),
                              ("password_hash", password_hash)):
            if value is not None:
                assignments.append(f"{column}=?")
                values.append(value)
        if is_active is not None:
            assignments.append("is_active=?")
            values.append(int(is_active))
        values.append(username)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE users SET {', '.join(assignments)} WHERE username=?", values
            )
            if cursor.rowcount != 1:
                raise KeyError(username)
            if is_active is False or password_hash is not None:
                connection.execute("DELETE FROM user_sessions WHERE username=?", (username,))
        return self.get_user(username)

    def touch_user_login(self, username: str, now: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE users SET last_login_at=?,updated_at=? WHERE username=?",
                (now, now, username),
            )

    def create_session(self, session_id: str, username: str, token_hash: str,
                       expires_at: str, now: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO user_sessions(session_id,username,token_hash,expires_at,created_at,last_seen_at) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, username, token_hash, expires_at, now, now),
            )

    def get_session_user(self, token_hash: str, now: str) -> Optional[dict[str, Any]]:
        with self.transaction() as connection:
            connection.execute("DELETE FROM user_sessions WHERE expires_at<=?", (now,))
            row = connection.execute(
                "SELECT u.*,s.session_id FROM user_sessions s JOIN users u ON u.username=s.username "
                "WHERE s.token_hash=? AND u.is_active=1",
                (token_hash,),
            ).fetchone()
            if row:
                connection.execute(
                    "UPDATE user_sessions SET last_seen_at=? WHERE session_id=?",
                    (now, row["session_id"]),
                )
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM user_sessions WHERE token_hash=?", (token_hash,))

    # Tasks
    def create_task(self, task: dict[str, Any], part_count: int, now: str,
                    part_specs: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
        effective_count = len(part_specs) if part_specs is not None else part_count
        if effective_count < 1 or effective_count > 10000:
            raise ValueError("part count must be between 1 and 10000")
        columns = ["task_id", "name", "status", "publisher", "manager", "product_tag",
                   "part_prefix", *TASK_FIELDS, "created_at", "updated_at"]
        values = [task.get(column, "") for column in columns]
        with self.transaction() as connection:
            connection.execute(
                f"INSERT INTO tasks({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                values,
            )
            self._insert_parts(
                connection, task["task_id"], effective_count,
                str(task.get("part_prefix", "")), now, part_specs=part_specs,
            )
        return self.get_task(str(task["task_id"]), now)

    def update_task(self, task_id: str, actor: str, changes: dict[str, str],
                    now: str) -> dict[str, Any]:
        allowed = {"product_tag", "part_prefix", *TASK_FIELDS}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported task fields: {', '.join(sorted(unknown))}")
        with self.transaction() as connection:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id=? AND deleted=0", (task_id,)
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if task["publisher"] != actor:
                raise PermissionError("only publisher can edit task")
            values = {key: str(value).strip() for key, value in changes.items()}
            project = values.get("project", task["project"])
            content = values.get("annotation_content", task["annotation_content"])
            values["name"] = f"{project or '未命名项目'} · {content or '标注任务'}"
            assignments = [f"{column}=?" for column in values]
            connection.execute(
                f"UPDATE tasks SET {','.join(assignments)},updated_at=? WHERE task_id=?",
                [*values.values(), now, task_id],
            )
        return self.get_task(task_id, now)

    def delete_task(self, task_id: str, actor: str, now: str) -> None:
        with self.transaction() as connection:
            task = connection.execute(
                "SELECT publisher FROM tasks WHERE task_id=? AND deleted=0", (task_id,)
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if task["publisher"] != actor:
                raise PermissionError("only publisher can delete task")
            connection.execute(
                "UPDATE tasks SET deleted=1,updated_at=? WHERE task_id=?", (now, task_id)
            )

    def _insert_parts(self, connection: sqlite3.Connection, task_id: str, count: int,
                      prefix: str, now: str,
                      part_specs: Optional[list[dict[str, str]]] = None) -> None:
        row = connection.execute(
            "SELECT COALESCE(MAX(part_index),0) FROM parts WHERE task_id=?", (task_id,)
        ).fetchone()
        start = int(row[0]) + 1
        clean = prefix.strip()
        separator = "" if not clean or clean.endswith(("_", "-", " ")) else "_"
        for offset, index in enumerate(range(start, start + count)):
            spec = part_specs[offset] if part_specs is not None else None
            name = str(spec.get("name", "")).strip() if spec else ""
            if not name:
                name = f"{clean}{separator}part_{index:03d}"
            work_path = str(spec.get("work_path", "")).strip() if spec else ""
            connection.execute(
                "INSERT INTO parts(task_id,part_index,name,work_path,status,created_at,updated_at) "
                "VALUES(?,?,?,?,'pending',?,?)",
                (task_id, index, name, work_path, now, now),
            )

    def add_parts(self, task_id: str, count: int, actor: str, now: str) -> list[dict[str, Any]]:
        if count < 1 or count > 10000:
            raise ValueError("part count must be between 1 and 10000")
        with self.transaction() as connection:
            task = connection.execute(
                "SELECT publisher,part_prefix FROM tasks WHERE task_id=? AND deleted=0", (task_id,)
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if task["publisher"] != actor:
                raise PermissionError("only publisher can add parts")
            self._insert_parts(connection, task_id, count, task["part_prefix"], now)
            connection.execute("UPDATE tasks SET status='in_progress',updated_at=? WHERE task_id=?",
                               (now, task_id))
        return self.list_parts(task_id, now)

    def _task_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = {key: row[key] for key in (
            "task_id", "name", "status", "publisher", "product_tag", "part_prefix",
            *TASK_FIELDS, "created_at", "updated_at"
        )}
        return result

    def list_tasks(self, now: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE deleted=0 ORDER BY updated_at DESC,task_id DESC"
            ).fetchall()
        tasks = []
        for row in rows:
            task = self._task_dict(row)
            task["part_summary"] = self.part_summary(task["task_id"])
            tasks.append(task)
        return tasks

    def get_task(self, task_id: str, now: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id=? AND deleted=0", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        task = self._task_dict(row)
        task["part_summary"] = self.part_summary(task_id)
        return task

    def part_summary(self, task_id: str) -> dict[str, int]:
        summary = {status: 0 for status in PART_STATUSES}
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT status,COUNT(*) count FROM parts WHERE task_id=? GROUP BY status", (task_id,)
            ).fetchall()
        for row in rows:
            summary[row["status"]] = int(row["count"])
        summary["total"] = sum(summary.values())
        return summary

    # Parts and timing
    def _comments(self, connection: sqlite3.Connection, part_id: int) -> list[dict[str, Any]]:
        return [dict(row) for row in connection.execute(
            "SELECT comment_id,actor,kind,content,created_at FROM part_comments "
            "WHERE part_id=? ORDER BY comment_id", (part_id,)
        )]

    def list_parts(self, task_id: str, now: str, actor: Optional[str] = None) -> list[dict[str, Any]]:
        where = "p.task_id=?"
        params: list[Any] = [task_id]
        if actor is not None:
            where += " AND p.annotator=?"
            params.append(actor)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT p.*,(SELECT started_at FROM part_work_sessions s WHERE s.part_id=p.part_id "
                f"AND s.ended_at IS NULL ORDER BY session_id DESC LIMIT 1) active_started_at "
                f"FROM parts p WHERE {where} ORDER BY p.part_index", params
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["annotator"] = item.get("annotator") or ""
                item["work_seconds"] = round(
                    float(item["work_seconds"]) + _elapsed_seconds(item["active_started_at"], now), 3
                )
                item["comments"] = self._comments(connection, int(item["part_id"]))
                result.append(item)
        return result

    def _open_session(self, connection: sqlite3.Connection, part_id: int, actor: str, now: str) -> None:
        connection.execute(
            "INSERT INTO part_work_sessions(part_id,annotator,started_at,created_at) VALUES(?,?,?,?)",
            (part_id, actor, now, now),
        )

    def _close_session(self, connection: sqlite3.Connection, part_id: int, now: str) -> None:
        session = connection.execute(
            "SELECT session_id,started_at FROM part_work_sessions WHERE part_id=? AND ended_at IS NULL",
            (part_id,),
        ).fetchone()
        if not session:
            return
        duration = _elapsed_seconds(session["started_at"], now)
        connection.execute(
            "UPDATE part_work_sessions SET ended_at=?,duration_seconds=? WHERE session_id=?",
            (now, duration, session["session_id"]),
        )
        connection.execute("UPDATE parts SET work_seconds=work_seconds+? WHERE part_id=?",
                           (duration, part_id))

    def claim_next_part(self, task_id: str, actor: str, now: str) -> Optional[dict[str, Any]]:
        with self.transaction() as connection:
            task = connection.execute(
                "SELECT publisher FROM tasks WHERE task_id=? AND deleted=0", (task_id,)
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            active = connection.execute(
                "SELECT 1 FROM parts WHERE task_id=? AND annotator=? AND status='in_progress'",
                (task_id, actor),
            ).fetchone()
            if active:
                raise ValueError("finish the active part before claiming another")
            part = connection.execute(
                "SELECT part_id FROM parts WHERE task_id=? AND status='pending' ORDER BY part_index LIMIT 1",
                (task_id,),
            ).fetchone()
            if part is None:
                return None
            connection.execute(
                "UPDATE parts SET status='in_progress',annotator=?,claimed_at=?,updated_at=? "
                "WHERE part_id=? AND status='pending'",
                (actor, now, now, part["part_id"]),
            )
            self._open_session(connection, int(part["part_id"]), actor, now)
            connection.execute("UPDATE tasks SET status='in_progress',updated_at=? WHERE task_id=?",
                               (now, task_id))
            part_id = int(part["part_id"])
        return next(item for item in self.list_parts(task_id, now, actor) if item["part_id"] == part_id)

    def start_rework(self, task_id: str, part_id: int, actor: str, now: str) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status,annotator FROM parts WHERE task_id=? AND part_id=?", (task_id, part_id)
            ).fetchone()
            if row is None:
                raise KeyError(part_id)
            if row["annotator"] != actor or row["status"] != "rework":
                raise PermissionError("only assigned annotator can start rework")
            connection.execute("UPDATE parts SET status='in_progress',updated_at=? WHERE part_id=?",
                               (now, part_id))
            self._open_session(connection, part_id, actor, now)
        return next(item for item in self.list_parts(task_id, now, actor) if item["part_id"] == part_id)

    def submit_part(self, task_id: str, part_id: int, actor: str, note: str, now: str) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status,annotator FROM parts WHERE task_id=? AND part_id=?", (task_id, part_id)
            ).fetchone()
            if row is None:
                raise KeyError(part_id)
            if row["annotator"] != actor or row["status"] != "in_progress":
                raise PermissionError("only active annotator can submit")
            self._close_session(connection, part_id, now)
            connection.execute(
                "UPDATE parts SET status='submitted',submitted_at=?,submission_note=?,updated_at=? "
                "WHERE part_id=?", (now, note.strip(), now, part_id)
            )
            if note.strip():
                connection.execute(
                    "INSERT INTO part_comments(part_id,actor,kind,content,created_at) VALUES(?,?,?,?,?)",
                    (part_id, actor, "submission", note.strip(), now),
                )
        return next(item for item in self.list_parts(task_id, now, actor) if item["part_id"] == part_id)

    def add_part_comment(self, task_id: str, part_id: int, actor: str, content: str,
                         now: str) -> dict[str, Any]:
        text = content.strip()
        if not text:
            raise ValueError("comment cannot be empty")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT p.annotator,t.publisher FROM parts p JOIN tasks t ON t.task_id=p.task_id "
                "WHERE p.task_id=? AND p.part_id=?", (task_id, part_id)
            ).fetchone()
            if row is None:
                raise KeyError(part_id)
            if actor not in {row["annotator"], row["publisher"]}:
                raise PermissionError("only assigned annotator or publisher can comment")
            connection.execute(
                "INSERT INTO part_comments(part_id,actor,kind,content,created_at) VALUES(?,?,?,?,?)",
                (part_id, actor, "note", text, now),
            )
        parts = self.list_parts(task_id, now, None if actor == row["publisher"] else actor)
        return next(item for item in parts if item["part_id"] == part_id)

    def review_part(self, task_id: str, part_id: int, actor: str, action: str,
                    note: str, now: str, *, is_admin: bool = False) -> dict[str, Any]:
        if action not in {"approve", "rework"}:
            raise ValueError("action must be approve or rework")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT p.status,t.publisher FROM parts p JOIN tasks t ON t.task_id=p.task_id "
                "WHERE p.task_id=? AND p.part_id=?", (task_id, part_id)
            ).fetchone()
            if row is None:
                raise KeyError(part_id)
            if row["publisher"] != actor and not is_admin:
                raise PermissionError("only publisher or admin can review")
            if row["status"] != "submitted":
                raise ValueError("only submitted part can be reviewed")
            status = "completed" if action == "approve" else "rework"
            connection.execute(
                "UPDATE parts SET status=?,reviewed_at=?,review_note=?,updated_at=? WHERE part_id=?",
                (status, now, note.strip(), now, part_id),
            )
            connection.execute(
                "INSERT INTO part_comments(part_id,actor,kind,content,created_at) VALUES(?,?,?,?,?)",
                (part_id, actor, f"review_{action}", note.strip() or ("同意" if action == "approve" else "退回修改"), now),
            )
            remaining = int(connection.execute(
                "SELECT COUNT(*) FROM parts WHERE task_id=? AND status!='completed'", (task_id,)
            ).fetchone()[0])
            connection.execute("UPDATE tasks SET status=?,updated_at=? WHERE task_id=?",
                               ("completed" if remaining == 0 else "in_progress", now, task_id))
        return next(item for item in self.list_parts(task_id, now) if item["part_id"] == part_id)

    def annotator_statistics(self, task_id: str, now: str) -> list[dict[str, Any]]:
        parts = self.list_parts(task_id, now)
        grouped: dict[str, dict[str, Any]] = {}
        for part in parts:
            actor = part["annotator"]
            if not actor:
                continue
            item = grouped.setdefault(actor, {
                "username": actor, "total": 0, "completed": 0, "submitted": 0,
                "rework": 0, "in_progress": 0, "work_seconds": 0.0,
            })
            item["total"] += 1
            if part["status"] in item:
                item[part["status"]] += 1
            item["work_seconds"] += float(part["work_seconds"])
        with self.connection() as connection:
            names = {row["username"]: row["display_name"] for row in connection.execute(
                "SELECT username,display_name FROM users"
            )}
        for item in grouped.values():
            item["display_name"] = names.get(item["username"], "")
            item["work_seconds"] = round(item["work_seconds"], 3)
        return sorted(grouped.values(), key=lambda item: (-item["completed"], item["username"]))
