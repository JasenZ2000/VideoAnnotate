from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


SCHEMA_VERSION = 2
DEFAULT_STAGE_NAMES = (
    "video",
    "prelabel",
    "locateanything",
    "tracking",
    "package",
    "review",
    "export",
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _without(source: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in keys}


def _elapsed_seconds(start: Optional[str], end: Optional[str]) -> float:
    if not start or not end:
        return 0.0
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except ValueError:
        return 0.0


class PlatformDatabase:
    """SQLite metadata store; large videos and generated artifacts stay on disk."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self._write_lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
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
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
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

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    assignee TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    prelabel_source TEXT NOT NULL DEFAULT 'none',
                    prompt TEXT NOT NULL DEFAULT 'person',
                    status TEXT NOT NULL DEFAULT 'created',
                    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
                    current_video_id TEXT,
                    task_type TEXT NOT NULL DEFAULT 'general',
                    publisher TEXT NOT NULL DEFAULT '',
                    manager TEXT NOT NULL DEFAULT '',
                    instructions TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS task_classes (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    class_id INTEGER NOT NULL CHECK (class_id >= 0),
                    name TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (task_id, class_id),
                    UNIQUE (task_id, name)
                );

                CREATE TABLE IF NOT EXISTS task_stages (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    message TEXT NOT NULL DEFAULT '',
                    updated_at TEXT,
                    PRIMARY KEY (task_id, stage)
                );

                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT NOT NULL,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    frame_count INTEGER,
                    fps REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    input_label_dir TEXT,
                    locany_label_dir TEXT,
                    split_status TEXT,
                    split_segment_length INTEGER,
                    split_label_source TEXT,
                    split_segment_count INTEGER,
                    split_message TEXT,
                    split_updated_at TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (task_id, video_id)
                );

                CREATE INDEX IF NOT EXISTS idx_videos_task
                    ON videos(task_id, created_at, video_id);

                CREATE TABLE IF NOT EXISTS segments (
                    task_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    video_path TEXT NOT NULL,
                    input_label_dir TEXT,
                    locany_label_dir TEXT,
                    labels_copied INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ready',
                    locate_status TEXT,
                    locate_message TEXT,
                    locate_label_dir TEXT,
                    locate_prompt TEXT,
                    locate_updated_at TEXT,
                    tracking_status TEXT,
                    tracking_message TEXT,
                    tracking_results TEXT,
                    tracking_updated_at TEXT,
                    created_at TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (task_id, video_id, segment_id),
                    FOREIGN KEY (task_id, video_id)
                        REFERENCES videos(task_id, video_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    time TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'info',
                    message TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_task_time
                    ON events(task_id, event_id DESC);

                CREATE TABLE IF NOT EXISTS task_annotators (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    username TEXT NOT NULL,
                    joined_at TEXT,
                    PRIMARY KEY (task_id, username)
                );

                CREATE TABLE IF NOT EXISTS parts (
                    part_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    part_index INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
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
                    UNIQUE (task_id, part_index)
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

                CREATE TABLE IF NOT EXISTS attachments (
                    attachment_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    part_id INTEGER REFERENCES parts(part_id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    media_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL,
                    uploaded_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_attachments_task
                    ON attachments(task_id, part_id, created_at);

                CREATE TABLE IF NOT EXISTS issues (
                    issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    part_id INTEGER REFERENCES parts(part_id) ON DELETE CASCADE,
                    reported_by TEXT NOT NULL,
                    assigned_to TEXT NOT NULL DEFAULT '',
                    severity TEXT NOT NULL DEFAULT 'normal',
                    status TEXT NOT NULL DEFAULT 'open',
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    resolution TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_issues_task_status
                    ON issues(task_id, status, part_id);
                """
            )
            existing_task_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(tasks)")
            }
            for column, definition in (
                ("task_type", "TEXT NOT NULL DEFAULT 'general'"),
                ("publisher", "TEXT NOT NULL DEFAULT ''"),
                ("manager", "TEXT NOT NULL DEFAULT ''"),
                ("instructions", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in existing_task_columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")
            connection.execute(
                "UPDATE tasks SET manager = assignee WHERE manager = '' AND assignee != ''"
            )
            connection.execute(
                "UPDATE tasks SET publisher = manager WHERE publisher = '' AND manager != ''"
            )
            connection.execute(
                "UPDATE tasks SET task_type = 'video_detection' "
                "WHERE task_type = 'general' AND task_id IN (SELECT DISTINCT task_id FROM videos)"
            )
            connection.execute(
                "INSERT INTO schema_info(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def task_exists(self, task_id: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return row is not None

    def save_task(self, task: dict[str, Any]) -> None:
        task_id = str(task["task_id"])
        known_task_keys = {
            "task_id", "name", "assignee", "notes", "prelabel_source", "prompt",
            "status", "deleted", "current_video_id", "created_at", "updated_at",
            "deleted_at", "task_type", "publisher", "manager", "instructions",
            "annotators", "classes", "classes_text", "stages", "videos", "events",
            "parts", "part_summary", "attachments", "issues",
        }
        extra = _without(task, known_task_keys)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, name, assignee, notes, prelabel_source, prompt, status,
                    deleted, current_video_id, created_at, updated_at, deleted_at, extra_json,
                    task_type, publisher, manager, instructions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    name=excluded.name,
                    assignee=excluded.assignee,
                    notes=excluded.notes,
                    prelabel_source=excluded.prelabel_source,
                    prompt=excluded.prompt,
                    status=excluded.status,
                    deleted=excluded.deleted,
                    current_video_id=excluded.current_video_id,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at,
                    extra_json=excluded.extra_json,
                    task_type=excluded.task_type,
                    publisher=excluded.publisher,
                    manager=excluded.manager,
                    instructions=excluded.instructions
                """,
                (
                    task_id,
                    str(task.get("name", task_id)),
                    str(task.get("assignee", "")),
                    str(task.get("notes", "")),
                    str(task.get("prelabel_source", "none")),
                    str(task.get("prompt", "person")),
                    str(task.get("status", "created")),
                    1 if task.get("deleted") else 0,
                    task.get("current_video_id"),
                    str(task.get("created_at", "")),
                    str(task.get("updated_at", "")),
                    task.get("deleted_at"),
                    _json_dumps(extra),
                    str(task.get("task_type", "general")),
                    str(task.get("publisher", "")),
                    str(task.get("manager", task.get("assignee", ""))),
                    str(task.get("instructions", "")),
                ),
            )

            connection.execute("DELETE FROM task_annotators WHERE task_id = ?", (task_id,))
            for username in task.get("annotators", []):
                connection.execute(
                    "INSERT INTO task_annotators(task_id, username, joined_at) VALUES (?, ?, ?)",
                    (task_id, str(username), task.get("created_at")),
                )

            connection.execute("DELETE FROM task_classes WHERE task_id = ?", (task_id,))
            for order, item in enumerate(task.get("classes", [])):
                connection.execute(
                    "INSERT INTO task_classes(task_id, class_id, name, sort_order) VALUES (?, ?, ?, ?)",
                    (task_id, int(item["id"]), str(item["name"]), order),
                )

            connection.execute("DELETE FROM task_stages WHERE task_id = ?", (task_id,))
            stages = task.get("stages", {})
            for stage in dict.fromkeys((*DEFAULT_STAGE_NAMES, *stages.keys())):
                value = stages.get(stage, {})
                connection.execute(
                    "INSERT INTO task_stages(task_id, stage, status, message, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        task_id,
                        stage,
                        str(value.get("status", "pending")),
                        str(value.get("message", "")),
                        value.get("updated_at"),
                    ),
                )

            connection.execute("DELETE FROM videos WHERE task_id = ?", (task_id,))
            for video in task.get("videos", []):
                self._insert_video(connection, task_id, video)

    def _insert_video(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        video: dict[str, Any],
    ) -> None:
        metadata = dict(video.get("metadata") or {})
        split = dict(video.get("split") or {})
        known_video_keys = {
            "video_id", "name", "path", "metadata", "status", "input_label_dir",
            "locany_label_dir", "split", "segments", "created_at", "updated_at",
        }
        connection.execute(
            """
            INSERT INTO videos(
                video_id, task_id, name, path, width, height, frame_count, fps,
                metadata_json, status, input_label_dir, locany_label_dir,
                split_status, split_segment_length, split_label_source,
                split_segment_count, split_message, split_updated_at,
                created_at, updated_at, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(video["video_id"]), task_id, str(video.get("name", video["video_id"])),
                str(video.get("path", "")), metadata.get("width"), metadata.get("height"),
                metadata.get("frame_count"), metadata.get("fps"), _json_dumps(metadata),
                str(video.get("status", "uploaded")), video.get("input_label_dir"),
                video.get("locany_label_dir"), split.get("status"), split.get("segment_length"),
                split.get("label_source"), split.get("segments"), split.get("message"),
                split.get("updated_at"), video.get("created_at"), video.get("updated_at"),
                _json_dumps(_without(video, known_video_keys)),
            ),
        )
        for segment in video.get("segments", []):
            self._insert_segment(connection, task_id, str(video["video_id"]), segment)

    def _insert_segment(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        video_id: str,
        segment: dict[str, Any],
    ) -> None:
        locate = dict(segment.get("locateanything") or {})
        tracking = dict(segment.get("tracking") or {})
        known_segment_keys = {
            "segment_id", "start_frame", "end_frame", "frame_count", "video_path",
            "input_label_dir", "locany_label_dir", "labels_copied", "status", "created_at",
            "locateanything", "tracking",
        }
        connection.execute(
            """
            INSERT INTO segments(
                task_id, video_id, segment_id, start_frame, end_frame, frame_count, video_path,
                input_label_dir, locany_label_dir, labels_copied, status,
                locate_status, locate_message, locate_label_dir, locate_prompt, locate_updated_at,
                tracking_status, tracking_message, tracking_results, tracking_updated_at,
                created_at, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id, video_id, str(segment["segment_id"]), int(segment.get("start_frame", 0)),
                int(segment.get("end_frame", -1)), int(segment.get("frame_count", 0)),
                str(segment.get("video_path", "")), segment.get("input_label_dir"),
                segment.get("locany_label_dir"), int(segment.get("labels_copied", 0)),
                str(segment.get("status", "ready")), locate.get("status"),
                locate.get("message"), locate.get("label_dir"), locate.get("prompt"),
                locate.get("updated_at"), tracking.get("status"), tracking.get("message"),
                tracking.get("results"), tracking.get("updated_at"), segment.get("created_at"),
                _json_dumps(_without(segment, known_segment_keys)),
            ),
        )

    def load_task(self, task_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = _json_loads(row["extra_json"], {})
            task.update({
                "task_id": row["task_id"],
                "name": row["name"],
                "assignee": row["assignee"],
                "notes": row["notes"],
                "prelabel_source": row["prelabel_source"],
                "prompt": row["prompt"],
                "status": row["status"],
                "deleted": bool(row["deleted"]),
                "task_type": row["task_type"],
                "publisher": row["publisher"],
                "manager": row["manager"],
                "instructions": row["instructions"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
            if row["current_video_id"]:
                task["current_video_id"] = row["current_video_id"]
            if row["deleted_at"]:
                task["deleted_at"] = row["deleted_at"]

            task["classes"] = [
                {"id": item["class_id"], "name": item["name"]}
                for item in connection.execute(
                    "SELECT class_id, name FROM task_classes WHERE task_id = ? "
                    "ORDER BY sort_order, class_id",
                    (task_id,),
                )
            ]
            task["annotators"] = [
                item["username"]
                for item in connection.execute(
                    "SELECT username FROM task_annotators WHERE task_id = ? ORDER BY rowid",
                    (task_id,),
                )
            ]
            task["stages"] = {
                item["stage"]: {
                    "status": item["status"],
                    "message": item["message"],
                    **({"updated_at": item["updated_at"]} if item["updated_at"] else {}),
                }
                for item in connection.execute(
                    "SELECT stage, status, message, updated_at FROM task_stages "
                    "WHERE task_id = ? ORDER BY rowid",
                    (task_id,),
                )
            }
            task["videos"] = [
                self._load_video(connection, video_row)
                for video_row in connection.execute(
                    "SELECT * FROM videos WHERE task_id = ? ORDER BY created_at, video_id",
                    (task_id,),
                )
            ]
            return task

    def _load_video(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        video = _json_loads(row["extra_json"], {})
        metadata = _json_loads(row["metadata_json"], {})
        for key in ("width", "height", "frame_count", "fps"):
            if row[key] is not None:
                metadata[key] = row[key]
        video.update({
            "video_id": row["video_id"],
            "name": row["name"],
            "path": row["path"],
            "metadata": metadata,
            "status": row["status"],
            "segments": [
                self._load_segment(item)
                for item in connection.execute(
                    "SELECT * FROM segments WHERE task_id = ? AND video_id = ? "
                    "ORDER BY start_frame, segment_id",
                    (row["task_id"], row["video_id"]),
                )
            ],
        })
        for key in ("input_label_dir", "locany_label_dir", "created_at", "updated_at"):
            if row[key] is not None:
                video[key] = row[key]
        if row["split_status"] is not None:
            split = {"status": row["split_status"]}
            split_columns = {
                "segment_length": "split_segment_length",
                "label_source": "split_label_source",
                "segments": "split_segment_count",
                "message": "split_message",
                "updated_at": "split_updated_at",
            }
            for key, column in split_columns.items():
                if row[column] is not None:
                    split[key] = row[column]
            video["split"] = split
        return video

    def _load_segment(self, row: sqlite3.Row) -> dict[str, Any]:
        segment = _json_loads(row["extra_json"], {})
        segment.update({
            "segment_id": row["segment_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "frame_count": row["frame_count"],
            "video_path": row["video_path"],
            "input_label_dir": row["input_label_dir"] or "",
            "labels_copied": row["labels_copied"],
            "status": row["status"],
        })
        if row["locany_label_dir"]:
            segment["locany_label_dir"] = row["locany_label_dir"]
        if row["created_at"]:
            segment["created_at"] = row["created_at"]
        if row["locate_status"]:
            locate = {"status": row["locate_status"]}
            for key, column in (
                ("message", "locate_message"), ("label_dir", "locate_label_dir"),
                ("prompt", "locate_prompt"), ("updated_at", "locate_updated_at"),
            ):
                if row[column] is not None:
                    locate[key] = row[column]
            segment["locateanything"] = locate
        if row["tracking_status"]:
            tracking = {"status": row["tracking_status"]}
            for key, column in (
                ("message", "tracking_message"), ("results", "tracking_results"),
                ("updated_at", "tracking_updated_at"),
            ):
                if row[column] is not None:
                    tracking[key] = row[column]
            segment["tracking"] = tracking
        return segment

    def list_tasks(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        where = "" if include_deleted else "WHERE deleted = 0"
        with self.connection() as connection:
            task_ids = [
                row["task_id"]
                for row in connection.execute(
                    f"SELECT task_id FROM tasks {where} ORDER BY updated_at DESC, task_id DESC"
                )
            ]
        return [self.load_task(task_id) for task_id in task_ids]

    def task_people(self, task_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            task = connection.execute(
                "SELECT publisher, manager FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            annotators = [
                row["username"]
                for row in connection.execute(
                    "SELECT username FROM task_annotators WHERE task_id = ? ORDER BY rowid",
                    (task_id,),
                )
            ]
        return {"publisher": task["publisher"], "manager": task["manager"], "annotators": annotators}

    def is_participant(self, task_id: str, username: str) -> bool:
        if not username:
            return False
        people = self.task_people(task_id)
        return username in {people["publisher"], people["manager"], *people["annotators"]}

    def is_manager(self, task_id: str, username: str) -> bool:
        if not username:
            return False
        people = self.task_people(task_id)
        return username in {people["publisher"], people["manager"]}

    def create_parts(
        self,
        task_id: str,
        count: int,
        name_prefix: str,
        instructions: str,
        now: str,
    ) -> list[dict[str, Any]]:
        if count < 1 or count > 10000:
            raise ValueError("part count must be between 1 and 10000")
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone() is None:
                raise KeyError(task_id)
            row = connection.execute(
                "SELECT COALESCE(MAX(part_index), 0) AS max_index FROM parts WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            start = int(row["max_index"]) + 1
            width = max(3, len(str(start + count - 1)))
            prefix = name_prefix.strip() or "Part"
            for offset in range(count):
                index = start + offset
                connection.execute(
                    """
                    INSERT INTO parts(
                        task_id, part_index, name, instructions, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (task_id, index, f"{prefix} {index:0{width}d}", instructions, now, now),
                )
            connection.execute(
                "UPDATE tasks SET status=CASE WHEN status='created' THEN 'ready' ELSE status END, "
                "updated_at=? WHERE task_id=?",
                (now, task_id),
            )
        return self.list_parts(task_id)

    def list_parts(self, task_id: str, now: Optional[str] = None) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM issues i
                        WHERE i.part_id = p.part_id AND i.status = 'open') AS open_issue_count,
                       (SELECT started_at FROM part_work_sessions s
                        WHERE s.part_id = p.part_id AND s.ended_at IS NULL
                        ORDER BY session_id DESC LIMIT 1) AS active_started_at
                FROM parts p
                WHERE p.task_id = ?
                ORDER BY p.part_index
                """,
                (task_id,),
            ).fetchall()
        return [self._part_dict(row, now) for row in rows]

    def _part_dict(self, row: sqlite3.Row, now: Optional[str] = None) -> dict[str, Any]:
        current_seconds = _elapsed_seconds(row["active_started_at"], now) if now else 0.0
        return {
            "part_id": row["part_id"],
            "task_id": row["task_id"],
            "part_index": row["part_index"],
            "name": row["name"],
            "instructions": row["instructions"],
            "status": row["status"],
            "annotator": row["annotator"] or "",
            "claimed_at": row["claimed_at"],
            "submitted_at": row["submitted_at"],
            "reviewed_at": row["reviewed_at"],
            "work_seconds": round(float(row["work_seconds"]) + current_seconds, 3),
            "submission_note": row["submission_note"],
            "review_note": row["review_note"],
            "open_issue_count": int(row["open_issue_count"]),
            "active_started_at": row["active_started_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def part_summary(self, task_id: str) -> dict[str, int]:
        summary = {status: 0 for status in ("pending", "in_progress", "submitted", "rework", "completed")}
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM parts WHERE task_id = ? GROUP BY status",
                (task_id,),
            ).fetchall()
        for row in rows:
            summary[str(row["status"])] = int(row["count"])
        summary["total"] = sum(value for key, value in summary.items() if key != "total")
        return summary

    def _require_annotator(self, connection: sqlite3.Connection, task_id: str, actor: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM task_annotators WHERE task_id = ? AND username = ?",
            (task_id, actor),
        ).fetchone()
        if row is None:
            raise PermissionError(f"{actor} is not an annotator of this task")

    def _open_session(self, connection: sqlite3.Connection, part_id: int, actor: str, now: str) -> None:
        connection.execute(
            "INSERT INTO part_work_sessions(part_id, annotator, started_at, created_at) VALUES (?, ?, ?, ?)",
            (part_id, actor, now, now),
        )

    def _close_session(self, connection: sqlite3.Connection, part_id: int, now: str) -> float:
        session = connection.execute(
            "SELECT session_id, started_at FROM part_work_sessions "
            "WHERE part_id = ? AND ended_at IS NULL ORDER BY session_id DESC LIMIT 1",
            (part_id,),
        ).fetchone()
        if session is None:
            return 0.0
        duration = _elapsed_seconds(session["started_at"], now)
        connection.execute(
            "UPDATE part_work_sessions SET ended_at = ?, duration_seconds = ? WHERE session_id = ?",
            (now, duration, session["session_id"]),
        )
        connection.execute(
            "UPDATE parts SET work_seconds = work_seconds + ? WHERE part_id = ?",
            (duration, part_id),
        )
        return duration

    def claim_next_part(self, task_id: str, actor: str, now: str) -> Optional[dict[str, Any]]:
        with self.transaction() as connection:
            self._require_annotator(connection, task_id, actor)
            part = connection.execute(
                "SELECT part_id FROM parts WHERE task_id = ? AND status = 'pending' "
                "ORDER BY part_index LIMIT 1",
                (task_id,),
            ).fetchone()
            if part is None:
                return None
            connection.execute(
                "UPDATE parts SET status='in_progress', annotator=?, claimed_at=?, "
                "submitted_at=NULL, reviewed_at=NULL, updated_at=? "
                "WHERE part_id=? AND status='pending'",
                (actor, now, now, part["part_id"]),
            )
            self._open_session(connection, int(part["part_id"]), actor, now)
            connection.execute(
                "UPDATE tasks SET status='in_progress', updated_at=? WHERE task_id=?",
                (now, task_id),
            )
            part_id = int(part["part_id"])
        return self.get_part(task_id, part_id, now)

    def get_part(self, task_id: str, part_id: int, now: Optional[str] = None) -> dict[str, Any]:
        parts = [part for part in self.list_parts(task_id, now) if part["part_id"] == part_id]
        if not parts:
            raise KeyError(part_id)
        return parts[0]

    def start_part(self, task_id: str, part_id: int, actor: str, now: str) -> dict[str, Any]:
        with self.transaction() as connection:
            self._require_annotator(connection, task_id, actor)
            part = connection.execute(
                "SELECT status, annotator FROM parts WHERE task_id=? AND part_id=?",
                (task_id, part_id),
            ).fetchone()
            if part is None:
                raise KeyError(part_id)
            if part["annotator"] != actor or part["status"] not in {"rework", "paused"}:
                raise ValueError("part cannot be started by this actor")
            connection.execute(
                "UPDATE parts SET status='in_progress', updated_at=? WHERE part_id=?",
                (now, part_id),
            )
            self._open_session(connection, part_id, actor, now)
        return self.get_part(task_id, part_id, now)

    def submit_part(
        self,
        task_id: str,
        part_id: int,
        actor: str,
        note: str,
        now: str,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            part = connection.execute(
                "SELECT status, annotator FROM parts WHERE task_id=? AND part_id=?",
                (task_id, part_id),
            ).fetchone()
            if part is None:
                raise KeyError(part_id)
            if part["annotator"] != actor or part["status"] != "in_progress":
                raise PermissionError("only the active annotator can submit this part")
            self._close_session(connection, part_id, now)
            connection.execute(
                "UPDATE parts SET status='submitted', submitted_at=?, submission_note=?, "
                "updated_at=? WHERE part_id=?",
                (now, note, now, part_id),
            )
        return self.get_part(task_id, part_id, now)

    def release_part(self, task_id: str, part_id: int, actor: str, now: str) -> dict[str, Any]:
        with self.transaction() as connection:
            part = connection.execute(
                "SELECT status, annotator FROM parts WHERE task_id=? AND part_id=?",
                (task_id, part_id),
            ).fetchone()
            if part is None:
                raise KeyError(part_id)
            if part["annotator"] != actor or part["status"] != "in_progress":
                raise PermissionError("only the active annotator can release this part")
            self._close_session(connection, part_id, now)
            connection.execute(
                "UPDATE parts SET status='pending', annotator=NULL, claimed_at=NULL, "
                "updated_at=? WHERE part_id=?",
                (now, part_id),
            )
        return self.get_part(task_id, part_id, now)

    def review_part(
        self,
        task_id: str,
        part_id: int,
        actor: str,
        action: str,
        note: str,
        now: str,
    ) -> dict[str, Any]:
        if action not in {"approve", "rework"}:
            raise ValueError("review action must be approve or rework")
        with self.transaction() as connection:
            task = connection.execute(
                "SELECT publisher, manager FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if actor not in {task["publisher"], task["manager"]}:
                raise PermissionError("only publisher or manager can review parts")
            part = connection.execute(
                "SELECT status FROM parts WHERE task_id=? AND part_id=?",
                (task_id, part_id),
            ).fetchone()
            if part is None:
                raise KeyError(part_id)
            if part["status"] != "submitted":
                raise ValueError("only submitted parts can be reviewed")
            status = "completed" if action == "approve" else "rework"
            connection.execute(
                "UPDATE parts SET status=?, review_note=?, reviewed_at=?, updated_at=? WHERE part_id=?",
                (status, note, now, now, part_id),
            )
            remaining = connection.execute(
                "SELECT COUNT(*) AS count FROM parts WHERE task_id=? AND status!='completed'",
                (task_id,),
            ).fetchone()
            task_status = "completed" if int(remaining["count"]) == 0 else "in_progress"
            connection.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                (task_status, now, task_id),
            )
        return self.get_part(task_id, part_id, now)

    def add_attachment(self, record: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO attachments(
                    attachment_id, task_id, part_id, filename, stored_path, media_type,
                    size_bytes, sha256, uploaded_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["attachment_id"], record["task_id"], record.get("part_id"),
                    record["filename"], record["stored_path"], record.get("media_type", ""),
                    int(record.get("size_bytes", 0)), record["sha256"],
                    record.get("uploaded_by", ""), record["created_at"],
                ),
            )
        return record

    def list_attachments(self, task_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM attachments WHERE task_id=? ORDER BY created_at, attachment_id",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_attachment(self, task_id: str, attachment_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM attachments WHERE task_id=? AND attachment_id=?",
                (task_id, attachment_id),
            ).fetchone()
        if row is None:
            raise KeyError(attachment_id)
        return dict(row)

    def create_issue(self, record: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO issues(
                    task_id, part_id, reported_by, assigned_to, severity, status,
                    title, description, resolution, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, '', ?, ?)
                """,
                (
                    record["task_id"], record.get("part_id"), record["reported_by"],
                    record.get("assigned_to", ""), record.get("severity", "normal"),
                    record["title"], record.get("description", ""),
                    record["created_at"], record["updated_at"],
                ),
            )
            issue_id = int(cursor.lastrowid)
        return self.get_issue(record["task_id"], issue_id)

    def get_issue(self, task_id: str, issue_id: int) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM issues WHERE task_id=? AND issue_id=?",
                (task_id, issue_id),
            ).fetchone()
        if row is None:
            raise KeyError(issue_id)
        return dict(row)

    def list_issues(self, task_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM issues WHERE task_id=? ORDER BY "
                "CASE status WHEN 'open' THEN 0 ELSE 1 END, created_at DESC",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_issue(
        self,
        task_id: str,
        issue_id: int,
        actor: str,
        resolution: str,
        now: str,
    ) -> dict[str, Any]:
        if not self.is_manager(task_id, actor):
            raise PermissionError("only publisher or manager can resolve issues")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE issues SET status='resolved', resolution=?, resolved_at=?, updated_at=? "
                "WHERE task_id=? AND issue_id=?",
                (resolution, now, now, task_id, issue_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(issue_id)
        return self.get_issue(task_id, issue_id)

    def add_event(self, task_id: str, time: str, level: str, message: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO events(task_id, time, level, message) VALUES (?, ?, ?, ?)",
                (task_id, time, level, message),
            )

    def list_events(self, task_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT time, level, message FROM events WHERE task_id = ? "
                "ORDER BY event_id DESC LIMIT ?",
                (task_id, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def event_count(self, task_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row["count"])

    def health(self) -> dict[str, Any]:
        with self.connection() as connection:
            check = connection.execute("PRAGMA quick_check").fetchone()[0]
            task_count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "path": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "quick_check": check,
            "task_count": int(task_count),
            "event_count": int(event_count),
        }

    def migrate_legacy_directory(self, tasks_root: Path) -> dict[str, int]:
        imported_tasks = 0
        imported_events = 0
        failed = 0
        for task_path in sorted(tasks_root.iterdir()):
            legacy_task = task_path / "task.json"
            if not task_path.is_dir() or not legacy_task.is_file():
                continue
            try:
                task = json.loads(legacy_task.read_text(encoding="utf-8"))
                task_id = str(task.get("task_id") or task_path.name)
                task["task_id"] = task_id
                task.setdefault("task_type", "video_detection" if task.get("videos") or task.get("video") else "general")
                task.setdefault("manager", task.get("assignee", ""))
                task.setdefault("publisher", task.get("manager", ""))
                task.setdefault("annotators", [])
                task.setdefault("instructions", task.get("notes", ""))
                if not self.task_exists(task_id):
                    self.save_task(task)
                    imported_tasks += 1
                legacy_events = task_path / "events.jsonl"
                if legacy_events.is_file() and self.event_count(task_id) == 0:
                    for line in legacy_events.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        self.add_event(
                            task_id,
                            str(event.get("time", task.get("created_at", ""))),
                            str(event.get("level", "info")),
                            str(event.get("message", "")),
                        )
                        imported_events += 1
            except (OSError, ValueError, KeyError, sqlite3.Error):
                failed += 1
        return {"tasks": imported_tasks, "events": imported_events, "failed": failed}
