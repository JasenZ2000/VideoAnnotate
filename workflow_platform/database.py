from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


SCHEMA_VERSION = 1
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
                """
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
            "deleted_at", "classes", "classes_text", "stages", "videos", "events",
        }
        extra = _without(task, known_task_keys)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, name, assignee, notes, prelabel_source, prompt, status,
                    deleted, current_video_id, created_at, updated_at, deleted_at, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    extra_json=excluded.extra_json
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
                ),
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
