import json
import uuid

import pytest
from fastapi.testclient import TestClient

from workflow_platform import server
from workflow_platform.database import PlatformDatabase


NOW = "2026-10-15T18:00:00+08:00"


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = PlatformDatabase(tmp_path / "platform.sqlite3")
    database.initialize()
    database.create_user("owner", "hash", "admin", "Owner", NOW)
    database.create_user("worker", "hash", "user", "Worker", NOW)
    with database.transaction() as connection:
        connection.execute("UPDATE users SET phone='13800138000'")
    database.create_task({"task_id": "task", "name": "测试项目", "project": "测试项目", "publisher": "owner", "status": "ready"}, 2, NOW)
    database.create_task({"task_id": "other", "name": "其他项目", "publisher": "owner"}, 1, NOW)
    database.save_ding_robot("测试机器人", "https://oapi.dingtalk.com/robot/send?access_token=test", "test-secret", NOW)
    monkeypatch.setattr(server, "database", lambda: database)
    monkeypatch.setattr(server, "now_iso", lambda: NOW)
    monkeypatch.setattr(server, "process_reminder_events", lambda: None)
    return database


def rule(**changes):
    return {"task_id": "task", "event": "priority", "robot_id": 1, "usernames": ["worker"],
            "enabled": True, "note": "请优先处理", **changes}


def test_completed_task_actual_hours_and_completion_date(db):
    with db.transaction() as connection:
        connection.execute("UPDATE parts SET status='completed',work_seconds=3600,reviewed_at=? WHERE task_id='task'", (NOW,))
        connection.execute("UPDATE parts SET work_seconds_override=0 WHERE task_id='task' AND part_index=1")
    task = next(task for task in db.task_overview(NOW) if task["task_id"] == "task")
    assert task["actual_hours"] == 1
    assert task["completed_at"] == NOW
    with db.transaction() as connection:
        connection.execute("UPDATE parts SET work_seconds=0,work_seconds_override=NULL,reviewed_at=NULL WHERE task_id='task' AND part_index=1")
    task = next(task for task in db.task_overview(NOW) if task["task_id"] == "task")
    assert task["actual_hours"] is None
    assert task["completed_at"] is None


def test_zero_duration_session_is_tracked_and_legacy_zero_is_unknown(db):
    with db.transaction() as connection:
        connection.execute("UPDATE parts SET status='completed',reviewed_at=?", (NOW,))
        connection.execute("INSERT INTO part_work_sessions(part_id,annotator,started_at,ended_at,duration_seconds,created_at) SELECT part_id,'worker',?,?,0,? FROM parts WHERE task_id='task'", (NOW, NOW, NOW))
    tasks = {task["task_id"]: task for task in db.task_overview(NOW)}
    assert tasks["task"]["actual_hours"] == 0
    assert tasks["other"]["actual_hours"] is None


def test_task_rule_api_crud_validation_and_admin_permission(db):
    for username in ("owner", "worker"):
        db.create_session(username, username, server.session_token_hash(username), "2026-10-16T18:00:00+08:00", NOW)
    identifier = str(uuid.uuid4())
    url = "/api/admin/task-reminders/" + identifier
    with TestClient(server.app) as client:
        client.cookies.set(server.SESSION_COOKIE_NAME, "owner")
        response = client.put(url, json=rule())
        assert response.status_code == 200, response.text
        assert response.json()["rules"][0]["note"] == "请优先处理"
        db.initialize()
        assert len(client.get("/api/admin/task-reminders").json()["rules"]) == 1
        assert client.put(url, json=rule(enabled=False, note="新的备注")).status_code == 200
        assert db.task_reminder_rules()[0]["enabled"] is False
        for changes, status in [({"event": "invalid"}, 400), ({"task_id": "missing"}, 404), ({"robot_id": 999}, 400), ({"usernames": ["missing"]}, 400), ({"usernames": []}, 422)]:
            assert client.put(url, json=rule(**changes)).status_code == status
        assert client.put("/api/admin/task-reminders/invalid-id", json=rule()).status_code == 422
        assert client.delete(url).json()["rules"] == []
        client.cookies.set(server.SESSION_COOKIE_NAME, "worker")
        assert client.get("/api/admin/task-reminders").status_code == 403
        assert client.put(url, json=rule()).status_code == 403
        assert client.delete(url).status_code == 403


@pytest.mark.parametrize("event", ["started", "priority", "completed"])
@pytest.mark.parametrize("global_enabled", [False, True])
def test_task_rules_coexist_with_global_rules_and_deduplicate(db, monkeypatch, event, global_enabled):
    global_rule = server.EventReminderRule(enabled=global_enabled, robot_id=1, usernames=["owner"]).model_dump()
    monkeypatch.setattr(server, "reminder_settings", lambda: {"rules": {event: global_rule}})
    monkeypatch.setattr(server, "notification_content", lambda req, now, event: event + " " + req.note)
    db.save_task_reminder_rule("matching", rule(event=event))
    db.save_task_reminder_rule("disabled", rule(event=event, enabled=False))
    db.save_task_reminder_rule("different-task", rule(event=event, task_id="other"))
    db.save_task_reminder_rule("different-event", rule(event="published"))
    for attempt in range(2):
        server.enqueue_task_reminder(event, "task", "owner", "event-123", "事件备注")
    with db.connection() as connection:
        events = connection.execute("SELECT * FROM reminder_events ORDER BY event_id").fetchall()
    assert len(events) == (2 if global_enabled else 1)
    specific = next(item for item in events if "task-rule:" in item["event_id"])
    payload = json.loads(specific["payload"])
    assert payload["usernames"] == ["worker"]
    assert "请优先处理" in payload["content"]
    assert "事件备注" in payload["content"]


def test_invalid_task_rule_does_not_block_other_rules(db, monkeypatch):
    monkeypatch.setattr(server, "reminder_settings", lambda: {"rules": {"priority": server.EventReminderRule().model_dump()}})
    monkeypatch.setattr(server, "notification_content", lambda *args, **kwargs: "通知")
    db.save_task_reminder_rule("bad", rule(robot_id=999))
    db.save_task_reminder_rule("good", rule())
    server.enqueue_task_reminder("priority", "task", "owner", "event")
    assert len(db.reminder_history()) == 1
    assert db.reminder_history()[0]["event_id"] == "event:task-rule:good"
    with db.connection() as connection:
        payload = json.loads(connection.execute("SELECT payload FROM reminder_events").fetchone()[0])
    assert "请优先处理" in payload["content"]


@pytest.mark.parametrize("late_parts", [False, True])
def test_task_lifecycle_enqueues_specific_reminders_through_api(db, late_parts):
    if late_parts:
        db.add_parts("task", 1, "owner", NOW)
    for username in ("owner", "worker"):
        db.create_session(username, username, server.session_token_hash(username), "2026-10-16T18:00:00+08:00", NOW)
    for event in ("started", "priority", "completed"):
        db.save_task_reminder_rule(event, rule(event=event))
    with TestClient(server.app) as client:
        client.cookies.set(server.SESSION_COOKIE_NAME, "owner")
        response = client.patch("/api/tasks/task/ordering", json={"priority": "urgent"})
        assert response.status_code == 200, response.text
        for index in range(3 if late_parts else 2):
            client.cookies.set(server.SESSION_COOKIE_NAME, "worker")
            response = client.post("/api/tasks/task/parts/claim-next")
            assert response.status_code == 200, response.text
            part_id = response.json()["part"]["part_id"]
            assert client.post(f"/api/tasks/task/parts/{part_id}/submit", json={}).status_code == 200
            client.cookies.set(server.SESSION_COOKIE_NAME, "owner")
            assert client.post(f"/api/tasks/task/parts/{part_id}/review", json={"action": "approve"}).status_code == 200
    assert sorted(item["event_type"] for item in db.reminder_history()) == ["completed", "priority", "started"]
