from workflow_platform.database import PlatformDatabase
from workflow_platform import server


NOW = "2026-10-14T18:00:00+08:00"


def test_completed_tasks_do_not_consume_schedule_capacity(tmp_path, monkeypatch):
    db = PlatformDatabase(tmp_path / "platform.sqlite3")
    db.initialize()
    db.create_user("owner", "hash", "admin", "", NOW)
    db.check_in("owner", "2026-10-14", NOW)
    for task_id, hours, rank in (("done", "80", 1), ("active", "8", 2)):
        db.create_task({"task_id": task_id, "name": task_id, "publisher": "owner", "estimated_hours": hours, "rank": rank}, 1, NOW)
    with db.transaction() as connection:
        connection.execute("UPDATE parts SET status='completed' WHERE task_id='done'")
    monkeypatch.setattr(server, "database", lambda: db)
    tasks = {task["task_id"]: task for task in server.build_task_overview(NOW)["tasks"]}
    assert tasks["done"]["estimated_completion_date"] is None
    assert tasks["done"]["estimated_completion_days"] is None
    assert tasks["active"]["estimated_completion_date"] == "2026-10-14"
    assert tasks["active"]["estimated_completion_days"] == 1


def test_settlement_totals_include_overrides_and_exclude_other_dates(tmp_path):
    db = PlatformDatabase(tmp_path / "platform.sqlite3")
    db.initialize()
    db.create_task({"task_id": "task", "name": "task", "publisher": "owner", "estimated_hours": "6"}, 3, NOW)
    with db.transaction() as connection:
        parts = connection.execute("SELECT part_id FROM parts ORDER BY part_id").fetchall()
        connection.execute("UPDATE parts SET submitted_at=?,work_seconds=3600 WHERE part_id=?", (NOW, parts[0][0]))
        connection.execute("UPDATE parts SET submitted_at=?,work_seconds=7200,work_seconds_override=1800 WHERE part_id=?", (NOW, parts[1][0]))
        connection.execute("UPDATE parts SET submitted_at='2026-10-13T18:00:00+08:00',work_seconds=36000 WHERE part_id=?", (parts[2][0],))
    work = db.attendance_settlements("2026-10-14")["work"]
    assert work["total_expected_hours"] == 4
    assert work["total_actual_hours"] == 1.5
    empty = db.attendance_settlements("2026-10-15")["work"]
    assert empty["total_expected_hours"] == 0
    assert empty["total_actual_hours"] == 0
