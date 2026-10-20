import pytest
from fastapi.testclient import TestClient

from workflow_platform.database import PlatformDatabase
import workflow_platform.server as platform


START = "2026-09-10T10:00:00+08:00"
END = "2026-09-10T11:00:00+08:00"


@pytest.fixture
def database(tmp_path):
    database = PlatformDatabase(tmp_path / "platform.sqlite3")
    database.initialize()
    database.create_task({"task_id": "bbox", "name": "BBox test", "publisher": "publisher", "applicant": "applicant"}, 4, START)
    return database


def test_bbox_migration_preserves_legacy_data(database):
    with database.connection() as connection:
        connection.execute("ALTER TABLE parts DROP COLUMN bbox_count")
        connection.execute("UPDATE schema_info SET value='7' WHERE key='schema_version'")
    database.initialize()
    database.initialize()
    assert len(database.list_parts("bbox", START)) == 4
    assert all(part["bbox_count"] is None for part in database.list_parts("bbox", START))
    assert database.health()["schema_version"] == 11


def test_bbox_edit_clear_and_permissions(database):
    part = database.claim_next_part("bbox", "worker", START)
    part_id = part["part_id"]
    assert database.update_part("bbox", part_id, "worker", START, bbox_count=0)["bbox_count"] == 0
    assert database.update_part("bbox", part_id, "publisher", START, image_count=5)["bbox_count"] == 0
    assert database.update_part("bbox", part_id, "applicant", START, bbox_count=20)["bbox_count"] == 20
    assert database.update_part("bbox", part_id, "publisher", START, bbox_count=None)["bbox_count"] is None
    for bad in (-1, 1.5, True, "3", 2**63):
        with pytest.raises(ValueError):
            database.update_part("bbox", part_id, "publisher", START, bbox_count=bad)
    with pytest.raises(PermissionError):
        database.update_part("bbox", part_id, "stranger", START, bbox_count=5)
    database.submit_part("bbox", part_id, "worker", "", END)
    database.review_part("bbox", part_id, "publisher", "approve", "", END)
    with pytest.raises(PermissionError):
        database.update_part("bbox", part_id, "worker", END, bbox_count=5)
    assert database.update_part("bbox", part_id, "publisher", END, bbox_count=5)["bbox_count"] == 5


def test_bbox_rework_latest_submission_and_zero_duration(database):
    part = database.claim_next_part("bbox", "worker", START)
    part_id = part["part_id"]
    database.update_part("bbox", part_id, "worker", START, bbox_count=10)
    database.submit_part("bbox", part_id, "worker", "", START)
    assert database.task_annotation_statistics("bbox", START)["bboxes_per_hour"] == 0
    database.review_part("bbox", part_id, "publisher", "rework", "", END)
    assert database.task_annotation_statistics("bbox", END)["bbox_count"] == 0
    rejected = database.admin_annotation_statistics("2026-09-10T00:00:00+08:00", "2026-09-11T00:00:00+08:00", END)
    assert rejected["total"]["bbox_count"] == 0
    next_day = "2026-09-11T10:00:00+08:00"
    database.start_rework("bbox", part_id, "worker", next_day)
    database.update_part("bbox", part_id, "worker", next_day, bbox_count=15)
    unfinished = database.admin_annotation_statistics("2026-09-10T00:00:00+08:00", "2026-09-11T00:00:00+08:00", next_day)
    assert unfinished["total"]["bbox_count"] == 0
    database.submit_part("bbox", part_id, "worker", "", next_day)
    old = database.admin_annotation_statistics("2026-09-10T00:00:00+08:00", "2026-09-11T00:00:00+08:00", next_day)
    new = database.admin_annotation_statistics("2026-09-11T00:00:00+08:00", "2026-09-12T00:00:00+08:00", next_day)
    assert old["total"]["bbox_count"] == 0
    assert new["total"]["bbox_count"] == 15
    assert database.task_annotation_statistics("bbox", next_day)["bbox_count"] == 15


def test_bbox_statistics_missing_zero_and_known(database):
    for count in (None, 0, 12):
        part = database.claim_next_part("bbox", "worker", START)
        if count is not None:
            database.update_part("bbox", part["part_id"], "worker", START, bbox_count=count)
        database.submit_part("bbox", part["part_id"], "worker", "", END)
    active = database.claim_next_part("bbox", "worker", END)
    database.update_part("bbox", active["part_id"], "worker", END, bbox_count=99)
    person = database.annotator_statistics("bbox", END)[0]
    task_stats = database.task_annotation_statistics("bbox", END)
    admin = database.admin_annotation_statistics("2026-09-10T00:00:00+08:00", "2026-09-11T00:00:00+08:00", END)
    for stats in (person, task_stats, admin["total"], admin["users"][0], admin["users"][0]["tasks"][0]):
        assert stats["bbox_count"] == 13
        assert stats["bboxes_per_hour"] == 4.333
    empty = database.admin_annotation_statistics("2026-09-11T00:00:00+08:00", "2026-09-12T00:00:00+08:00", END)
    assert empty["total"]["bbox_count"] == 0


def test_bbox_api_roundtrip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setitem(platform.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setitem(platform.SETTINGS, "database_path", str(tmp_path / "api.sqlite3"))
    with TestClient(platform.app) as client:
        assert client.post("/api/auth/bootstrap-admin", json={"username": "publisher", "password": "publisher-pass"}).status_code == 200
        database = platform.database()
        database.create_task({"task_id": "bbox", "name": "BBox test", "publisher": "publisher"}, 1, START)
        part_id = database.list_parts("bbox", START)[0]["part_id"]
        route = f"/api/tasks/bbox/parts/{part_id}"
        assert client.patch(route, json={"bbox_count": 24}).json()["part"]["bbox_count"] == 24
        assert client.patch(route, json={"image_count": 10}).json()["part"]["bbox_count"] == 24
        assert client.patch(route, json={"bbox_count": 0}).json()["part"]["bbox_count"] == 0
        assert client.patch(route, json={"bbox_count": None}).json()["part"]["bbox_count"] is None
        for bad in (-1, 1.5, True, "3", 2**63):
            assert client.patch(route, json={"bbox_count": bad}).status_code == 422
