import pytest
from fastapi.testclient import TestClient

import workflow_platform.server as platform


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setitem(platform.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setitem(platform.SETTINGS, "database_path", str(tmp_path / "platform.sqlite3"))
    with TestClient(platform.app) as admin, TestClient(platform.app) as publisher, TestClient(platform.app) as viewer:
        assert admin.post("/api/auth/bootstrap-admin", json={"username": "admin", "password": "admin-pass"}).status_code == 200
        for username in ("publisher", "viewer"):
            assert admin.post("/api/users", json={"username": username, "password": username + "-pass"}).status_code == 200
        for client, username in ((publisher, "publisher"), (viewer, "viewer")):
            assert client.post("/api/auth/login", json={"username": username, "password": username + "-pass"}).status_code == 200
        yield admin, publisher, viewer


def publish(client, count=0, applicant="自由填写姓名"):
    payload = {"input_mode": "manual", "product_tag": "BSD", "annotation_tool": "Label Studio",
               "part_count": count, "task": {"application_date": "2026-10-01", "applicant": applicant,
               "project": "道路", "annotation_content": "车辆", "data_path": r"D:\dataset",
               "data_amount": "100", "guide_path": r"D:\guide.pdf"}}
    preview = client.post("/api/tasks/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["part_count"] == count
    response = client.post("/api/tasks", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["tasks"][0]["task_id"]


def test_manual_publish_without_parts_then_append_manifest(clients):
    admin, publisher, viewer = clients
    task_id = publish(publisher)
    route = f"/api/tasks/{task_id}"
    detail = publisher.get(route).json()
    assert detail["annotation_tool"] == "Label Studio"
    assert detail["application_date"] == "2026-10-01"
    assert detail["applicant"] == "自由填写姓名"
    assert detail["part_summary"]["total"] == 0
    assert viewer.get("/api/task-overview").json()["tasks"][0]["overview_status"] == "awaiting_parts"
    assert viewer.post(route + "/parts/claim-next").status_code == 409
    manifest = {"part_manifest": "split_a\nsplit_b"}
    assert viewer.post(route + "/parts/preview", json=manifest).status_code == 403
    assert viewer.post(route + "/parts", json=manifest).status_code == 403
    preview = publisher.post(route + "/parts/preview", json=manifest)
    assert preview.status_code == 200, preview.text
    assert preview.json()["part_preview"][0]["work_path"] == r"D:\dataset\split_a"
    assert publisher.post(route + "/parts", json=manifest).json()["added"] == 2
    assert viewer.get("/api/task-overview").json()["tasks"][0]["overview_status"] == "pending"
    assert publisher.post(route + "/parts", json={"part_manifest": "split_c\nSPLIT_A"}).status_code == 400
    assert publisher.get(route).json()["part_summary"]["total"] == 2
    assert publisher.post(route + "/parts/preview", json=manifest).status_code == 400
    assert publisher.patch(route, json={"data_path": r"D:\new", "application_date": "2026-10-02"}).status_code == 200
    assert publisher.post(route + "/parts", json={"part_manifest": "split_c"}).status_code == 200
    parts = publisher.get(route).json()["parts"]
    assert parts[0]["work_path"] == r"D:\dataset\split_a"
    assert parts[2]["work_path"] == r"D:\new\split_c"


def test_applicant_matches_active_user_by_username_or_display_name(clients):
    admin, publisher, viewer = clients
    assert admin.patch('/api/users/viewer', json={'display_name': '目录申请人'}).status_code == 200
    assert any(user['username'] == 'viewer' for user in publisher.get('/api/active-users').json()['users'])
    by_display_name = publish(publisher, 1, applicant='目录申请人')
    by_username = publish(publisher, 1, applicant='viewer')
    assert publisher.get(f'/api/tasks/{by_display_name}').json()['applicant'] == 'viewer'
    assert viewer.get(f'/api/tasks/{by_display_name}').json()['is_applicant']
    assert publisher.get(f'/api/tasks/{by_username}').json()['applicant'] == 'viewer'
    assert publish(publisher, 1, applicant='外部申请人')


@pytest.mark.parametrize('applicant_name', ['viewer', '目录申请人'])
def test_matched_applicant_reviews_other_workers_without_task_management(clients, applicant_name, monkeypatch):
    admin, publisher, applicant = clients
    assert admin.patch('/api/users/viewer', json={'display_name': '目录申请人'}).status_code == 200
    for username in ('annotator', 'outsider'):
        assert admin.post('/api/users', json={'username': username, 'password': username + '-pass'}).status_code == 200
    task_id = publish(publisher, 1, applicant=applicant_name)
    route = f'/api/tasks/{task_id}'
    with TestClient(platform.app) as annotator, TestClient(platform.app) as outsider:
        for client, username in ((annotator, 'annotator'), (outsider, 'outsider')):
            assert client.post('/api/auth/login', json={'username': username, 'password': username + '-pass'}).status_code == 200
        claim = annotator.post(route + '/parts/claim-next')
        assert claim.status_code == 200, claim.text
        part_id = claim.json()['part']['part_id']
        part_route = f'{route}/parts/{part_id}'
        assert annotator.post(part_route + '/submit', json={'note': 'ready for applicant'}).status_code == 200
        detail = applicant.get(route).json()
        assert detail['applicant'] == 'viewer'
        assert detail['is_applicant'] and detail['can_review'] and detail['can_view_all']
        assert detail['parts'][0]['annotator'] == 'annotator'
        listed = next(task for task in applicant.get('/api/tasks').json()['tasks'] if task['task_id'] == task_id)
        assert listed['can_review'] and listed['is_applicant']
        assert not outsider.get(route).json()['can_review']
        assert outsider.post(part_route + '/review', json={'action': 'approve'}).status_code == 403
        assert annotator.post(part_route + '/review', json={'action': 'approve'}).status_code == 403
        assert applicant.post(part_route + '/review', json={'action': 'rework', 'note': '请修正'}).status_code == 200
        assert annotator.post(part_route + '/start-rework').status_code == 200
        assert annotator.post(part_route + '/submit', json={'note': 'fixed'}).status_code == 200
        events = []
        monkeypatch.setattr(platform, 'enqueue_task_reminder', lambda *args, **kwargs: events.append(args))
        assert applicant.post(part_route + '/review', json={'action': 'approve'}).status_code == 200
        assert events == [('completed', task_id, 'viewer', 'completed:' + task_id)]
        assert applicant.get(route).json()['status'] == 'completed'
        assert applicant.patch(route, json={'project': 'unauthorized'}).status_code == 403
        assert applicant.post(route + '/parts', json={'count': 1}).status_code == 403
        assert applicant.delete(route).status_code == 403
        assert applicant.post(part_route + '/time-review', json={'decision': 'estimate_reasonable'}).status_code == 403


def test_admin_manages_other_publishers_task_and_audits_actor(clients):
    admin, publisher, viewer = clients
    task_id = publish(publisher, 1)
    route = f"/api/tasks/{task_id}"
    detail = admin.get(route).json()
    assert detail["publisher"] == "publisher"
    assert detail["can_manage_task"] and not detail["is_publisher"]
    assert "statistics" in detail and "annotation_statistics" in detail
    assert viewer.patch(route, json={"project": "越权"}).status_code == 403
    assert viewer.delete(route).status_code == 403
    assert admin.patch(route, json={"project": "修改项目", "expected_part_seconds": 3600}).status_code == 200
    assert admin.post(route + "/parts", json={"count": 1}).status_code == 200
    part = viewer.post(route + "/parts/claim-next").json()["part"]
    part_route = route + "/parts/" + str(part["part_id"])
    assert admin.patch(part_route, json={"bbox_count": 20, "image_count": 80, "work_seconds": 60, "name": "修正目录"}).status_code == 200
    assert viewer.post(part_route + "/submit", json={"note": "完成"}).status_code == 200
    assert admin.post(part_route + "/time-review", json={"decision": "estimate_reasonable"}).status_code == 200
    assert admin.post(part_route + "/review", json={"action": "approve"}).status_code == 200
    assert admin.post(part_route + "/comments", json={"content": "管理员备注"}).status_code == 200
    detail = admin.get(route).json()
    assert detail["publisher"] == "publisher"
    assert any(log["actor"] == "admin" and log["action"] == "edit_task" for log in detail["audit_logs"])
    assert any(log["actor"] == "admin" and log["action"] == "add_parts" for log in detail["audit_logs"])
    assert viewer.delete(part_route).status_code == 403
    assert admin.delete(part_route).status_code == 200
    assert admin.delete(route).status_code == 200
    assert publisher.get(route).status_code == 404
    assert admin.get("/api/task-overview").json()["tasks"] == []
    assert admin.get("/api/directory-overview").json()["total"] == 0


def test_public_overview_states_directory_filters_and_no_private_notes(clients):
    admin, publisher, viewer = clients
    task_id = publish(publisher, 2)
    route = f"/api/tasks/{task_id}"
    part = publisher.post(route + "/parts/claim-next").json()["part"]
    part_route = route + "/parts/" + str(part["part_id"])
    assert publisher.patch(part_route, json={"work_path": r"D:\dataset\active"}).status_code == 200
    assert publisher.post(part_route + "/comments", json={"content": "private note"}).status_code == 200
    assert viewer.get("/api/task-overview").json()["tasks"][0]["overview_status"] == "in_progress"
    data = viewer.get("/api/directory-overview", params={"status": "in_progress", "query": "active", "product_tag": "BSD", "page_size": 1}).json()
    assert data["total"] == 1 and len(data["items"]) == 1
    assert data["items"][0]["work_path"] == r"D:\dataset\active"
    assert "comments" not in data["items"][0] and "work_seconds" not in data["items"][0]
    assert viewer.get("/api/directory-overview?product_tag=Other").json()["total"] == 0
    assert viewer.get("/api/directory-overview?status=invalid").status_code == 400
    assert viewer.get("/api/directory-overview?page=0").status_code == 400
    assert viewer.get("/api/directory-overview?page_size=201").status_code == 400
    assert viewer.get("/api/directory-overview?page_size=1&page=2").json()["items"][0]["part_id"] != part["part_id"]
    assert publisher.post(part_route + "/submit", json={}).status_code == 200
    assert viewer.get("/api/task-overview").json()["tasks"][0]["overview_status"] == "mixed"
    assert admin.post(part_route + "/review", json={"action": "approve"}).status_code == 200
    second = publisher.post(route + "/parts/claim-next").json()["part"]
    second_route = route + "/parts/" + str(second["part_id"])
    assert publisher.post(second_route + "/submit", json={}).status_code == 200
    assert viewer.get("/api/task-overview").json()["tasks"][0]["overview_status"] == "submitted"
    assert admin.post(second_route + "/review", json={"action": "approve"}).status_code == 200
    summary = viewer.get("/api/task-overview").json()["tasks"][0]
    assert summary["overview_status"] == "completed" and summary["completion_percent"] == 100
    assert admin.post(route + "/parts", json={"part_manifest": "next"}).status_code == 200
    summary = viewer.get("/api/task-overview").json()["tasks"][0]
    assert summary["overview_status"] != "completed" and summary["part_summary"]["pending"] == 1
    assert admin.get(route).json()["status"] != "completed"
    viewer.post("/api/auth/logout")
    assert viewer.get("/api/task-overview").status_code == 401
    assert viewer.get("/api/directory-overview").status_code == 401


@pytest.mark.parametrize("payload", [
    {"input_mode": "manual", "task": {}},
    {"input_mode": "invalid", "task": {"project": "test"}},
    {"input_mode": "manual", "task": {"project": "test"}, "part_count": -1},
    {"input_mode": "manual", "task": {"project": "test"}, "part_count": 10001},
])
def test_publication_validation_matches_preview(clients, payload):
    admin, publisher, viewer = clients
    payload["product_tag"] = "BSD"
    assert publisher.post("/api/tasks/preview", json=payload).status_code == 400
    assert publisher.post("/api/tasks", json=payload).status_code == 400


def test_table_publication_can_defer_parts(clients):
    admin, publisher, viewer = clients
    payload = {"clipboard_text": "2026-10-12\tpublisher\tProject\tContent", "product_tag": "BSD", "part_count": 0}
    assert publisher.post("/api/tasks/preview", json=payload).json()["part_count"] == 0
    response = publisher.post("/api/tasks", json=payload)
    assert response.status_code == 200
    task_id = response.json()["tasks"][0]["task_id"]
    assert publisher.get(f"/api/tasks/{task_id}").json()["part_summary"]["total"] == 0




def test_attendance_and_estimated_completion(clients):
    admin, publisher, viewer = clients
    payload = {"input_mode": "manual", "product_tag": "OPS", "part_count": 1, "task": {
        "application_date": "2026-10-12", "applicant": "publisher", "project": "工时任务",
        "annotation_content": "内容", "estimated_hours": "16", "data_path": "D:/data"}}
    assert publisher.post("/api/tasks", json=payload).status_code == 200
    assert publisher.get("/api/attendance").json()["checked_in_count"] == 0
    assert publisher.post("/api/attendance/check-in").status_code == 200
    assert publisher.post("/api/attendance/check-in").json()["work_date"]
    assert admin.patch("/api/admin/attendance", json={"work_start": "08:30", "work_end": "17:30"}).status_code == 200
    summary = viewer.get("/api/task-overview").json()
    assert summary["attendance"]["checked_in_count"] == 1
    task = summary["tasks"][0]
    assert task["estimated_completion_hours"] == 16
    assert task["estimated_completion_days"] == 2
    assert task["estimated_completion_date"]
    assert viewer.patch("/api/admin/attendance", json={"work_start": "09:00", "work_end": "18:00"}).status_code == 403


def test_part_work_actions_automatically_check_in_once(clients):
    admin, publisher, worker = clients
    task_id = publish(publisher, 2)
    assert worker.get('/api/attendance').json()['checked_in_count'] == 0
    assert worker.post(f'/api/tasks/{task_id}/parts/999/resume').status_code == 404
    assert worker.get('/api/attendance').json()['checked_in_count'] == 0
    claimed = worker.post(f'/api/tasks/{task_id}/parts/claim-next')
    assert claimed.status_code == 200, claimed.text
    part_id = claimed.json()['part']['part_id']
    status = worker.get('/api/attendance').json()
    assert status['checked_in'] and status['checked_in_count'] == 1

    assert worker.post(f'/api/tasks/{task_id}/parts/{part_id}/pause').status_code == 200
    assert worker.post(f'/api/tasks/{task_id}/parts/{part_id}/resume').status_code == 200
    assert worker.get('/api/attendance').json()['checked_in_count'] == 1

    assert worker.post(f'/api/tasks/{task_id}/parts/{part_id}/submit', json={}).status_code == 200
    assert publisher.post(f'/api/tasks/{task_id}/parts/{part_id}/review', json={'action': 'rework'}).status_code == 200
    assert worker.post(f'/api/tasks/{task_id}/parts/{part_id}/start-rework').status_code == 200
    assert worker.get('/api/attendance').json()['checked_in_count'] == 1

    assert worker.get('/api/attendance').json()['checked_in_count'] == 1

@pytest.mark.parametrize('start,days,expected', [('2026-10-16',1,'2026-10-16'),('2026-10-16',2,'2026-10-19'),('2026-10-17',1,'2026-10-19'),('2026-10-18',6,'2026-10-26')])
def test_working_day_completion(start, days, expected):
    from datetime import date
    assert platform.working_day_completion(date.fromisoformat(start), days).isoformat() == expected


def test_seven_day_settlement_window_and_user_changes(clients):
    admin, publisher, viewer = clients
    db = platform.database()
    for work_date in ('2026-10-01', '2026-10-08', '2026-10-13'):
        db.check_in('publisher', work_date, work_date + 'T08:00:00+08:00')
    result = viewer.get('/api/attendance/settlements?days=30&date=2026-10-13').json()
    assert [item['date'] for item in result['attendance']['daily']] == [f'2026-10-{day:02d}' for day in range(7,14)]
    assert sum(item['count'] for item in result['attendance']['daily']) == 2
    assert result['attendance']['users'][0]['username'] == 'publisher'
    assert len(result['work']['daily']) == 7
    assert admin.patch('/api/users/publisher', json={'phone':'13800138000','display_name':'Updated'}).status_code == 200
    assert next(user for user in admin.get('/api/users').json()['users'] if user['username']=='publisher')['phone'] == '13800138000'
    assert viewer.patch('/api/users/publisher', json={'phone':'13900139000'}).status_code == 403
    assert admin.delete('/api/users/viewer').status_code == 200
    assert all(user['username'] != 'viewer' for user in admin.get('/api/users').json()['users'])


def test_notification_uses_current_user_phones(clients, monkeypatch):
    import json
    import io
    admin, publisher, viewer = clients
    admin.patch('/api/users/publisher', json={'phone':'13800138000'})
    admin.patch('/api/users/viewer', json={'phone':'13900139000'})
    robot = admin.post('/api/admin/dingtalk/robots', json={'name':'Test','webhook_url':'https://oapi.dingtalk.com/robot/send?access_token=test','secret':'test-secret'}).json()['robot_id']
    sent=[]
    def send(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{"errcode":0}')
    monkeypatch.setattr(platform.urllib.request, 'urlopen', send)
    assert admin.post('/api/admin/dingtalk/test', json={'robot_id':robot,'content':'hello','usernames':['publisher','viewer']}).status_code == 200
    assert sent[-1]['at']['atMobiles'] == ['13800138000','13900139000']
    admin.patch('/api/users/publisher', json={'phone':'13700137000'})
    assert admin.post('/api/admin/dingtalk/test', json={'robot_id':robot,'content':'single','usernames':['publisher']}).status_code == 200
    assert sent[-1]['at']['atMobiles'] == ['13700137000']
    assert admin.post('/api/admin/dingtalk/test', json={'robot_id':robot,'content':'invalid','usernames':['missing']}).status_code == 400
    assert viewer.post('/api/admin/dingtalk/test', json={'robot_id':robot,'content':'forbidden'}).status_code == 403


def test_task_table_notifications_and_daily_totals(clients, monkeypatch):
    admin, publisher, viewer = clients
    first = publish(publisher, 2)
    second = publish(publisher, 1)
    admin.patch(f'/api/tasks/{first}', json={'estimated_hours':'8'})
    admin.patch(f'/api/tasks/{second}', json={'estimated_hours':'4'})
    admin.patch(f'/api/tasks/{first}/ordering', json={'rank':1})
    db = platform.database()
    db.check_in('publisher','2026-10-13','2026-10-13T08:30:00+08:00')
    part = db.claim_next_part(first,'publisher','2026-10-13T09:00:00+08:00')
    db.submit_part(first,part['part_id'],'publisher','done','2026-10-13T10:00:00+08:00')
    now='2026-10-13T18:00:00+08:00'
    monkeypatch.setattr(platform,'now_iso',lambda:now)
    robot = admin.post('/api/admin/dingtalk/robots',json={'name':'Report','webhook_url':'https://oapi.dingtalk.com/robot/send?access_token=test','secret':'secret'}).json()['robot_id']
    admin.patch('/api/users/publisher',json={'phone':'13800138000'})
    overview=admin.get('/api/task-overview').json()['tasks']
    assert [item['task_id'] for item in overview]==[first,second]
    body={'kind':'task','task_id':first,'robot_id':robot,'usernames':['publisher'],'note':'check parts'}
    content=admin.post('/api/admin/notifications/preview',json=body).json()['content']
    assert '任务提醒' in content and 'check parts' in content and '道路' in content
    body['kind']='daily'
    report=admin.post('/api/admin/notifications/preview',json=body).json()['content']
    assert '任务表格' in report and '今日报到：1 人' in report
    assert '预计 4.00 小时；实际 1.00 小时' in report
    sent=[]
    monkeypatch.setattr(platform,'send_dingtalk',lambda req:sent.append(req) or {'ok':True})
    assert admin.post('/api/admin/notifications/send',json=body).status_code==200
    assert sent[0].usernames==['publisher'] and sent[0].content==report
    assert viewer.post('/api/admin/notifications/send',json=body).status_code==403
    assert viewer.patch('/api/admin/notifications',json={'enabled':False}).status_code==403


def test_automatic_daily_report_schedule_and_deduplication(clients, monkeypatch):
    admin, publisher, viewer = clients
    robot = admin.post('/api/admin/dingtalk/robots',json={'name':'Report','webhook_url':'https://oapi.dingtalk.com/robot/send?access_token=test','secret':'secret'}).json()['robot_id']
    admin.patch('/api/users/publisher',json={'phone':'13800138000'})
    assert admin.patch('/api/admin/notifications',json={'enabled':True,'robot_id':robot,'usernames':['missing']}).status_code==400
    assert admin.patch('/api/admin/notifications',json={'enabled':True,'robot_id':robot,'usernames':['publisher'],'note':'Daily'}).status_code==200
    sent=[]
    monkeypatch.setattr(platform,'send_dingtalk',lambda req:sent.append(req) or {'ok':True})
    platform.run_automatic_report('2026-10-13T17:59:00+08:00')
    platform.run_automatic_report('2026-10-17T18:00:00+08:00')
    assert not sent
    platform.run_automatic_report('2026-10-13T18:00:00+08:00')
    platform.run_automatic_report('2026-10-13T18:01:00+08:00')
    assert len(sent)==1
    assert admin.get('/api/admin/notifications').json()['latest_run']['status']=='sent'
    platform.database().initialize()
    platform.run_automatic_report('2026-10-13T19:00:00+08:00')
    assert len(sent)==1
    admin.patch('/api/admin/notifications',json={'enabled':False})
    platform.run_automatic_report('2026-10-14T18:00:00+08:00')
    assert len(sent)==1


def test_automatic_daily_report_failure_is_visible(clients, monkeypatch):
    admin, publisher, viewer=clients
    robot=admin.post('/api/admin/dingtalk/robots',json={'name':'Report','webhook_url':'https://oapi.dingtalk.com/robot/send?access_token=test','secret':'secret'}).json()['robot_id']
    admin.patch('/api/admin/notifications',json={'enabled':True,'robot_id':robot})
    attempts=[]
    def fail(req):
        attempts.append(req)
        raise RuntimeError('network failure')
    monkeypatch.setattr(platform,'send_dingtalk',fail)
    platform.run_automatic_report('2026-10-13T18:00:00+08:00')
    platform.run_automatic_report('2026-10-13T18:01:00+08:00')
    assert len(attempts)==1
    assert admin.get('/api/admin/notifications').json()['latest_run']['status']=='failed'


def test_reminder_management_permissions_templates_and_event_delivery(clients, monkeypatch):
    admin, publisher, viewer=clients
    assert admin.get('/reminders').status_code==200
    assert viewer.get('/reminders').status_code==403
    assert viewer.get('/api/admin/reminders').status_code==403
    assert admin.patch('/api/users/publisher',json={'phone':'13800138000'}).status_code==200
    assert admin.patch('/api/users/viewer',json={'phone':'13900139000'}).status_code==200
    robot=admin.post('/api/admin/dingtalk/robots',json={'name':'Events','webhook_url':'https://oapi.dingtalk.com/robot/send?access_token=test','secret':'secret'}).json()['robot_id']
    settings=admin.get('/api/admin/reminders').json()
    for rule in settings['rules'].values():
        rule.update(enabled=True,robot_id=robot,usernames=['viewer'],notify_publisher=True,notify_applicant=True)
    settings['task_template']='{event}: {project} / {publisher} / {note}'
    settings['daily_template']='Daily {date}: {attendance_count} people / {completed_parts} parts / {actual_hours}h\n{content}'
    assert viewer.patch('/api/admin/reminders',json=settings).status_code==403
    assert admin.patch('/api/admin/reminders',json=settings).status_code==200
    assert admin.post('/api/users',json={'username':'applicant','password':'applicant-pass','display_name':'任务申请人'}).status_code==200
    assert admin.patch('/api/users/applicant',json={'phone':'13700137000'}).status_code==200
    task_id=publish(publisher,1,applicant='任务申请人')
    assert admin.patch(f'/api/tasks/{task_id}/ordering',json={'priority':'high'}).status_code==200
    assert admin.patch(f'/api/tasks/{task_id}/ordering',json={'priority':'high'}).status_code==200
    claim=publisher.post(f'/api/tasks/{task_id}/parts/claim-next')
    assert claim.status_code==200
    part_id=claim.json()['part']['part_id']
    assert publisher.post(f'/api/tasks/{task_id}/parts/{part_id}/submit',json={}).status_code==200
    assert admin.post(f'/api/tasks/{task_id}/parts/{part_id}/review',json={'action':'approve'}).status_code==200
    sent=[]
    monkeypatch.setattr(platform,'send_dingtalk',lambda req:sent.append(req) or {'ok':True})
    platform.process_reminder_events()
    platform.process_reminder_events()
    assert len(sent)==4
    assert all(set(req.usernames)=={'publisher','viewer','applicant'} for req in sent)
    assert any('新任务发布: 道路 / publisher' in req.content for req in sent)
    assert any('任务优先级变动' in req.content for req in sent)
    assert any('任务开始' in req.content for req in sent)
    assert any('任务完成' in req.content for req in sent)
    assert all(item['status']=='sent' for item in admin.get('/api/admin/reminders').json()['history'])
    preview=admin.post('/api/admin/notifications/preview',json={'kind':'daily','robot_id':robot}).json()['content']
    assert preview.startswith('Daily ') and '任务表格' in preview
    settings['task_template']='{project.__class__}'
    assert admin.patch('/api/admin/reminders',json=settings).status_code==400
    settings['task_template']='unclosed {project'
    assert admin.patch('/api/admin/reminders',json=settings).status_code==400


def test_disabled_reminders_and_public_table_access(clients):
    admin,publisher,viewer=clients
    publish(publisher,1)
    assert admin.get('/api/admin/reminders').json()['history']==[]
    assert viewer.get('/api/task-overview').status_code==200
    for route in ('/api/admin/attendance','/api/admin/notifications','/api/admin/reminders'):
        assert viewer.get(route).status_code==403


def test_default_templates_are_complete_and_navigation_is_symmetric(clients):
    admin, publisher, viewer = clients
    settings = admin.get('/api/admin/reminders').json()
    assert '标签：{product_tag}' in settings['task_template']
    assert '状态：{status}' in settings['task_template']
    assert '今日报到：{attendance_count}' in settings['daily_template']
    assert '任务表格\n{task_table}' in settings['daily_template']
    assert settings['defaults']['task_template'] == platform.DEFAULT_TASK_TEMPLATE
    platform.database().save_reminder_config({'task_template':'Custom {project}','daily_template':'Custom {actual_hours}'})
    custom = admin.get('/api/admin/reminders').json()
    assert custom['task_template'] == 'Custom {project}'
    assert custom['daily_template'] == 'Custom {actual_hours}'
    assert custom['defaults']['daily_template'] == platform.DEFAULT_DAILY_TEMPLATE
    platform.database().save_reminder_config({'task_template':'{event}\n{content}','daily_template':'{content}'})
    upgraded = admin.get('/api/admin/reminders').json()
    assert upgraded['task_template'] == platform.DEFAULT_TASK_TEMPLATE
    assert upgraded['daily_template'] == platform.DEFAULT_DAILY_TEMPLATE
    assert 'href="/reminders"' in admin.get('/users').text
    assert 'href="/users"' in admin.get('/reminders').text
