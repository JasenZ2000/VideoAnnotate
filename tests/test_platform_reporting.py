import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from workflow_platform.database import PlatformDatabase
from workflow_platform.reporting import (
    daily_report, generate_daily_report, notifications, read_notifications, task_estimate,
)
import workflow_platform.server as server


NOW = '2026-10-13T18:00:00+08:00'


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = PlatformDatabase(Path(self.temp.name) / 'test.sqlite3')
        self.db.initialize()
        for name in ('owner', 'viewer', 'worker', 'other'):
            self.db.create_user(name, 'hash', 'admin' if name == 'other' else 'user', name, NOW)
        self.task = {'task_id': 't', 'name': '项目 · 标注', 'publisher': 'owner', 'manager': 'viewer',
                     'status': 'ready', 'data_amount': '1,000', 'hourly_capacity': '150-200',
                     'created_at': NOW, 'updated_at': NOW}
        self.db.create_task(self.task, 2, NOW)

    def tearDown(self):
        self.temp.cleanup()

    def test_recipients_lifecycle_and_atomic_rollback(self):
        self.assertEqual(notifications(self.db, 'viewer')['unread_count'], 1)
        self.assertEqual(notifications(self.db, 'owner')['unread_count'], 0)
        part = self.db.claim_next_part('t', 'worker', NOW)
        self.db.pause_part('t', part['part_id'], 'worker', NOW)
        before = notifications(self.db, 'viewer')['unread_count']
        with self.assertRaises(PermissionError):
            self.db.pause_part('t', part['part_id'], 'worker', NOW)
        self.assertEqual(notifications(self.db, 'viewer')['unread_count'], before)
        self.db.resume_part('t', part['part_id'], 'worker', NOW)
        self.db.submit_part('t', part['part_id'], 'worker', '', NOW)
        self.db.review_part('t', part['part_id'], 'viewer', 'approve', '', NOW)
        second = self.db.claim_next_part('t', 'worker', NOW)
        self.db.submit_part('t', second['part_id'], 'worker', '', NOW)
        self.db.review_part('t', second['part_id'], 'viewer', 'approve', '', NOW)
        self.assertIn('任务完成', notifications(self.db, 'owner')['notifications'][0]['message'])
        self.assertEqual(notifications(self.db, 'worker')['unread_count'], 0)
        count = notifications(self.db, 'owner')['unread_count']
        with self.assertRaises(RuntimeError):
            with self.db.transaction() as c:
                c.execute("UPDATE tasks SET priority='urgent' WHERE task_id='t'")
                raise RuntimeError('rollback')
        self.assertEqual(notifications(self.db, 'owner')['unread_count'], count)

    def test_priority_dedup_and_read_isolation(self):
        self.db.update_task('t', 'owner', {'manager': 'owner'}, NOW)
        self.db.update_task_ordering('t', 'other', rank=None, priority='urgent', now=NOW, is_admin=True)
        owner = notifications(self.db, 'owner')
        self.assertEqual(sum('优先级变动' in n['message'] for n in owner['notifications']), 1)
        read_notifications(self.db, 'viewer', owner['notifications'][0]['notification_id'], NOW)
        self.assertEqual(notifications(self.db, 'owner')['unread_count'], owner['unread_count'])
        read_notifications(self.db, 'owner', 0, NOW)
        self.assertEqual(notifications(self.db, 'owner')['unread_count'], 0)

    def test_snapshot_timezone_idempotency_and_visibility(self):
        self.assertFalse(generate_daily_report(self.db, datetime.fromisoformat('2026-10-13T09:59:59+00:00')))
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: generate_daily_report(self.db, datetime.fromisoformat(NOW)), range(2)))
        self.assertEqual(sum(outcomes), 1)
        report = daily_report(self.db, {'username': 'owner', 'role': 'user'})
        self.assertEqual(report['remaining_count'], 1)
        self.assertEqual(report['tasks'][0]['remaining_hours'], [5, 6.67])
        self.assertEqual(daily_report(self.db, {'username': 'worker', 'role': 'user'})['remaining_count'], 0)
        with self.db.transaction() as c:
            c.execute("UPDATE tasks SET deleted=1 WHERE task_id='t'")
        self.assertEqual(daily_report(self.db, {'username': 'owner', 'role': 'user'}), report)
        self.db.initialize()
        self.assertFalse(generate_daily_report(self.db, datetime.fromisoformat(NOW)))
        self.assertTrue(generate_daily_report(self.db, datetime.fromisoformat('2026-10-14T19:00:00+08:00')))

    def test_estimates_do_not_double_count_and_preserve_uncertainty(self):
        parts = [{'image_count': 500, 'progress_count': 500, 'status': 'submitted'},
                 {'image_count': 500, 'progress_count': 100, 'status': 'paused'}]
        estimate = task_estimate(self.task, parts)
        self.assertEqual(estimate['remaining_amount'], 400)
        self.assertEqual(estimate['remaining_hours'], [2, 2.67])
        self.assertEqual(estimate['pending_review_parts'], 1)
        parts[0]['image_count'] = 0
        self.assertIsNone(task_estimate(self.task, parts)['remaining_hours'])
        parts[0]['status'] = 'pending'
        parts[0]['progress_count'] = 50
        self.assertEqual(task_estimate(self.task, parts)['remaining_amount'], 850)
        self.assertIsNone(task_estimate({**self.task, 'data_amount': ''}, parts)['remaining_amount'])
        self.assertIsNone(task_estimate({**self.task, 'hourly_capacity': '0'}, parts)['remaining_hours'])
        parts[0]['status'] = parts[1]['status'] = 'submitted'
        self.assertEqual(task_estimate(self.task, parts)['remaining_amount'], 0)


class RegistrationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = dict(server.SETTINGS)
        server.SETTINGS['database_path'] = str(Path(self.temp.name) / 'api.sqlite3')
        server.SETTINGS['data_dir'] = Path(self.temp.name)
        self.client = TestClient(server.app)
        self.client.post('/api/auth/bootstrap-admin', json={'username': 'owner', 'password': 'owner-password'})
        self.client.post('/api/users', json={'username': 'viewer', 'password': 'viewer-password'})
        self.body = {'product_tag': 'BSD', 'part_count': 1, 'manager': 'viewer', 'annotation_tool': '工具',
                     'registration': {'application_date': '2026-10-13', 'applicant': '申请人', 'project': '项目',
                                      'annotation_content': '标注', 'hourly_capacity': '150-200', 'data_amount': '1000',
                                      'data_path': 'D:/data', 'desired_completion_date': '2026-10-20'}}

    def tearDown(self):
        self.client.close()
        server.SETTINGS.clear()
        server.SETTINGS.update(self.old)
        self.temp.cleanup()

    def test_online_registration_saved_notified_and_private(self):
        preview = self.client.post('/api/tasks/preview', json=self.body)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()['rows'][0]['estimated_hours'], '5.00-6.67')
        response = self.client.post('/api/tasks', json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        task_id = response.json()['tasks'][0]['task_id']
        task = self.client.get('/api/tasks/' + task_id).json()
        self.assertEqual(task['manager'], 'viewer')
        self.assertEqual(task['annotation_tool'], '工具')
        self.assertEqual(task['desired_completion_date'], '2026-10-20')
        viewer = TestClient(server.app)
        try:
            viewer.post('/api/auth/login', json={'username': 'viewer', 'password': 'viewer-password'})
            inbox = viewer.get('/api/notifications').json()
            self.assertEqual(inbox['unread_count'], 1)
            notification_id = inbox['notifications'][0]['notification_id']
            self.client.post(f'/api/notifications/{notification_id}/read')
            self.assertEqual(viewer.get('/api/notifications').json()['unread_count'], 1)
            viewer.post(f'/api/notifications/{notification_id}/read')
            self.assertEqual(viewer.get('/api/notifications').json()['unread_count'], 0)
        finally:
            viewer.close()

    def test_invalid_registration_and_collaborator_rejected(self):
        for field, value in [('data_amount', 'abc'), ('hourly_capacity', '0'), ('project', ''), ('application_date', 'yesterday')]:
            body = {**self.body, 'registration': {**self.body['registration'], field: value}}
            self.assertEqual(self.client.post('/api/tasks', json=body).status_code, 400)
        self.assertEqual(self.client.post('/api/tasks', json={**self.body, 'manager': 'missing'}).status_code, 400)
        self.assertEqual(self.client.get('/api/daily-reports?date=bad').status_code, 400)
        self.assertEqual(self.client.get('/api/tasks').json()['tasks'], [])

    def test_reordered_excel_headers_include_extended_fields(self):
        rows = server.parse_spreadsheet_rows('项目\t希望完成时间\t数据路径\t每小时可标\t标注内容\n测试\t2026-10-20\tD:/data\t150-200\t标注')
        self.assertEqual(rows[0]['desired_completion_date'], '2026-10-20')
        self.assertEqual(rows[0]['data_path'], 'D:/data')

    def test_scheduler_runs_and_stops_with_lifespan(self):
        from unittest.mock import patch
        from threading import Event
        called = Event()
        with patch.object(server, 'generate_daily_report', side_effect=lambda db: called.set()):
            with TestClient(server.app):
                self.assertTrue(called.wait(5))


if __name__ == '__main__':
    unittest.main()
