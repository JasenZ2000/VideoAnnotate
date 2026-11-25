"""Durable site notifications and daily snapshots (China time)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

CHINA_TIME = timezone(timedelta(hours=8))


def initialize_reporting(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS notifications (
            notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
            task_id TEXT NOT NULL DEFAULT '',
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            read_at TEXT,
            report_date TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_notifications_user
            ON notifications(username,notification_id DESC);
        CREATE TABLE IF NOT EXISTS daily_reports (
            report_date TEXT PRIMARY KEY,
            generated_at TEXT NOT NULL,
            snapshot TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS notify_task_created AFTER INSERT ON tasks
        BEGIN
            INSERT INTO notifications(username,task_id,message,created_at)
            SELECT username,NEW.task_id,'新任务提交：' || NEW.name,NEW.created_at
            FROM users WHERE is_active=1 AND username=NEW.manager;
        END;
        CREATE TRIGGER IF NOT EXISTS notify_task_priority AFTER UPDATE OF priority ON tasks
        WHEN OLD.priority != NEW.priority AND NEW.deleted=0
        BEGIN
            INSERT INTO notifications(username,task_id,message,created_at)
            SELECT username,NEW.task_id,'任务优先级变动：' || NEW.name || '（' || OLD.priority || ' → ' || NEW.priority || '）',NEW.updated_at
            FROM users WHERE is_active=1 AND username IN (NEW.publisher,NEW.manager);
        END;
        CREATE TRIGGER IF NOT EXISTS notify_task_completed AFTER UPDATE OF status ON tasks
        WHEN OLD.status != NEW.status AND NEW.status='completed' AND NEW.deleted=0
        BEGIN
            INSERT INTO notifications(username,task_id,message,created_at)
            SELECT username,NEW.task_id,'任务完成（全部审核通过）：' || NEW.name,NEW.updated_at
            FROM users WHERE is_active=1 AND username IN (NEW.publisher,NEW.manager);
        END;
        CREATE TRIGGER IF NOT EXISTS notify_task_manager AFTER UPDATE OF manager ON tasks
        WHEN OLD.manager != NEW.manager AND NEW.deleted=0
        BEGIN
            INSERT INTO notifications(username,task_id,message,created_at)
            SELECT username,NEW.task_id,'协同查看人变更：' || NEW.name,NEW.updated_at
            FROM users WHERE is_active=1 AND username IN (NEW.publisher,NEW.manager,OLD.manager);
        END;
        CREATE TRIGGER IF NOT EXISTS notify_part_state AFTER UPDATE OF status ON parts
        WHEN OLD.status != NEW.status
        BEGIN
            INSERT INTO notifications(username,task_id,message,created_at)
            SELECT u.username,t.task_id,t.name || ' / ' || NEW.name || '：' ||
                CASE NEW.status WHEN 'in_progress' THEN '开始/恢复标注'
                WHEN 'paused' THEN '暂停' WHEN 'submitted' THEN '标注完成，待审核'
                WHEN 'completed' THEN '审核通过' WHEN 'rework' THEN '退修'
                WHEN 'pending' THEN '退回待领取' ELSE NEW.status END ||
                '（标注员：' || COALESCE(NEW.annotator,OLD.annotator,'') || '）',NEW.updated_at
            FROM tasks t JOIN users u ON u.username IN (t.publisher,t.manager)
            WHERE t.task_id=NEW.task_id AND t.deleted=0 AND u.is_active=1;
        END;
    """)


def numeric_range(value):
    text = str(value).strip().replace(',', '').replace('，', '')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:[-–—~～]\s*(\d+(?:\.\d+)?))?', text)
    if not match:
        return None
    low = float(match[1])
    high = float(match[2] or match[1])
    return (low, high) if 0 < low <= high else None


def task_estimate(task, parts):
    """Count each part once; estimates exclude review time, preserve uncertainty."""
    amount = numeric_range(task['data_amount'])
    known_total = sum(max(0, p['image_count']) for p in parts)
    all_measured = bool(parts) and all(p['image_count'] > 0 for p in parts)
    total = amount[0] if amount and amount[0] == amount[1] else known_total if all_measured else 0
    done = sum(p['image_count'] if p['status'] in ('submitted', 'completed')
               else min(max(0, p['progress_count']), p['image_count']) if p['image_count'] > 0
               else max(0, p['progress_count']) for p in parts)
    complete = bool(parts) and all(p['status'] in ('submitted', 'completed') for p in parts)
    remaining = 0 if complete else max(0, total - done) if total > 0 else None
    # Unknown-size submitted parts or inconsistent totals cannot support a reliable estimate.
    uncertain = (any(p['image_count'] <= 0 and p['status'] in ('submitted', 'completed') for p in parts)
                 or (all_measured and amount and total != known_total)
                 or (total > 0 and total < done) or (total > 0 and known_total > total)
                 or any(p['status'] == 'rework' for p in parts)) and not complete
    capacity = numeric_range(task['hourly_capacity'])
    hours = ([round(remaining / capacity[1], 2), round(remaining / capacity[0], 2)]
             if remaining is not None and capacity and not uncertain else None)
    return {'total_amount': total or None, 'annotated_amount': done,
            'remaining_amount': None if uncertain else remaining, 'remaining_hours': hours,
            'pending_review_parts': sum(p['status'] == 'submitted' for p in parts),
            'estimate_note': '单人工时，不含审核；按剩余标注量 ÷ 计划每小时可标量计算'
            if hours is not None else '数据量、Part进度、产能不完整/不一致或待退修，无法可靠估算'}


def generate_daily_report(db, now=None):
    current = (now or datetime.now(CHINA_TIME)).astimezone(CHINA_TIME)
    if current.hour < 18:
        return False
    day = current.date().isoformat()
    timestamp = current.isoformat(timespec='seconds')
    with db.transaction() as connection:
        if connection.execute('SELECT 1 FROM daily_reports WHERE report_date=?', (day,)).fetchone():
            return False
        rows = []
        for row in connection.execute("SELECT * FROM tasks WHERE deleted=0 AND status!='completed' ORDER BY rank ASC"):
            task = dict(row)
            parts = [dict(p) for p in connection.execute('SELECT * FROM parts WHERE task_id=?', (task['task_id'],))]
            rows.append({**task, **task_estimate(task, parts)})
        connection.execute('INSERT INTO daily_reports VALUES(?,?,?)',
                           (day, timestamp, json.dumps(rows, ensure_ascii=False)))
        for user in connection.execute('SELECT username,role FROM users WHERE is_active=1'):
            visible = [r for r in rows if user['role'] == 'admin' or user['username'] in (r['publisher'], r['manager'])]
            # Send a zero-remaining report as well, so task owners see completion.
            owner = connection.execute('SELECT 1 FROM tasks WHERE deleted=0 AND (publisher=? OR manager=?) LIMIT 1',
                                       (user['username'], user['username'])).fetchone()
            if visible or owner or user['role'] == 'admin':
                connection.execute('INSERT INTO notifications(username,message,created_at,report_date) VALUES(?,?,?,?)',
                                   (user['username'], f'{day} 下班任务日报：剩余 {len(visible)} 个任务（含待审核）', timestamp, day))
    return True


def notifications(db, username):
    with db.connection() as connection:
        items = [dict(r) for r in connection.execute('SELECT * FROM notifications WHERE username=? ORDER BY notification_id DESC LIMIT 200', (username,))]
        unread = connection.execute('SELECT COUNT(*) FROM notifications WHERE username=? AND read_at IS NULL', (username,)).fetchone()[0]
    return {'notifications': items, 'unread_count': unread}


def read_notifications(db, username, notification_id, now):
    with db.transaction() as connection:
        connection.execute('UPDATE notifications SET read_at=? WHERE username=? AND read_at IS NULL AND (?=0 OR notification_id=?)',
                           (now, username, notification_id, notification_id))


def daily_report(db, user, day=''):
    with db.connection() as connection:
        row = connection.execute('SELECT * FROM daily_reports WHERE report_date=?', (day,)).fetchone() if day else connection.execute('SELECT * FROM daily_reports ORDER BY report_date DESC LIMIT 1').fetchone()
    if not row:
        return {'report_date': day, 'tasks': [], 'remaining_count': 0, 'available': False}
    tasks = [t for t in json.loads(row['snapshot']) if user['role'] == 'admin' or user['username'] in (t['publisher'], t['manager'])]
    return {'report_date': row['report_date'], 'generated_at': row['generated_at'], 'tasks': tasks,
            'remaining_count': len(tasks), 'available': True}
