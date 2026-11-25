const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const staticDir = path.resolve(__dirname, '../workflow_platform/static');
const reportSource = fs.readFileSync(path.join(staticDir, 'production_reports.js'), 'utf8');
const progressSource = fs.readFileSync(path.join(staticDir, 'production.js'), 'utf8');

function reportPage() {
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', innerHTML: '', textContent: '', checked: true,
      classList: {toggle() {}}, addEventListener() {}, focus() {}, select() {},
    });
    return nodes.get(id);
  };
  const context = vm.createContext({document: {getElementById: node}, Intl, Date, URLSearchParams,
    navigator: {clipboard: {writeText: async () => {throw new Error('insecure http');}}},
  });
  // Keep page initialization explicit for deterministic request/error tests.
  vm.runInContext(reportSource.replace(/\nloadProduction\(\);\s*$/, '\n'), context);
  return {context, node};
}

test('weekly copy fills only A-D or F and leaves plan and review fields alone', () => {
  const {context} = reportPage();
  const user = {display_name: '李欣', username: 'lixin', image_count: 100,
    project_names: ['车辆标注', '行人清洗']};
  const report = {start: '2026-10-13T00:00:00+08:00', end: '2026-10-20T00:00:00+08:00'};
  assert.equal(context.productionCopyValue(user, 'identity', report, 'week'),
    '李欣\t2026-10-13\t2026-10-19\t车辆标注；行人清洗');
  assert.equal(context.productionCopyValue(user, 'count', report, 'week'), '100');
  assert.equal(context.productionCopyValue(user, 'projects', report, 'day'), '车辆标注；行人清洗');
  user.project_names = ['=SUM(1,2)\tbad\nrow'];
  assert.equal(context.productionCopyValue(user, 'projects', report, 'day'), "'=SUM(1,2) bad row");
});

test('report loading shows incomplete history, escapes task labels, and permits copying over HTTP', async () => {
  const {context, node} = reportPage();
  const data = {start: '2026-10-19T00:00:00+08:00', end: '2026-10-20T00:00:00+08:00',
    tracking_since: '2026-10-19T10:00:00+08:00', legacy_parts_excluded: 2,
    includes_untracked_time: true, workdays_only: false,
    users: [{username: 'worker', display_name: '李欣', image_count: 60,
      project_names: ['<script>'], tasks: [{task_id: 't', name: '<script>', image_count: 60}]}],
    events: [{annotator: 'worker', submitted_at: '2026-10-19T18:00:00+08:00',
      task_name: '<img>', part_name: 'p', progress_count: 60, credited_count: 60, event_kind: 'progress_submission'}],
  };
  context.fetch = async () => ({ok: true, json: async () => data});
  node('period').value = 'day';
  await context.loadProduction();
  assert.match(node('coverage').textContent, /数量可能不完整/);
  assert.match(node('summary').textContent, /新增完成 60 张/);
  assert.ok(node('users').innerHTML.includes('&lt;script&gt;'));
  assert.ok(!node('events').innerHTML.includes('<img>'));
  await context.copyProductionValue(0, 'count');
  assert.equal(node('copy-text').value, '60');
  assert.equal(node('copy-fallback').hidden, false);
  context.fetch = async () => ({ok: false, json: async () => ({detail: '需要管理员权限'})});
  await context.loadProduction();
  assert.equal(node('users').innerHTML, '');
  assert.equal(node('error').textContent, '需要管理员权限');
});

test('submission actions open the page dialog and hide the rework action', () => {
  const calls = [];
  const link = {classList: {toggle() {}}};
  const context = vm.createContext({
    me: {username: 'worker', role: 'user'}, updateUser() {}, adminStatsView() {},
    selected: {task_id: 't', parts: [{part_id: 1, progress_count: 60, credited_count: 40}]},
    partCard: () => '<div class="part-actions"></div>',
    document: {getElementById: () => link},
  });
  context.window = context;
  vm.runInContext(progressSource, context);
  context.openPartSubmission = (...args) => calls.push(args);
  context.submitDailyProgress(1);
  assert.deepEqual(calls, [[1, 'progress']]);
  const part = {part_id: 1, annotator: 'worker', status: 'paused', credited_count: 60};
  assert.ok(context.partCard(part).includes('提交当日进度'));
  part.production_finalized = 1;
  assert.ok(!context.partCard(part).includes('提交当日进度'));
});
