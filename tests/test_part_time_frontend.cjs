const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'workflow_platform/static/part_time.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'workflow_platform/static/index.html'), 'utf8');

test('part time frontend saves, clears, and validates minutes', async () => {
  const requests = [];
  const messages = [];
  const context = vm.createContext({
    selected: {task_id: 'task'},
    api: async (...args) => requests.push(args),
    loadTasks: async () => {},
    toast: message => messages.push(message),
    fmtTime: seconds => String(seconds),
    Number,
  });
  vm.runInContext(source, context);
  await context.editPartWorkSecondsInline(1, '12.5');
  await context.editPartWorkSecondsInline(1, '0');
  await context.editPartWorkSecondsInline(1, '');
  await context.editPartWorkSecondsInline(1, '-1');
  await context.editPartWorkSecondsInline(1, '44641');
  assert.deepEqual(requests.map(request => request[2].work_seconds), [750, 0, null]);
  assert.equal(requests[0][1], '/api/tasks/task/parts/1');
  assert.match(context.partWorkTimeMeta({part_id: 1, work_seconds: 3600, work_seconds_override: null}, true), /placeholder="60"/);
  assert.match(context.partWorkTimeMeta({part_id: 1, work_seconds: 120, work_seconds_override: 120}, false), /手动耗时/);
  assert.equal(messages.length, 5);
});

test('platform loads part time UI', () => {
  assert.ok(html.includes('src="/static/part_time.js"'));
  assert.ok(html.includes('partWorkTimeMeta(p,canEditTime)'));
  assert.match(html, /me?.role==='admin'/);
});
