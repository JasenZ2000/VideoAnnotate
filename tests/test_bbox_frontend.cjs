const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'workflow_platform/static/bbox.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'workflow_platform/static/index.html'), 'utf8');

test('bbox frontend accepts zero and clearing, rejects invalid and unsafe counts', async () => {
  const requests = [];
  const context = vm.createContext({
    selected: {task_id: 'task'}, me: {username: 'worker'},
    api: async (...args) => requests.push(args), loadTasks: async () => {}, toast: () => {},
    esc: text => String(text), metric: (value, label) => `${label}:${value}`,
  });
  vm.runInContext(source, context);
  await context.editBboxInline(1, '0');
  await context.editBboxInline(1, '');
  await context.editBboxInline(1, '99');
  for (const value of ['-1', '1.5', 'abc', '9007199254740992']) {
    await context.editBboxInline(1, value);
  }
  assert.deepEqual(requests.map(request => request[2].bbox_count), [0, null, 99]);
  assert.equal(requests[0][1], '/api/tasks/task/parts/1');
  assert.match(context.bboxPartMeta({bbox_count: null, part_id: 1}, false), /按 1 计/);
  assert.match(context.bboxPartMeta({bbox_count: 0, part_id: 1}, false), /BBox 0/);
  assert.doesNotMatch(context.bboxPartMeta({bbox_count: 3, part_id: 1}, false), /<input/);
  assert.match(context.bboxPartMeta({bbox_count: 3, part_id: 1, annotator: 'worker', status: 'submitted'}, true), /<input/);
  assert.doesNotMatch(context.bboxPartMeta({bbox_count: 3, part_id: 1, annotator: 'worker', status: 'completed'}, false), /<input/);
});

test('platform loads bbox UI and all inline scripts parse', () => {
  assert.match(html, /src="\/static\/bbox.js(?:\?[^" ]*)?"/);
  assert.match(html, /bboxMetrics\(total\)/);
  assert.match(html, /bboxMetrics\(u\)/);
  assert.match(html, /bboxMetrics\(x\)/);
  assert.match(html, /bboxPartMeta\(p,canEdit\)/);
  for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) {
    new vm.Script(match[1]);
  }
});
