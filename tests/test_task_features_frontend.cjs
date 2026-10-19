const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../workflow_platform/static/task_features.js'), 'utf8');
function frontend(values = {}) {
  const nodes = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, {value}]));
  const requests = [];
  const context = vm.createContext({
    document: {getElementById: key => nodes[key]},
    publishBody: () => ({clipboard_text: 'stale table', part_count: 10, part_manifest: 'stale manifest'}),
    openPublish: () => {}, addParts: () => {}, fields: ['project', 'applicant'],
    esc: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;'),
    api: async (...args) => {requests.push(args); return {total: 0, items: []};},
    statusName: {pending: '待领取'}, URLSearchParams,
  });
  vm.runInContext(source, context);
  return {context, nodes, requests};
}
test('publish modes discard inactive Part inputs and validate integer counts', () => {
  const {context, nodes} = frontend({'pub-mode':'manual', 'pub-parts-mode':'later', 'pub-project':' 项目 ', 'pub-applicant':' 张三 ', 'pub-count':'1.5'});
  let body = context.publishBody();
  assert.equal(body.input_mode, 'manual');
  assert.equal(body.task.project, '项目');
  assert.equal(body.part_count, 0);
  assert.equal(body.part_manifest, '');
  nodes['pub-parts-mode'].value = 'count';
  assert.throws(() => context.publishBody(), /Part 数量/);
  nodes['pub-count'].value = '2';
  assert.equal(context.publishBody().part_count, 2);
  nodes['pub-parts-mode'].value = 'manifest';
  assert.equal(context.publishBody().part_count, 0);
  assert.equal(context.publishBody().part_manifest, 'stale manifest');
});

test('ordinary users see task table without administrator settings requests', async () => {
  const {context,nodes,requests}=frontend({'content':'','task-table-results':''});
  context.me={username:'worker',role:'user'};
  context.api=async(method,url)=>{requests.push(url);return {tasks:[]};};
  await context.openMainView('tasks');
  assert.ok(nodes.content.innerHTML.includes('任务表格'));
  assert.ok(!nodes.content.innerHTML.includes('保存考勤时间'));
  assert.deepEqual(requests,['/api/task-overview']);
});

test('task table includes attendance and both notification forms for admins', async () => {
  const {context,nodes}=frontend({'content':'','task-table-results':'','task-settings':'','notify-daily-robot':''});
  context.me={username:'admin',role:'admin'};
  context.api=async (method,url)=>{
    if(url==='/api/task-overview')return {tasks:[{task_id:'task-1',rank:1,priority:'high',overview_status:'pending',product_tag:'BSD',project:'道路',annotation_content:'车辆',part_summary:{completed:0,total:2},estimated_completion_hours:4,estimated_completion_date:'2026-10-13'}]};
    if(url==='/api/admin/attendance')return {work_start:'09:00',work_end:'18:00'};
    if(url==='/api/admin/notifications')return {enabled:false,robot_id:1,usernames:['worker'],note:'日报备注',latest_run:null};
    if(url==='/api/admin/dingtalk')return {robots:[{robot_id:1,name:'Test'}]};
    if(url==='/api/users')return {users:[{username:'worker',display_name:'标注员',phone:'13800138000',is_active:true}]};
    throw Error(url);
  };
  await context.openMainView('tasks');
  assert.ok(nodes['task-table-results'].innerHTML.includes('排名'));
  assert.ok(nodes['task-settings'].innerHTML.includes('table-work-start'));
  assert.ok(nodes['task-settings'].innerHTML.includes('任务提醒'));
  assert.ok(nodes['task-settings'].innerHTML.includes('下班日报提醒'));
  assert.ok(nodes['task-settings'].innerHTML.includes('notify-daily-enabled'));
  assert.ok(nodes['task-settings'].innerHTML.includes('notify-task-user'));
  assert.ok(nodes['task-settings'].innerHTML.includes('previewTaskNotification(&quot;daily&quot;)'));
});

test('combined settlement view requests seven days and renders both tables', async () => {
  const {context,nodes,requests}=frontend({'content':'','task-table-results':''});
  context.api=async (...args)=>{requests.push(args);return {attendance:{selected_date:'2026-10-13',daily:[{date:'2026-10-13',count:2}],users:[{username:'worker',display_name:'张三',checked_in_at:'08:30'}]},work:{daily:[{date:'2026-10-13',count:1}],parts:[{product_tag:'BSD',project:'道路',username:'worker',expected_hours:2,actual_hours:1}]}};};
  await context.openMainView('attendance');
  assert.ok(requests[0][1].includes('days=7'));
  assert.ok(nodes.content.innerHTML.includes('每日报到人数'));
  assert.ok(nodes.content.innerHTML.includes('每日完成 Part 数'));
  assert.ok(nodes.content.innerHTML.includes('张三'));
  assert.ok(nodes.content.innerHTML.includes('2.00 小时'));
});
test('append modes isolate manifests and reject empty or fractional quantities', () => {
  const {context, nodes} = frontend({'parts-mode':'manifest', 'parts-manifest':'split_1', 'parts-count':'1.5'});
  assert.equal(context.partsBody().count, 0);
  nodes['parts-manifest'].value = '';
  assert.throws(() => context.partsBody(), /清单/);
  nodes['parts-mode'].value = 'count';
  assert.throws(() => context.partsBody(), /Part 数量/);
  nodes['parts-count'].value = '3';
  assert.equal(context.partsBody().count, 3);
  assert.equal(context.partsBody().part_manifest, '');
});
test('overview escapes user text and sends directory filters with pagination', async () => {
  const {context, nodes, requests} = frontend({'overview-status':'pending', 'overview-query':' split_1 ', 'overview-tag':'QA', 'overview-results':'', 'overview-error':'', 'overview-pagination':''});
  const html = context.overviewTable([['name','名称']], [{name:'<img src=x>'}]);
  assert.ok(html.includes('&lt;img src=x>'));
  assert.ok(!html.includes('<img'));
  vm.runInContext('overviewMode="directories"; overviewPage=2;', context);
  await context.renderOverview();
  const url = new URL(requests[0][1], 'http://localhost');
  assert.equal(url.searchParams.get('query'), 'split_1');
  assert.equal(url.searchParams.get('status'), 'pending');
  assert.equal(url.searchParams.get('product_tag'), 'QA');
  assert.equal(url.searchParams.get('page'), '2');
  assert.equal(url.searchParams.get('page_size'), '50');
  assert.ok(nodes['overview-results'].innerHTML.includes('没有符合条件'));
});

test('task tables split completion states and filter both by product tag', async () => {
  const {context,nodes}=frontend({'content':'','task-table-results':''});
  context.me={username:'worker',role:'user'};
  const task=(project,tag,state)=>({project,product_tag:tag,overview_status:state,priority:'high',part_summary:{completed:state==='completed'?1:0,total:1},estimated_completion_hours:4});
  context.api=async()=>({tasks:[task('ActiveBSD','BSD','pending'),task('DoneBSD','BSD','completed'),task('ActiveQA','QA','in_progress'),task('DoneQA','QA','completed')]});
  await context.openMainView('tasks');
  assert.match(nodes.content.innerHTML,/产品标签筛选/);
  let html=nodes['task-table-results'].innerHTML;
  assert.match(html,/未完成任务（2）/);
  assert.match(html,/已完成任务（2）/);
  assert.ok(!html.split('已完成任务')[1].includes('待估算'));
  assert.ok(!html.split('已完成任务')[1].includes('预计完成'));
  context.filterTaskTables('BSD');
  html=nodes['task-table-results'].innerHTML;
  assert.match(html,/ActiveBSD/);assert.match(html,/DoneBSD/);
  assert.ok(!html.includes('ActiveQA'));assert.ok(!html.includes('DoneQA'));
  context.filterTaskTables('');
  assert.match(nodes['task-table-results'].innerHTML,/DoneQA/);
});
test('settlement totals show actual and expected hours including empty dates',()=>{
  const {context,nodes}=frontend({'content':''});
  const data={attendance:{selected_date:'2026-10-14',daily:[],users:[]},work:{daily:[],parts:[{expected_hours:2,actual_hours:1.25},{expected_hours:3,actual_hours:0.75}]}};
  context.renderSettlement(data);
  assert.match(nodes.content.innerHTML,/预计 5.00 小时；实际总耗时 2.00 小时/);
  data.work.parts=[];context.renderSettlement(data);
  assert.match(nodes.content.innerHTML,/预计 0.00 小时；实际总耗时 0.00 小时/);
});
