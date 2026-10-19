const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const {test}=require('node:test');
test('reminder page renders four rules and task/daily templates',async()=>{
  const nodes={};const node=id=>nodes[id]||=( {value:'',innerHTML:'',textContent:'',addEventListener(){}} );
  const rules=Object.fromEntries(['published','priority','started','completed'].map(key=>[key,{enabled:false,robot_id:null,usernames:[],notify_publisher:false,notify_applicant:false}]));
  const context=vm.createContext({document:{getElementById:node},fetch:async url=>({ok:true,json:async()=>url==='/api/auth/me'?{user:{role:'admin'}}:url==='/api/admin/reminders'?{rules,task_template:'{event}: {project}',daily_template:'{content}',history:[]}:url==='/api/admin/dingtalk'?{robots:[{robot_id:1,name:'Robot'}]}:{users:[{username:'worker',phone:'13800138000',display_name:'Worker',is_active:true}]}})});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../workflow_platform/static/reminders.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal((nodes.rules.innerHTML.match(/data-event=/g)||[]).length,4);
  assert.ok(nodes.rules.innerHTML.includes('同时提醒任务提交人'));
  assert.ok(nodes.rules.innerHTML.includes('同时提醒申请人'));
  assert.ok(nodes.rules.innerHTML.includes('Worker'));
  assert.equal(nodes['task-template'].value,'{event}: {project}');
  assert.equal(nodes['daily-template'].value,'{content}');
});
