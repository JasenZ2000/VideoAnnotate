const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
test('user management renders users and allows multiple phone recipients', async () => {
  const nodes = {};
  const node = id => nodes[id] ||= {value:'',innerHTML:'',textContent:'',addEventListener(){}};
  const requests = [];
  const context = vm.createContext({
    document:{getElementById:node,querySelectorAll:()=>[]},
    fetch:async url=>{requests.push(url);return {ok:true,json:async()=>url==='/api/auth/me'?{user:{username:'admin',role:'admin'}}:url==='/api/users'?{users:[{username:'admin',display_name:'管理员',phone:'13800138000',role:'admin',is_active:true},{username:'worker',display_name:'标注员',phone:'13900139000',role:'user',is_active:true}]}:{robots:[{robot_id:1,name:'通知机器人'}]}};},
  });
  vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../workflow_platform/static/users.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(nodes.users.innerHTML.includes('管理员'));
  assert.ok(nodes.users.innerHTML.includes('data-action="save"'));
  assert.ok(nodes.users.innerHTML.includes('data-action="delete"'));
  assert.equal((nodes['recipient-options'].innerHTML.match(/type="checkbox"/g)||[]).length,2);
  assert.ok(nodes['recipient-options'].innerHTML.includes('13900139000'));
  assert.ok(!requests.some(url=>url.includes('/recipients')));
});
