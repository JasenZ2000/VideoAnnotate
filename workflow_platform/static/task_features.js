let overviewMode='tasks',overviewTasks=[],overviewPage=1,overviewRequest=0;
const overviewStates={awaiting_parts:'待补充 Part',pending:'待领取',in_progress:'执行中',submitted:'待审核',completed:'已完成',mixed:'混合状态'};
function togglePublishMode(){
  const manual=document.getElementById('pub-mode').value==='manual';
  const parts=document.getElementById('pub-parts-mode').value;
  document.getElementById('pub-table-field').classList.toggle('hidden',manual);
  document.getElementById('pub-manual-fields').classList.toggle('hidden',!manual);
  document.getElementById('pub-count').closest('.field').classList.toggle('hidden',parts!=='count');
  document.getElementById('pub-manifest-field').classList.toggle('hidden',parts!=='manifest');
  document.getElementById('pub-preview').classList.add('hidden');
}
const originalPublishBody=publishBody;
publishBody=function(){
  const body=originalPublishBody();
  const manual=document.getElementById('pub-mode').value==='manual';
  const parts=document.getElementById('pub-parts-mode').value;
  body.input_mode=manual?'manual':'table';
  if(manual)body.task=Object.fromEntries(fields.map(f=>[f,document.getElementById('pub-'+f).value.trim()]));
  body.part_count=parts==='count'?Number(document.getElementById('pub-count').value):0;
  if(parts!=='manifest')body.part_manifest='';
  if(parts==='count'&&(!Number.isInteger(body.part_count)||body.part_count<1||body.part_count>10000))throw new Error('Part 数量必须在 1 到 10000 之间');
  if(parts==='manifest'&&!body.part_manifest.trim())throw new Error('请粘贴 Part 工作目录清单');
  return body;
};
const originalOpenPublish=openPublish;
openPublish=function(){
  if(!me){openAuth();return;}
  const grid=document.getElementById('pub-manual-grid');
  if(!grid.childElementCount){
    grid.innerHTML=fields.map((f,i)=>'<div class="field"><label>'+esc(headers[i])+'</label><input id="pub-'+f+'"'+(f==='application_date'?' type="date"':'')+'></div>').join('');
    document.getElementById('pub-application_date').value=localDateValue();
    document.getElementById('pub-applicant').value=me.username;
  }
  originalOpenPublish();togglePublishMode();
};
function localDateValue(){const date=new Date();return date.getFullYear()+'-'+String(date.getMonth()+1).padStart(2,'0')+'-'+String(date.getDate()).padStart(2,'0');}
function setupPartsDialog(){
  if(document.getElementById('parts-dialog'))return;
  const dialog=document.createElement('dialog');dialog.id='parts-dialog';
  dialog.innerHTML='<div class="dialog-head"><h2>补充 / 追加 Part</h2><button class="close" onclick="this.closest(\'dialog\').close()">×</button></div><div class="dialog-body"><div class="hint">追加只生成新的可领取 Part。相对目录使用任务当前的数据根目录，已有 Part 路径保持不变。</div><div class="form-grid"><div class="field"><label>创建方式</label><select id="parts-mode" onchange="togglePartsMode()"><option value="manifest">按目录清单生成</option><option value="count">按数量生成</option></select></div><div id="parts-count-field" class="field hidden"><label>Part 数量</label><input id="parts-count" type="number" min="1" max="10000" value="10"></div><div id="parts-manifest-field" class="field span2"><label>Part 工作目录清单</label><textarea id="parts-manifest" placeholder="每行一个目录，也支持显示名称、目录路径和图片数量的制表符格式"></textarea></div></div><div id="parts-preview" class="hint hidden"></div><div id="parts-error" class="error"></div></div><div class="dialog-foot"><button class="ghost" onclick="previewParts()">预览与检查</button><button class="primary" onclick="saveParts()">生成 Part</button></div>';
  document.body.appendChild(dialog);
}
let partsTaskId='';
function togglePartsMode(){const count=document.getElementById('parts-mode').value==='count';document.getElementById('parts-count-field').classList.toggle('hidden',!count);document.getElementById('parts-manifest-field').classList.toggle('hidden',count);document.getElementById('parts-preview').classList.add('hidden');}
addParts=function(){setupPartsDialog();partsTaskId=selected.task_id;document.getElementById('parts-error').textContent='';document.getElementById('parts-manifest').value='';document.getElementById('parts-preview').classList.add('hidden');togglePartsMode();document.getElementById('parts-dialog').showModal();};
function partsBody(){const manifest=document.getElementById('parts-mode').value==='manifest';const body={count:manifest?0:Number(document.getElementById('parts-count').value),part_manifest:manifest?document.getElementById('parts-manifest').value:''};if(manifest&&!body.part_manifest.trim())throw Error('请粘贴 Part 目录清单');if(!manifest&&(!Number.isInteger(body.count)||body.count<1||body.count>10000))throw Error('Part 数量必须在 1 到 10000 之间');return body;}
async function previewParts(){try{const result=await api('POST','/api/tasks/'+partsTaskId+'/parts/preview',partsBody());const node=document.getElementById('parts-preview');node.classList.remove('hidden');node.innerHTML='将新增 '+result.count+' 个 Part'+previewPaths(result.part_preview);document.getElementById('parts-error').textContent='';}catch(error){document.getElementById('parts-error').textContent=error.message;document.getElementById('parts-preview').classList.add('hidden');}}
function previewPaths(parts){return parts.length?'<div class="table-wrap"><table><thead><tr><th>名称</th><th>完整工作目录</th><th>图片数量</th></tr></thead><tbody>'+parts.map(p=>'<tr><td>'+esc(p.name)+'</td><td>'+esc(p.work_path)+'</td><td>'+Number(p.image_count||0)+'</td></tr>').join('')+'</tbody></table></div>':'';}
async function saveParts(){try{const result=await api('POST','/api/tasks/'+partsTaskId+'/parts',partsBody());document.getElementById('parts-dialog').close();await loadTasks();toast('已新增 '+result.added+' 个可领取 Part');}catch(error){document.getElementById('parts-error').textContent=error.message;}}
async function openOverview(){openMainView("tasks");
  if(!me){openAuth();return;}
  const node=document.getElementById('overview-content');
  node.innerHTML='<div class="form-grid"><div class="field"><label>查看内容</label><select id="overview-mode" onchange="changeOverviewMode()"><option value="tasks">任务汇总</option><option value="directories">Part 目录明细</option></select></div><div class="field"><label>状态</label><select id="overview-status" onchange="applyOverviewFilters()"></select></div><div class="field"><label>任务 / 目录搜索</label><input id="overview-query" placeholder="项目、申请人、任务信息或目录" onkeydown="if(event.key===\'Enter\')applyOverviewFilters()"></div><div class="field"><label>产品标签</label><select id="overview-tag" onchange="applyOverviewFilters()"><option value="">全部</option></select></div></div><div class="part-actions"><button class="ghost" onclick="applyOverviewFilters()">查询 / 刷新</button></div><div id="overview-error" class="error"></div><div id="overview-results" class="table-wrap"></div><div id="overview-pagination" class="part-actions"></div>';
  overviewMode='tasks';overviewPage=1;updateOverviewStates();document.getElementById('overview-dialog').showModal();
  try{overviewTasks=(await api('GET','/api/task-overview')).tasks;document.getElementById('overview-tag').innerHTML='<option value="">全部</option>'+[...new Set(overviewTasks.map(t=>t.product_tag).filter(Boolean))].map(tag=>'<option value="'+esc(tag)+'">'+esc(tag)+'</option>').join('');await renderOverview();}catch(error){document.getElementById('overview-error').textContent=error.message;}
}
function updateOverviewStates(){const states=overviewMode==='tasks'?overviewStates:statusName;document.getElementById('overview-status').innerHTML='<option value="">全部状态</option>'+Object.entries(states).map(([key,value])=>'<option value="'+key+'">'+value+'</option>').join('');}
function changeOverviewMode(){overviewMode=document.getElementById('overview-mode').value;overviewPage=1;updateOverviewStates();renderOverview().catch(error=>document.getElementById('overview-error').textContent=error.message);}
async function applyOverviewFilters(){overviewPage=1;try{if(overviewMode==='tasks')overviewTasks=(await api('GET','/api/task-overview')).tasks;await renderOverview();}catch(error){document.getElementById('overview-error').textContent=error.message;}}
function overviewTable(columns,rows){return '<table><thead><tr>'+columns.map(([,label])=>'<th>'+esc(label)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+columns.map(([field])=>'<td style="white-space:normal;min-width:100px;overflow-wrap:anywhere">'+esc(row[field]??'-')+'</td>').join('')+'</tr>').join('')+(rows.length?'':'<tr><td colspan="'+columns.length+'">没有符合条件的记录</td></tr>')+'</tbody></table>';}
async function renderOverview(){
  const request=++overviewRequest;
  const status=document.getElementById('overview-status').value,query=document.getElementById('overview-query').value.trim(),tag=document.getElementById('overview-tag').value;
  let total=0,html='';
  if(overviewMode==='tasks'){
    const filtered=overviewTasks.filter(t=>(!status||t.overview_status===status)&&(!tag||t.product_tag===tag)&&(!query||Object.values(t).filter(v=>typeof v==='string').join(' ').toLowerCase().includes(query.toLowerCase())));
    total=filtered.length;
    const rows=filtered.slice((overviewPage-1)*50,overviewPage*50).map(t=>({...t,state:overviewStates[t.overview_status],...t.part_summary,percent:t.completion_percent+'%'}));
    html=overviewTable([['state','状态'],['application_date','申请日期'],['applicant','申请人'],['publisher','发布者'],['product_tag','产品标签'],['project','项目'],['annotation_content','标注内容'],['annotation_tool','标注工具'],['dataset_source','数据集溯源'],['hourly_capacity','每小时可标'],['data_amount','数据量'],['estimated_hours','预计工时 / 单人'],['data_path','数据根目录'],['guide_path','说明书路径'],['total','Part 总数'],['pending','待领取'],['in_progress','进行中'],['paused','暂停'],['submitted','待审核'],['rework','退修'],['completed','审核通过'],['percent','通过率']],rows);
  }else{
    const params=new URLSearchParams({status,query,product_tag:tag,page:String(overviewPage),page_size:'50'});
    const data=await api('GET','/api/directory-overview?'+params);
    total=data.total;
    html=overviewTable([['project','项目'],['annotation_content','标注内容'],['product_tag','标签'],['application_date','申请日期'],['applicant','申请人'],['publisher','发布者'],['annotation_tool','标注工具'],['part_name','Part'],['work_path','工作目录'],['state','状态'],['annotator','标注者'],['claimed_at','领取时间'],['submitted_at','提交时间'],['reviewed_at','审核时间'],['guide_path','说明书路径']],data.items.map(p=>({...p,state:statusName[p.status]})));
  }
  if(request!==overviewRequest)return;
  document.getElementById('overview-error').textContent='';document.getElementById('overview-results').innerHTML=html;
  const pages=Math.max(1,Math.ceil(total/50));
  document.getElementById('overview-pagination').innerHTML='<span>共 '+total+' 条 · 第 '+overviewPage+' / '+pages+' 页</span><button class="ghost" '+(overviewPage===1?'disabled':'')+' onclick="moveOverviewPage(-1)">上一页</button><button class="ghost" '+(overviewPage>=pages?'disabled':'')+' onclick="moveOverviewPage(1)">下一页</button>';
}
async function moveOverviewPage(delta){overviewPage+=delta;try{await renderOverview();}catch(error){document.getElementById('overview-error').textContent=error.message;}}
const _updateUserAttendance=typeof updateUser==='function'?updateUser:function(){};
updateUser=function(){_updateUserAttendance();const reminders=document.getElementById('reminders-btn');if(reminders)reminders.classList.toggle('hidden',!(me&&me.role==='admin'));const button=document.getElementById('attendance-btn');if(button)button.classList.toggle('hidden',!me);loadAttendanceStatus().catch(()=>{});};
async function loadAttendanceStatus(){if(!me)return;const status=await api('GET','/api/attendance');const button=document.getElementById('attendance-btn'),count=document.getElementById('attendance-count');button.classList.remove('hidden');button.textContent=status.checked_in?'已报到':'标注员报到';button.disabled=status.checked_in;count.classList.remove('hidden');count.textContent='今日已报到 '+status.checked_in_count+' 人';}
async function checkInAttendance(){try{await api('POST','/api/attendance/check-in',{});await loadAttendanceStatus();toast('今日报到成功')}catch(error){toast(error.message)}}

if(typeof me!=='undefined'&&me)updateUser();

let mainView='tasks';
let taskTableTasks=[],taskTableTag='';
const taskTablePages={unfinished:{page:1,size:5},completed:{page:1,size:5}};
function filterTaskTables(tag){taskTableTag=tag;Object.values(taskTablePages).forEach(state=>state.page=1);renderTaskTables();}
function moveTaskTablePage(kind,delta){taskTablePages[kind].page+=delta;renderTaskTables();}
function changeTaskTablePageSize(kind,value){const size=Number(value);if(![5,10].includes(size))return;taskTablePages[kind].size=size;taskTablePages[kind].page=1;renderTaskTables();}
function paginatedTaskTable(kind,title,columns,rows){
  const state=taskTablePages[kind],pages=Math.max(1,Math.ceil(rows.length/state.size));
  state.page=Math.max(1,Math.min(state.page,pages));
  const start=(state.page-1)*state.size;
  return '<section><h2>'+title+'（'+rows.length+'）</h2><div class="table-wrap">'+overviewTable(columns,rows.slice(start,start+state.size))+'</div><div class="pager"><label>每页显示 <select aria-label="'+title+'每页显示数量" onchange="changeTaskTablePageSize(&quot;'+kind+'&quot;,this.value)">'+[5,10].map(size=>'<option value="'+size+'" '+(state.size===size?'selected':'')+'>'+size+'</option>').join('')+'</select> 条</label><span class="sub">共 '+rows.length+' 条，当前 '+(rows.length?start+1:0)+'-'+Math.min(rows.length,start+state.size)+'</span><button class="ghost" '+(state.page===1?'disabled':'')+' onclick="moveTaskTablePage(&quot;'+kind+'&quot;,-1)">上一页</button><span>第 '+state.page+' / '+pages+' 页</span><button class="ghost" '+(state.page===pages?'disabled':'')+' onclick="moveTaskTablePage(&quot;'+kind+'&quot;,1)">下一页</button></div></section>';
}
function renderTaskTables(){
  const rows=taskTableTasks.filter(task=>!taskTableTag||task.product_tag===taskTableTag).map(task=>({...task,state:overviewStates[task.overview_status],priority_text:typeof priorityName==='undefined'?task.priority:priorityName[task.priority],part_progress:task.part_summary.completed+' / '+task.part_summary.total,hours:Number(task.estimated_completion_hours||0).toFixed(2)+' 小时',completion:task.overview_status==='completed'?'已完成':task.estimated_completion_date||'待估算'}));
  const unfinished=rows.filter(task=>task.overview_status!=='completed'),completed=rows.filter(task=>task.overview_status==='completed');
  const columns=[['rank','排名'],['priority_text','优先级'],['state','状态'],['product_tag','标签'],['project','项目'],['annotation_content','内容'],['part_progress','Part 进度'],['hours','预计工时']];
  document.getElementById('task-table-results').innerHTML=paginatedTaskTable('unfinished','未完成任务',[...columns,['completion','预计完成']],unfinished)+paginatedTaskTable('completed','已完成任务',columns,completed);
}

function barChart(items,valueKey){const max=Math.max(1,...items.map(x=>Number(x[valueKey]||0)));return '<div class="bar-chart">'+items.map(x=>'<div class="bar-item"><div class="bar" style="height:'+Math.max(6,Number(x[valueKey]||0)/max*100)+'%"><b>'+esc(x[valueKey]??0)+'</b></div><small>'+esc(x.date.slice(5))+'</small></div>').join('')+'</div>'}
async function openMainView(view){
  mainView=view;selected=null;document.getElementById('content').innerHTML='<div class="detail"><div class="hint">正在加载...</div></div>';
  try{
    if(view==='tasks'){
      const data=await api('GET','/api/task-overview');
      taskTableTasks=data.tasks;
      const tags=[...new Set(taskTableTasks.map(task=>task.product_tag).filter(Boolean))].sort();
      if(!tags.includes(taskTableTag))taskTableTag='';
      document.getElementById('content').innerHTML='<div class="detail"><h1>任务表格</h1><p class="sub">未完成任务按预计完成日期升序排列，同一天按排名升序；排期跳过周六和周日，已完成任务不参与排期。</p><div class="field" style="max-width:280px"><label>产品标签筛选</label><select id="task-table-tag" onchange="filterTaskTables(this.value)"><option value="">全部标签</option>'+tags.map(tag=>'<option value="'+esc(tag)+'" '+(tag===taskTableTag?'selected':'')+'>'+esc(tag)+'</option>').join('')+'</select></div><div id="task-table-results"></div><div id="task-settings"></div></div>';
      renderTaskTables();
      if(typeof me!=='undefined'&&me?.role==='admin')await loadTaskSettings(data);
    }else{renderSettlement(await api('GET','/api/attendance/settlements?days=7'));}
  }catch(error){document.getElementById('content').innerHTML='<div class="detail"><div class="error">'+esc(error.message)+'</div></div>';}
}
async function loadTaskSettings(overview){
  if(typeof me==='undefined'||me?.role!=='admin')return;
  const [attendance,config,robots,people]=await Promise.all([api('GET','/api/admin/attendance'),api('GET','/api/admin/notifications'),api('GET','/api/admin/dingtalk'),api('GET','/api/users')]);
  if(mainView!=='tasks'||selected)return;
  const options=robots.robots.map(robot=>'<option value="'+robot.robot_id+'">'+esc(robot.name)+'</option>').join('');
  const recipients=people.users.filter(user=>user.is_active&&user.phone.trim());
  const recipientsHtml=kind=>recipients.map(user=>'<label style="display:inline-flex;gap:5px;margin:6px 12px 6px 0"><input type="checkbox" name="notify-'+kind+'-user" value="'+esc(user.username)+'" '+(kind==='daily'&&config.usernames.includes(user.username)?'checked':'')+'>'+esc(user.display_name||user.username)+'（'+esc(user.username)+'）</label>').join('')||'<p class="sub">请先在用户管理中填写手机号。</p>';
  const notificationForm=kind=>'<section><h2>'+(kind==='task'?'任务提醒':'下班日报提醒')+'</h2><div class="form-grid"><div class="field"><label>机器人</label><select id="notify-'+kind+'-robot">'+options+'</select></div>'+(kind==='task'?'<div class="field"><label>选择任务</label><select id="notify-task-id">'+overview.tasks.map(task=>'<option value="'+esc(task.task_id)+'">'+esc(task.product_tag+' · '+task.project+' · '+task.annotation_content)+'</option>').join('')+'</select></div>':'<div class="hint">日报仅列出按当前排序的前五项未完成任务，并附当天报到人数、完成 Part 数及预计和实际完成工时。</div>')+'<div class="field span2"><label>要 @ 的用户（可多选）</label>'+recipientsHtml(kind)+'</div><div class="field span2"><label>补充备注</label><textarea id="notify-'+kind+'-note" style="min-height:70px">'+esc(kind==='daily'?config.note:'')+'</textarea></div></div>'+(kind==='daily'?'<p><label><input id="notify-daily-enabled" type="checkbox" '+(config.enabled?'checked':'')+'> 开启自动下班日报</label></p><p class="sub">服务运行期间，周一至周五到下班时间后发送。每个日期自动尝试一次，失败时可手动补发。</p><button class="ghost" onclick="saveAutomaticReport()">保存自动通知设置</button><p class="sub" id="notify-latest">'+esc(config.latest_run?'最近自动日报：'+config.latest_run.work_date+' · '+config.latest_run.status+' · '+config.latest_run.detail:'尚无自动发送记录')+'</p>':'')+'<div class="part-actions"><button class="ghost" onclick="previewTaskNotification(&quot;'+kind+'&quot;)">预览通知</button><button class="primary" id="notify-'+kind+'-send" onclick="sendTaskNotification(&quot;'+kind+'&quot;)">手动发送</button></div><pre id="notify-'+kind+'-preview" class="hint hidden" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre><div id="notify-'+kind+'-error" class="error"></div></section>';
  document.getElementById('task-settings').innerHTML='<section><h2>考勤设置</h2><div class="form-grid"><div class="field"><label>上班时间</label><input id="table-work-start" type="time" value="'+esc(attendance.work_start)+'"></div><div class="field"><label>下班时间</label><input id="table-work-end" type="time" value="'+esc(attendance.work_end)+'"></div></div><div class="part-actions"><button class="primary" onclick="saveTableAttendance()">保存考勤时间</button></div><div id="table-attendance-error" class="error"></div></section><section><h2>通知设置</h2><p class="sub">机器人和用户手机号可在<a href="/users">用户管理</a>中设置。</p>'+notificationForm('task')+notificationForm('daily')+'</section>';
  if(config.robot_id)document.getElementById('notify-daily-robot').value=String(config.robot_id);
}
async function saveTableAttendance(){try{await api('PATCH','/api/admin/attendance',{work_start:document.getElementById('table-work-start').value,work_end:document.getElementById('table-work-end').value});document.getElementById('table-attendance-error').textContent='';toast('考勤时间已保存')}catch(error){document.getElementById('table-attendance-error').textContent=error.message}}
function taskNotificationBody(kind){return {kind,robot_id:Number(document.getElementById('notify-'+kind+'-robot').value),task_id:kind==='task'?document.getElementById('notify-task-id').value:'',usernames:[...document.querySelectorAll('[name="notify-'+kind+'-user"]:checked')].map(node=>node.value),note:document.getElementById('notify-'+kind+'-note').value};}
async function previewTaskNotification(kind){try{const result=await api('POST','/api/admin/notifications/preview',taskNotificationBody(kind));const preview=document.getElementById('notify-'+kind+'-preview');preview.textContent=result.content;preview.classList.remove('hidden');document.getElementById('notify-'+kind+'-error').textContent=''}catch(error){document.getElementById('notify-'+kind+'-error').textContent=error.message}}
async function sendTaskNotification(kind){const button=document.getElementById('notify-'+kind+'-send');button.disabled=true;try{await api('POST','/api/admin/notifications/send',taskNotificationBody(kind));document.getElementById('notify-'+kind+'-error').textContent='';toast('通知已发送')}catch(error){document.getElementById('notify-'+kind+'-error').textContent=error.message}finally{button.disabled=false}}
async function saveAutomaticReport(){try{const body=taskNotificationBody('daily');await api('PATCH','/api/admin/notifications',{enabled:document.getElementById('notify-daily-enabled').checked,robot_id:body.robot_id||null,usernames:body.usernames,note:body.note});document.getElementById('notify-daily-error').textContent='';toast('自动日报设置已保存')}catch(error){document.getElementById('notify-daily-error').textContent=error.message}}
function renderSettlement(data){document.getElementById('content').innerHTML=`<div class="detail"><h1>考勤与工时结算</h1><div class="field" style="max-width:240px"><label>选择日期（时间轴为截至该日的七天）</label><input type="date" value="${esc(data.attendance.selected_date)}" onchange="loadSettlementDate(this.value)"></div><h2>每日报到人数</h2>${barChart(data.attendance.daily,'count')}${attendanceDetail(data.attendance.users)}<h2>每日完成 Part 数</h2><p class="sub">每 Part 预计工时 = 任务总预计工时 ÷ Part 总数；实际耗时使用累计计时或管理员修正值。</p>${barChart(data.work.daily,'count')}<div class="hint">所选日期完成 Part 工时合计：预计 ${Number(data.work.total_expected_hours??data.work.parts.reduce((total,part)=>total+Number(part.expected_hours||0),0)).toFixed(2)} 小时；实际总耗时 ${Number(data.work.total_actual_hours??data.work.parts.reduce((total,part)=>total+Number(part.actual_hours||0),0)).toFixed(2)} 小时</div>${workDetail(data.work.parts)}</div>`;}
function attendanceDetail(rows){return '<div class="table-wrap"><table><thead><tr><th>用户名</th><th>显示名</th><th>报到时间</th></tr></thead><tbody>'+rows.map(x=>'<tr><td>'+esc(x.username)+'</td><td>'+esc(x.display_name||'-')+'</td><td>'+esc(x.checked_in_at)+'</td></tr>').join('')+(rows.length?'':'<tr><td colspan="3">当天没有报到记录</td></tr>')+'</tbody></table></div>'}
function workDetail(rows){return '<div class="table-wrap"><table><thead><tr><th>标签</th><th>项目</th><th>完成者</th><th>预计工时</th><th>实际耗时</th></tr></thead><tbody>'+rows.map(x=>'<tr><td>'+esc(x.product_tag)+'</td><td>'+esc(x.project||x.annotation_content)+'</td><td>'+esc(x.display_name||x.username||'-')+'</td><td>'+Number(x.expected_hours||0).toFixed(2)+' 小时</td><td>'+Number(x.actual_hours||0).toFixed(2)+' 小时</td></tr>').join('')+(rows.length?'':'<tr><td colspan="5">当天没有完成 Part</td></tr>')+'</tbody></table></div>'}
async function loadSettlementDate(date){try{const data=await api("GET","/api/attendance/settlements?days=7&date="+encodeURIComponent(date));renderSettlement(data)}catch(e){toast(e.message)}}
