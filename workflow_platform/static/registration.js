// Registration, durable inbox and daily snapshots share the existing authenticated API.
headers.push('希望完成时间','预计开始日期','预计完成日期','确认人','标签路径');
fields.push('desired_completion_date','planned_start_date','planned_completion_date','confirmer','label_path');
const bar = document.querySelector('.userbar');
const featureButtons = document.createElement('div');
featureButtons.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap';
featureButtons.innerHTML = '<button class="ghost" onclick="openRegistrationTable()">登记表</button><button class="ghost" onclick="openDailyReport()">下班日报</button><button id="inbox-button" class="ghost" onclick="openInbox()">消息</button>';
bar.prepend(featureButtons);
const featureDialog = document.createElement('dialog');
featureDialog.id = 'feature-dialog';
featureDialog.style.width = 'min(1300px,calc(100vw - 30px))';
featureDialog.innerHTML = '<div class="dialog-head"><h2 id="feature-title"></h2><button class="close" onclick="this.closest(\'dialog\').close()">×</button></div><div id="feature-body" class="dialog-body"></div>';
document.body.appendChild(featureDialog);
let featureView = '';
function showFeature(title, html, view) {
  featureView = view;
  document.getElementById('feature-title').textContent = title;
  document.getElementById('feature-body').innerHTML = html;
  if (!featureDialog.open) featureDialog.showModal();
}
const publishGrid = document.querySelector('#publish-dialog .form-grid');
const registrationForm = document.createElement('div');
registrationForm.className = 'span2';
registrationForm.innerHTML = '<div class="field"><label>提交方式</label><select id="pub-mode" onchange="changePublishMode()"><option value="form">填写任务登记表</option><option value="paste">从 Excel/WPS 粘贴</option></select></div><div id="registration-fields" class="form-grid" style="margin-top:13px">' + fields.map((f,i) => `<div class="field"><label>${esc(headers[i])}${['application_date','applicant','project','annotation_content','hourly_capacity','data_amount','data_path'].includes(f)?' *':''}</label><input id="registration-${f}" ${f==='application_date'?'type="date"':''} ${f==='estimated_hours'?'readonly placeholder="根据数据量和产能自动计算"':''} ${f==='hourly_capacity'?'placeholder="例如 150 或 150-200"':''}></div>`).join('') + '</div>';
publishGrid.prepend(registrationForm);
const collaborator = document.createElement('div');
collaborator.className = 'field';
collaborator.innerHTML = '<label>协同查看人 *</label><select id="pub-manager"><option value="">请选择平台账号</option></select>';
publishGrid.insertBefore(collaborator, document.getElementById('pub-table').parentElement);
function changePublishMode(){
  const paste = document.getElementById('pub-mode').value === 'paste';
  document.getElementById('registration-fields').classList.toggle('hidden', paste);
  document.getElementById('pub-table').parentElement.classList.toggle('hidden', !paste);
  document.getElementById('pub-preview').classList.add('hidden');
}
function formNumber(s){return /^\d+(\.\d+)?$/.test(s)?Number(s):NaN}
function updateEstimatedHours(){
  const amount = formNumber(document.getElementById('registration-data_amount').value.replace(/[,，]/g,''));
  const rates = document.getElementById('registration-hourly_capacity').value.trim().split(/[-–—~～]/).map(s=>formNumber(s.trim()));
  const low = rates[0], high = rates[1] ?? low;
  document.getElementById('registration-estimated_hours').value = amount>0 && low>0 && high>=low && rates.length<=2 ? (low===high?(amount/low).toFixed(2):`${(amount/high).toFixed(2)}-${(amount/low).toFixed(2)}`) : '';
}
['data_amount','hourly_capacity'].forEach(f=>document.getElementById('registration-'+f).addEventListener('input',updateEstimatedHours));
const originalOpenPublish = openPublish;
openPublish = async function(){
  if (!me){openAuth();return}
  try {
    const options = await api('GET','/api/user-options');
    const select = document.getElementById('pub-manager');
    const chosen = select.value;
    select.innerHTML = '<option value="">请选择平台账号</option>' + options.users.map(u=>`<option value="${esc(u.username)}">${esc(u.display_name||u.username)} (${esc(u.username)})</option>`).join('');
    select.value = chosen;
    document.getElementById('registration-application_date').value ||= new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai'}).format(new Date());
    document.getElementById('registration-applicant').value ||= me.display_name||me.username;
    changePublishMode();
    originalOpenPublish();
  } catch(e){toast(e.message)}
};
const originalPublishBody = publishBody;
publishBody = function(){
  const body = originalPublishBody();
  body.manager = document.getElementById('pub-manager').value;
  if (!body.manager) throw new Error('请选择协同查看人');
  if(document.getElementById('pub-mode').value==='form'){
    body.clipboard_text = '';
    body.registration = Object.fromEntries(fields.map(f=>[f,document.getElementById('registration-'+f).value]));
  }
  return body;
};
function registrationTable(rows){
  return '<div class="table-wrap"><table style="min-width:1700px"><thead><tr>'+['产品',...headers,'提交人','协同查看人','状态','优先级','标注进度','操作'].map(h=>`<th>${esc(h)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(t=>'<tr>'+[t.product_tag,...fields.map(f=>t[f]),t.publisher,t.manager,t.status==='ready'?'待开始':t.status==='completed'?'已完成':'进行中',priorityName[t.priority],`${t.part_summary?.annotated||0}/${t.part_summary?.total||0} Part 已提交/通过`].map(v=>`<td style="white-space:pre-wrap;max-width:220px;overflow-wrap:anywhere">${esc(v??'')}</td>`).join('')+`<td><button class="ghost" data-task="${esc(t.task_id)}">查看任务</button></td></tr>`).join('')+'</tbody></table></div>';
}
function wireTaskLinks(){document.querySelectorAll('#feature-body [data-task]').forEach(b=>b.addEventListener('click',async()=>{featureDialog.close();try{await openTask(b.dataset.task)}catch(e){toast(e.message)}}))}
async function openRegistrationTable(){
  if(!me){openAuth();return}
  try{const r=await api('GET','/api/tasks');showFeature('任务登记表',`<p class="sub">按登记表字段展示任务；进度为已提交及已审核通过的 Part 数量。</p>${registrationTable(r.tasks)}`,'table');wireTaskLinks()}catch(e){toast(e.message)}
}
async function refreshInboxBadge(){
  featureButtons.classList.toggle('hidden',!me);
  if(!me)return;
  const r=await api('GET','/api/notifications');
  document.getElementById('inbox-button').textContent=`消息${r.unread_count?' ('+r.unread_count+')':''}`;
  return r;
}
async function openInbox(){
  if(!me){openAuth();return}
  try{
    const r=await refreshInboxBadge();
    showFeature('消息中心',`<button id="read-all" class="ghost">全部标为已读</button><p class="sub">未读 ${r.unread_count} 条，显示最近 200 条。消息保存在平台内。</p>` + (r.notifications.length?r.notifications.map(n=>`<div class="audit-item" style="margin-bottom:10px"><div>${n.read_at?'':'● '}${esc(n.message)}</div><div class="sub">${esc(n.created_at)}</div><button class="ghost" data-notification="${n.notification_id}">${n.report_date?'查看日报':n.task_id?'查看任务':'标为已读'}</button></div>`).join(''):'<p>暂无消息</p>'),'inbox');
    document.getElementById('read-all').onclick=async()=>{try{await api('POST','/api/notifications/0/read');await openInbox()}catch(e){toast(e.message)}};
    document.querySelectorAll('[data-notification]').forEach(b=>b.onclick=async()=>{try{const n=r.notifications.find(n=>String(n.notification_id)===b.dataset.notification);await api('POST',`/api/notifications/${n.notification_id}/read`);await refreshInboxBadge();if(n.report_date)await openDailyReport(n.report_date);else if(n.task_id){featureDialog.close();await openTask(n.task_id)}else await openInbox()}catch(e){toast(e.message)}});
  }catch(e){toast(e.message)}
}
async function openDailyReport(day=''){
  if(!me){openAuth();return}
  try{
    const r=await api('GET','/api/daily-reports'+(day?'?date='+encodeURIComponent(day):''));
    const content = !r.available?'<p>该日期暂无日报。服务运行时每天北京时间 18:00 自动统计，18:00 后重启会补生成当天日报。</p>':`<p>日报日期：${esc(r.report_date)}，生成时间：${esc(r.generated_at)}。剩余 <b>${r.remaining_count}</b> 个任务（含待审核）。${me.role==='admin'?'管理员查看全平台':'仅显示你提交或协同查看的任务'}。</p><p class="sub">预计剩余时间为单人工时，不含审核与等待；150–200 的产能显示工时范围。未知或不一致的数量不强行估算。日报保存生成时快照。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>优先级</th><th>剩余标注量</th><th>计划每小时标注量</th><th>预计剩余工时</th><th>待审核 Part</th></tr></thead><tbody>`+r.tasks.map(t=>`<tr><td><button class="ghost" data-task="${esc(t.task_id)}">${esc(t.name)}</button></td><td>${esc(priorityName[t.priority])}</td><td>${t.remaining_amount??'待完善数据'}</td><td>${esc(t.hourly_capacity)}</td><td title="${esc(t.estimate_note)}">${t.remaining_hours?(t.remaining_hours[0]===t.remaining_hours[1]?t.remaining_hours[0]:t.remaining_hours.join('–'))+' 小时':'无法可靠估算'}</td><td>${t.pending_review_parts}</td></tr>`).join('')+'</tbody></table></div>';
    showFeature('18:00 下班任务日报',`<div class="admin-stat-tools"><input id="report-day" type="date" value="${esc(r.report_date||day)}"><button id="report-query" class="ghost">查询</button></div>${content}`,'report');
    document.getElementById('report-query').onclick=()=>openDailyReport(document.getElementById('report-day').value);
    wireTaskLinks();
  }catch(e){toast(e.message)}
}
setInterval(()=>refreshInboxBadge().catch(()=>{}),15000);
refreshInboxBadge().catch(()=>{});
// Login completes asynchronously; refresh the badge whenever authentication changes.
const originalUpdateUser = updateUser;
updateUser = function(){originalUpdateUser();refreshInboxBadge().catch(()=>{})};
