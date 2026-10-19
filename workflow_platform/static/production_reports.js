'use strict';
let productionData = null;
let productionPeriod = 'day';
let productionRequest = 0;
const byId = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
// Paste as one text cell; prevent task labels from becoming spreadsheet formulas.
function productionCell(value) {
  const text = String(value ?? '').replace(/[\t\r\n]+/g, ' ').trim();
  return /^[=+\-@]/.test(text) ? "'" + text : text;
}
function productionEndDate(end) {
  return new Date(Date.parse(end.slice(0, 10) + 'T00:00:00Z') - 86400000).toISOString().slice(0, 10);
}
function productionCopyValue(user, kind, data, period) {
  if (kind === 'count') return String(user.image_count);
  if (kind === 'projects') return productionCell(user.project_names.join('；'));
  if (kind === 'identity' && period === 'week') {
    return [productionCell(user.display_name || user.username), data.start.slice(0, 10),
      productionEndDate(data.end), productionCell(user.project_names.join('；'))].join('\t');
  }
  throw new Error('不支持的复制内容');
}
async function copyProductionValue(index, kind) {
  try {
    const text = productionCopyValue(productionData.users[index], kind, productionData, productionPeriod);
    if (!text) throw new Error('该人员在此周期没有新增提交项目');
    try {
      await navigator.clipboard.writeText(text);
      byId('copy-fallback').hidden = true;
      byId('copy-status').textContent = '已复制，请在钉钉表格对应单元格粘贴。';
    } catch {
      byId('copy-fallback').hidden = false;
      byId('copy-text').value = text;
      byId('copy-text').focus(); byId('copy-text').select();
      byId('copy-status').textContent = '请按 Ctrl+C 复制下方已选中的内容。';
    }
  } catch (error) { byId('copy-status').textContent = error.message; }
}
function renderProduction() {
  if (!productionData) return;
  const data = productionData;
  const username = byId('person').value;
  const users = data.users.map((user, index) => ({user, index})).filter(({user}) => !username || user.username === username);
  byId('summary').textContent = `${data.start.slice(0, 10)} 至 ${productionEndDate(data.end)} · 新增完成 ${users.reduce((total, {user}) => total + user.image_count, 0)} 张`;
  byId('users').innerHTML = users.map(({user, index}) => `<section><div class="user-head"><h2>${escapeHtml(user.display_name || user.username)} <span class="sub">（${escapeHtml(user.username)}）</span></h2><strong>${user.image_count} 张</strong></div><div class="copy-buttons">${productionPeriod === 'week' ? `<button data-index="${index}" data-kind="identity">复制周报 A—D（姓名、周期、项目）</button>` : ''}<button data-index="${index}" data-kind="projects">复制${productionPeriod === 'day' ? '日报 D' : '周报 D'}：项目</button><button data-index="${index}" data-kind="count">复制 F：实际完成数量</button></div><div class="scroll"><table><thead><tr><th>项目</th><th>新增完成数量</th></tr></thead><tbody>${user.tasks.map(task => `<tr><td>${escapeHtml(task.name)} <span class="sub">${escapeHtml(task.task_id)}</span></td><td>${task.image_count}</td></tr>`).join('') || '<tr><td colspan="2">本周期暂无新增提交</td></tr>'}</tbody></table></div></section>`).join('');
  const kinds = {progress_submission: '分批提交', final_submission: '最终提交'};
  byId('events').innerHTML = data.events.filter(event => !username || event.annotator === username).map(event => `<tr><td>${escapeHtml(event.submitted_at.replace('T', ' '))}</td><td>${escapeHtml(event.annotator)}</td><td>${escapeHtml(event.task_name)} / ${escapeHtml(event.part_name)}</td><td>${event.progress_count}</td><td>${event.credited_count}</td><td>${escapeHtml(kinds[event.event_kind] || event.event_kind)}</td></tr>`).join('') || '<tr><td colspan="6">暂无提交记录</td></tr>';
}
async function loadProduction() {
  const request = ++productionRequest;
  const period = byId('period').value;
  const query = new URLSearchParams({period, date: byId('anchor').value, workdays_only: String(period === 'week' && byId('workdays').checked)});
  byId('refresh').disabled = true;
  byId('error').textContent = '';
  productionData = null;
  byId('users').innerHTML = ''; byId('events').innerHTML = '';
  byId('summary').textContent = ''; byId('coverage').textContent = '正在查询…';
  byId('copy-status').textContent = ''; byId('copy-fallback').hidden = true;
  try {
    const response = await fetch('/api/admin/production-report?' + query, {credentials: 'same-origin'});
    const data = await response.json();
    if (request !== productionRequest) return;
    if (!response.ok) throw new Error(data.detail || '查询失败，请先用管理员账号登录平台');
    productionData = data; productionPeriod = period;
    const previous = byId('person').value;
    byId('person').innerHTML = '<option value="">全部人员</option>' + data.users.map(user => `<option value="${escapeHtml(user.username)}">${escapeHtml(user.display_name || user.username)}（${escapeHtml(user.username)}）</option>`).join('');
    byId('person').value = data.users.some(user => user.username === previous) ? previous : '';
    byId('coverage').classList.toggle('warning', data.includes_untracked_time);
    byId('coverage').textContent = `首次提交记账启用时间：${data.tracking_since.replace('T', ' ')}。旧 Part ${data.legacy_parts_excluded} 个仅用于防止返工重复计数，不计入历史日期。${data.includes_untracked_time ? '所选周期包含启用前的时间，数量可能不完整，不能直接作为完整绩效。' : ''}${data.workdays_only ? ` 已排除周末提交 ${data.excluded_weekend_count} 张。` : ''}`;
    renderProduction();
  } catch (error) {
    if (request === productionRequest) { byId('error').textContent = error.message; byId('coverage').textContent = '未取得报告，请重新查询。'; }
  } finally { if (request === productionRequest) byId('refresh').disabled = false; }
}
byId('anchor').value = new Intl.DateTimeFormat('en-CA', {timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit'}).format(new Date());
byId('refresh').addEventListener('click', loadProduction);
byId('period').addEventListener('change', () => { byId('workdays-label').hidden = byId('period').value !== 'week'; loadProduction(); });
byId('anchor').addEventListener('change', loadProduction);
byId('workdays').addEventListener('change', loadProduction);
byId('person').addEventListener('change', () => { byId('copy-fallback').hidden = true; byId('copy-status').textContent = ''; renderProduction(); });
byId('users').addEventListener('click', event => {
  const button = event.target.closest('button[data-kind]');
  if (button) copyProductionValue(Number(button.dataset.index), button.dataset.kind);
});
loadProduction();
