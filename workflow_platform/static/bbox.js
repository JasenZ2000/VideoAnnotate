// BBox display and editing helpers shared by task details and statistics.
function bboxMetrics(value) {
  const count = Number(value?.bbox_count ?? 0);
  const rate = Number(value?.bboxes_per_hour ?? 0);
  return metric(Number.isFinite(count) ? count : 0, 'BBox 总数')
    + metric(Number.isFinite(rate) ? rate.toFixed(1) : '0.0', '框/小时');
}

function bboxPartMeta(part, canEdit) {
  const value = part.bbox_count === null || part.bbox_count === undefined
    ? '' : String(part.bbox_count);
  if (!canEdit) {
    return `<span class="bbox-meta">BBox ${value === '' ? '未填写（按 1 计）' : value}</span>`;
  }
  return `<label class="bbox-meta">BBox <input class="part-count-input" type="number" min="0" step="1" value="${esc(value)}" placeholder="未填写" aria-label="BBox 总数" title="留空按 1 计入" onchange="editBboxInline(${part.part_id},this.value)"></label>`;
}

async function editBboxInline(partId, value) {
  const text = String(value ?? '').trim();
  const count = text === '' ? null : Number(text);
  if (count !== null && (!Number.isSafeInteger(count) || count < 0)) {
    toast('BBox 总数必须是非负整数；留空按 1 计入');
    return;
  }
  try {
    await api('PATCH', `/api/tasks/${selected.task_id}/parts/${partId}`, {bbox_count: count});
    await loadTasks();
    toast(count === null ? '已恢复默认 BBox 计数' : 'BBox 总数已保存');
  } catch (error) {
    toast(error.message);
  }
}
