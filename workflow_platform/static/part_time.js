function partWorkTimeMeta(part, canEdit) {
  const overridden = part.work_seconds_override !== null && part.work_seconds_override !== undefined;
  const label = overridden ? '手动耗时' : '计时耗时';
  if (!canEdit) return `<span>${label} ${fmtTime(part.work_seconds || 0)}</span>`;
  const minutes = overridden ? Math.round(Number(part.work_seconds_override) / 6) / 10 : '';
  const placeholder = Math.round(Number(part.work_seconds || 0) / 6) / 10;
  return `<span>${label}</span><input class="part-count-input" type="number" min="0" max="44640" step="0.1" value="${minutes}" placeholder="${placeholder}" aria-label="手动完成耗时（分钟）" title="分钟；留空恢复原始计时" onchange="editPartWorkSecondsInline(${part.part_id},this.value)">`;
}

async function editPartWorkSecondsInline(id, value) {
  const text = String(value).trim();
  const minutes = text === '' ? null : Number(text);
  if (minutes !== null && (!Number.isFinite(minutes) || minutes < 0 || minutes > 44640)) {
    toast('完成耗时必须在 0 到 44640 分钟之间；留空恢复原始计时');
    return;
  }
  try {
    await api('PATCH', `/api/tasks/${selected.task_id}/parts/${id}`, {
      work_seconds: minutes === null ? null : minutes * 60,
    });
    await loadTasks();
    toast(minutes === null ? '已恢复原始计时' : `完成耗时已修正为 ${minutes} 分钟`);
  } catch (error) {
    toast(error.message);
  }
}
