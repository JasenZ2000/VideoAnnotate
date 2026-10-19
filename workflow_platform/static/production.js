/* Production credit is explicit submission, separate from editable progress. */
(function () {
  const previousUpdateUser = updateUser;
  function reportNavigation() {
    let link = document.getElementById('production-reports-link');
    if (!link) {
      link = document.createElement('a');
      link.id = 'production-reports-link';
      link.className = 'ghost hidden';
      link.href = '/reports';
      link.textContent = '日报与周绩效';
      document.getElementById('admin-stats-btn').after(link);
    }
    link.classList.toggle('hidden', me?.role !== 'admin');
  }
  updateUser = function () { previousUpdateUser(); reportNavigation(); };
  reportNavigation();

  const previousAdminStatsView = adminStatsView;
  adminStatsView = function () {
    previousAdminStatsView();
    const notice = document.createElement('div');
    notice.className = 'hint';
    notice.innerHTML = '此处图片数量按 Part 最后一次提交归属周期。填写日报和周绩效请使用 <a href="/reports">首次提交增量报表</a>。';
    document.querySelector('#content .detail')?.prepend(notice);
  };

  const previousPartCard = partCard;
  partCard = function (part, viewAll = false) {
    const html = previousPartCard(part, viewAll);
    const eligible = part.annotator === me?.username &&
      ['in_progress', 'paused'].includes(part.status) && !part.production_finalized;
    const info = `<div class="note">已首次提交 ${Number(part.credited_count) || 0} 张${part.production_finalized ? ' · 返工不增加完成量' : ' · 编辑进度后，须提交才计入日报'}</div>`;
    const action = eligible ? `<button type="button" class="compact" onclick="submitDailyProgress(${part.part_id})">提交当日进度</button>` : '';
    return html.replace('<div class="part-actions">', info + '<div class="part-actions">' + action);
  };

  let pendingSubmission = null;
  function submissionDialog() {
    let dialog = document.getElementById('part-submission-dialog');
    if (dialog) return dialog;
    dialog = document.createElement('dialog');
    dialog.id = 'part-submission-dialog';
    dialog.setAttribute('aria-labelledby', 'part-submission-title');
    dialog.innerHTML = `<form id="part-submission-form" novalidate>
      <div class="dialog-head"><h2 id="part-submission-title"></h2><button type="button" class="close" aria-label="关闭">×</button></div>
      <div class="dialog-body"><div id="part-submission-hint" class="hint"></div>
        <div id="part-submission-count-field" class="field"><label for="part-submission-count">累计已提交图片数量（不是今天新增量）</label><input id="part-submission-count" type="number" min="0" step="1"></div>
        <div id="part-submission-note-field" class="field"><label for="part-submission-note">提交备注（可选）</label><textarea id="part-submission-note"></textarea></div>
        <div id="part-submission-error" class="error" role="alert"></div></div>
      <div class="dialog-foot"><button type="button" class="ghost">取消</button><button id="part-submission-confirm" type="submit" class="primary">确认提交</button></div>
    </form>`;
    dialog.querySelectorAll('button[type="button"]').forEach(button => button.addEventListener('click', () => dialog.close()));
    dialog.addEventListener('close', () => { pendingSubmission = null; });
    dialog.querySelector('form').addEventListener('submit', async event => {
      event.preventDefault();
      const submission = pendingSubmission;
      const button = document.getElementById('part-submission-confirm');
      if (!submission || button.disabled) return;
      const error = document.getElementById('part-submission-error');
      error.textContent = '';
      let body;
      if (submission.mode === 'progress') {
        const input = document.getElementById('part-submission-count').value;
        const count = Number(input);
        if (!input.trim() || !Number.isSafeInteger(count) || count < 0) {
          error.textContent = '请输入非负整数'; return;
        }
        body = {progress_count: count};
      } else body = {note: document.getElementById('part-submission-note').value};
      button.disabled = true;
      try {
        const result = await api('POST', `/api/tasks/${submission.taskId}/parts/${submission.id}/${submission.mode === 'progress' ? 'submit-progress' : 'submit'}`, body);
        dialog.close();
        toast(submission.mode === 'progress' ? `进度提交成功，累计已计入 ${result.part.credited_count} 张；Part 继续保留当前状态` : '提交完成，已进入待审核');
        await loadTasks();
      } catch (failure) {
        if (dialog.open) error.textContent = failure.message;
        else toast(failure.message);
      } finally { button.disabled = false; }
    });
    document.body.append(dialog);
    return dialog;
  }

  window.openPartSubmission = function (id, mode) {
    const part = (selected?.parts || []).find(item => item.part_id === id);
    if (!part) { toast('未找到 Part，请刷新任务后重试'); return; }
    const dialog = submissionDialog();
    if (dialog.open || document.getElementById('part-submission-confirm').disabled) return;
    pendingSubmission = {id, taskId: selected.task_id, mode};
    const progress = mode === 'progress';
    document.getElementById('part-submission-title').textContent = progress ? '提交当日进度' : '提交完成';
    document.getElementById('part-submission-hint').textContent = progress
      ? `此前已计入 ${part.credited_count || 0} 张，本次只计新增部分。请确认这些图片已经在标注工具中成功保存/提交。`
      : '请确认该 Part 已全部标注并保存。提交后进入待审核，返工再次提交不重复计数。';
    document.getElementById('part-submission-count-field').hidden = !progress;
    document.getElementById('part-submission-note-field').hidden = progress;
    document.getElementById('part-submission-count').value = String(part.progress_count || 0);
    document.getElementById('part-submission-note').value = '';
    document.getElementById('part-submission-error').textContent = '';
    dialog.showModal();
    document.getElementById(progress ? 'part-submission-count' : 'part-submission-note').focus();
  };
  window.submitDailyProgress = id => openPartSubmission(id, 'progress');

  let pendingPartAction = null;
  function partActionDialog() {
    let dialog = document.getElementById('part-action-dialog');
    if (dialog) return dialog;
    dialog = document.createElement('dialog');
    dialog.id = 'part-action-dialog';
    dialog.setAttribute('aria-labelledby', 'part-action-title');
    dialog.innerHTML = `<form id="part-action-form" novalidate>
      <div class="dialog-head"><h2 id="part-action-title"></h2><button type="button" class="close" aria-label="关闭">×</button></div>
      <div class="dialog-body"><div id="part-action-hint" class="hint"></div>
        <div class="field"><label id="part-action-label" for="part-action-note"></label><textarea id="part-action-note"></textarea></div>
        <div id="part-action-error" class="error" role="alert"></div></div>
      <div class="dialog-foot"><button type="button" class="ghost">取消</button><button id="part-action-confirm" type="submit" class="primary">确认</button></div>
    </form>`;
    dialog.querySelectorAll('button[type="button"]').forEach(button => button.addEventListener('click', () => dialog.close()));
    dialog.addEventListener('close', () => { pendingPartAction = null; });
    dialog.querySelector('form').addEventListener('submit', async event => {
      event.preventDefault();
      const action = pendingPartAction;
      const button = document.getElementById('part-action-confirm');
      if (!action || button.disabled) return;
      const note = document.getElementById('part-action-note').value.trim();
      const error = document.getElementById('part-action-error');
      error.textContent = '';
      if (action.kind === 'comment' && !note) {
        error.textContent = '请填写备注内容'; return;
      }
      if (action.kind === 'review' && action.reviewAction === 'rework' && !note) {
        error.textContent = '请填写修改意见'; return;
      }
      button.disabled = true;
      try {
        const endpoint = action.kind === 'comment' ? 'comments' : 'review';
        const body = action.kind === 'comment' ? {content: note} : {action: action.reviewAction, note};
        await api('POST', `/api/tasks/${action.taskId}/parts/${action.id}/${endpoint}`, body);
        dialog.close();
        toast(action.kind === 'comment' ? '备注已添加' : action.reviewAction === 'approve' ? '审核已同意' : '已要求修改');
        await loadTasks();
      } catch (failure) {
        if (dialog.open) error.textContent = failure.message;
        else toast(failure.message);
      } finally { button.disabled = false; }
    });
    document.body.append(dialog);
    return dialog;
  }

  window.openPartAction = function (id, kind, reviewAction = '') {
    const part = (selected?.parts || []).find(item => item.part_id === id);
    if (!part) { toast('未找到 Part，请刷新任务后重试'); return; }
    const dialog = partActionDialog();
    if (dialog.open || document.getElementById('part-action-confirm').disabled) return;
    pendingPartAction = {id, taskId: selected.task_id, kind, reviewAction};
    const comment = kind === 'comment';
    const approve = reviewAction === 'approve';
    document.getElementById('part-action-title').textContent = comment ? '补充备注' : approve ? '审核同意' : '要求修改';
    document.getElementById('part-action-hint').textContent = comment
      ? '备注会追加到该 Part 的记录中。'
      : approve ? '同意后该 Part 将标记为审核通过。' : '请写明需要修改的问题，标注员会在返工时看到。';
    document.getElementById('part-action-label').textContent = comment ? '备注内容' : approve ? '审核意见（可选）' : '修改意见（必填）';
    document.getElementById('part-action-note').value = approve ? '检查通过' : '';
    document.getElementById('part-action-error').textContent = '';
    document.getElementById('part-action-confirm').textContent = comment ? '添加备注' : approve ? '同意' : '要求修改';
    dialog.showModal();
    document.getElementById('part-action-note').focus();
  };
  window.notePart = id => openPartAction(id, 'comment');
  window.reviewPart = (id, action) => openPartAction(id, 'review', action);
})();
