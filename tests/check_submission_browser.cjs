const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH || chromium.executablePath()});
  try {
    const platformUrl = process.env.PLATFORM_URL || 'http://127.0.0.1:8000/';
    const target = new URL(platformUrl);
    const page = await browser.newPage({ignoreHTTPSErrors: target.protocol === 'https:' && ['127.0.0.1', 'localhost'].includes(target.hostname)});
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    // Exercise the served UI without changing real users, Parts, or performance data.
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/auth/me') return route.fulfill({json: {user: {username: 'worker', role: 'user'}, bootstrap_required: false}});
      return route.fulfill({json: {tasks: [], items: [], users: []}});
    });
    await page.goto(platformUrl);
    await page.waitForFunction(() => typeof window.submitDailyProgress === 'function');
    await page.evaluate(async () => {
      await openPublish();
      const modes = document.querySelectorAll('#pub-mode');
      if (modes.length !== 1) throw new Error('Duplicate publish mode controls');
      document.getElementById('pub-mode').value = 'manual';
      document.getElementById('pub-parts-mode').value = 'later';
      togglePublishMode();
      document.getElementById('pub-project').value = 'Integration project';
      document.getElementById('pub-applicant').value = 'worker';
      const body = publishBody();
      if (body.input_mode !== 'manual' || body.task.project !== 'Integration project' || body.part_count !== 0 || body.manager !== '') throw new Error('Publish wrappers conflict');
      document.getElementById('publish-dialog').close();
      if (!document.getElementById('inbox-button')) throw new Error('Inbox entry is missing');
      me = {username: 'admin', role: 'admin'};
      updateUser();
      const reportLink = document.getElementById('production-reports-link');
      if (!reportLink || reportLink.classList.contains('hidden')) throw new Error('Production report entry is missing');
    });
    await page.evaluate(() => {
      window.prompt = () => {throw new Error('Native prompt must not be used');};
      const part = {part_id: 1, name: '测试 Part', annotator: 'worker', status: 'in_progress', progress_count: 60, credited_count: 40, image_count: 100};
      me = {username: 'worker', role: 'user'};
      selected = {task_id: 'test-task', parts: [part]};
      document.getElementById('content').innerHTML = partCard(part);
      api = async (method, url, body) => {
        window.submissionRequests.push({method, url, body});
        if (window.rejectSubmission) throw new Error('测试提交失败');
        return {part: {credited_count: body.progress_count}};
      };
      window.submissionRequests = [];
      loadTasks = async () => {};
    });
    const dialog = page.locator('#part-submission-dialog');
    await page.getByRole('button', {name: '提交当日进度', exact: true}).click();
    await dialog.waitFor({state: 'visible'});
    assert.equal(await page.locator('#part-submission-count').inputValue(), '60');
    await dialog.getByRole('button', {name: '取消', exact: true}).click();
    assert.equal(await page.evaluate(() => submissionRequests.length), 0);
    await page.getByRole('button', {name: '提交当日进度', exact: true}).click();
    await page.locator('#part-submission-count').fill('1.5');
    await dialog.getByRole('button', {name: '确认提交'}).click();
    await page.getByRole('alert').filter({hasText: '请输入非负整数'}).waitFor();
    assert.equal(await page.evaluate(() => submissionRequests.length), 0);
    await page.locator('#part-submission-count').fill('60');
    await dialog.getByRole('button', {name: '确认提交'}).click();
    await dialog.waitFor({state: 'hidden'});
    requests.push(...await page.evaluate(() => submissionRequests));
    assert.equal(requests[0].url, '/api/tasks/test-task/parts/1/submit-progress');
    assert.deepEqual(requests[0].body, {progress_count: 60});
    await page.getByRole('button', {name: '提交完成', exact: true}).click();
    await page.locator('#part-submission-note').fill('已完成标注');
    await page.evaluate(() => {window.rejectSubmission = true;});
    await dialog.getByRole('button', {name: '确认提交'}).click();
    await page.getByRole('alert').filter({hasText: '测试提交失败'}).waitFor();
    assert.equal(await dialog.isVisible(), true);
    assert.equal(await page.locator('#part-submission-confirm').isEnabled(), true);
    await page.evaluate(() => {window.rejectSubmission = false;});
    await dialog.getByRole('button', {name: '确认提交'}).click();
    await dialog.waitFor({state: 'hidden'});
    const final = await page.evaluate(() => submissionRequests.at(-1));
    assert.equal(final.url, '/api/tasks/test-task/parts/1/submit');
    assert.deepEqual(final.body, {note: '已完成标注'});
    await page.evaluate(() => {
      submissionRequests.length = 0;
      const part = {part_id: 2, name: '待审核 Part', annotator: 'worker', status: 'submitted', progress_count: 100, credited_count: 100, image_count: 100};
      selected = {task_id: 'review-task', parts: [part], can_review: true};
      document.getElementById('content').innerHTML = partCard(part);
    });
    const actionDialog = page.locator('#part-action-dialog');
    await page.getByRole('button', {name: '补充备注', exact: true}).click();
    await page.locator('#part-action-note').fill('复核记录');
    await actionDialog.getByRole('button', {name: '添加备注'}).click();
    await actionDialog.waitFor({state: 'hidden'});
    let actionRequest = await page.evaluate(() => submissionRequests.at(-1));
    assert.equal(actionRequest.url, '/api/tasks/review-task/parts/2/comments');
    assert.deepEqual(actionRequest.body, {content: '复核记录'});
    await page.getByRole('button', {name: '要求修改', exact: true}).click();
    await actionDialog.getByRole('button', {name: '要求修改'}).click();
    await page.getByRole('alert').filter({hasText: '请填写修改意见'}).waitFor();
    await page.locator('#part-action-note').fill('边框需要调整');
    await actionDialog.getByRole('button', {name: '要求修改'}).click();
    await actionDialog.waitFor({state: 'hidden'});
    actionRequest = await page.evaluate(() => submissionRequests.at(-1));
    assert.equal(actionRequest.url, '/api/tasks/review-task/parts/2/review');
    assert.deepEqual(actionRequest.body, {action: 'rework', note: '边框需要调整'});
    await page.evaluate(() => {
      const part = selected.parts[0];
      document.getElementById('content').innerHTML = partCard(part);
    });
    await page.getByRole('button', {name: '同意', exact: true}).click();
    assert.equal(await page.locator('#part-action-note').inputValue(), '检查通过');
    await actionDialog.getByRole('button', {name: '同意'}).click();
    await actionDialog.waitFor({state: 'hidden'});
    actionRequest = await page.evaluate(() => submissionRequests.at(-1));
    assert.deepEqual(actionRequest.body, {action: 'approve', note: '检查通过'});
    assert.deepEqual(errors, []);
    console.log('Submission and review browser checks passed: progress, completion, notes, approve, rework, validation, retry, no native prompt or page errors.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
