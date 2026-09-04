# 多人标注协作平台

`workflow_platform` 只负责人员、任务、Part、计时、备注和审核协作，不上传或处理视频，不调用 GPU Services，也不生成标注文件。

## 任务发布

发布者从 Excel/WPS 直接复制任务表的一行或多行。平台识别以下十列，支持同时粘贴表头：

1. 申请日期
2. 申请人
3. 项目
4. 标注内容
5. 数据集溯源
6. 每小时可标
7. 数据量
8. 预计工时/单人
9. 数据路径
10. 标注说明书路径

发布时额外设置产品大标签，并选择按数量生成 Part，或粘贴扫描脚本生成的 Part 工作目录清单。按数量生成时继续使用 Part 数量和前缀；使用目录清单时，每行目录直接绑定一个 Part。为避免同一清单套到不同数据根目录，目录清单模式一次只发布一行任务。

## Part 工作目录清单

推荐普通使用者直接双击 `PartDirectoryScannerTool.exe`：点击“选择根目录”，调整最大深度或识别标志，扫描并检查结果后点击“复制清单”，再粘贴到平台发布窗口。无需打开 PowerShell。

开发或部署人员可用以下命令构建：

```powershell
pip install -r requirements\part-scanner-windows.txt
.\scripts\windows\build-part-directory-scanner.ps1
```

生成文件为 `dist\PartDirectoryScannerTool.exe`。下面的 PowerShell 扫描脚本保留为高级或自动化用法：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\scan-part-directories.ps1 `
  -Root D:\dataset -MaxDepth 4 -CopyToClipboard
```

脚本把扫描根目录记为深度 `0`，最多检查到 `MaxDepth`。当某个目录的直接子目录中出现 `images`、`labels` 或 `annotations` 时，该目录会被认定为一个工作目录，并停止继续向下扫描，因此不会把这些数据子目录误当作 Part。可通过 `-MarkerDirectories` 修改标志目录，通过 `-MinimumMarkerCount 2` 要求至少命中两个标志以减少误判。

默认清单是相对于扫描根目录的路径：

```text
split_001
camera_a/split_002
```

平台根据任务表中的“数据路径”拼出完整工作路径。也可以手工使用两列格式 `显示名称<Tab>目录路径`，或直接粘贴完整 UNC、Windows、Linux 路径。标注者领取后会在 Part 卡片中看到并可复制实际工作目录。

## 权限

- 所有登录用户都能查看任务并领取待领取 Part；发布者也可以领取自己发布的任务。
- 标注者只能看到自己的 Part、备注和计时，以及任务整体数量进度。
- 只有任务发布者能编辑或删除任务、追加 Part；发布者可以为任务指定一名协同审核人，该用户能查看全部 Part 明细和人员综合统计，并审核待审核 Part，但不能编辑或删除任务、追加 Part，也不能确认耗时设置是否合理。
- 任务发布者可以删除任意状态的 Part；未领取 Part 会移出领取队列，进行中或已暂停 Part 会连同当前计时会话一起删除。
- 任务发布者、指定的协同审核人和管理员都能查看任务的完整 Part 明细，并对待审核 Part 执行同意或退回修改。
- 发布者可以删除任意状态的自有任务，包括进行中、待审核和已完成的任务。
- 编辑任务时可更新产品标签、Part 前缀和任务表格中的十项信息；新前缀只用于之后追加的 Part。
- 管理员负责平台账号管理和 Part 审核，可以创建或删除用户、修改用户资料并重置密码。不能删除当前登录账号或最后一个启用的管理员；删除用户时，该用户发布的任务转交给执行操作的管理员，正在标注、暂停或返工的 Part 自动退回待领取，已提交和已完成记录保留原用户名。管理员不会自动获得其他发布者任务的个人统计或任务编辑权限。
- 管理员可以在任务列表修改数字排名和低、中、高、加急四级优先级；未完成任务按名次从小到大展示，排名 1 最高、数字越小越靠前。移动一个任务时，其余任务会自动前移或后移，刷新列表也会自动修复缺号和重复号，始终保持从 1 开始连续排名。已完成任务不参与排名并始终置于列表底部。排名联动变化、优先级和 Part 删除操作均记录操作人、调整前后值和时间。
- 管理员可打开“人员统计”，按日、周或月查看每个用户参与的标注任务、完成 Part 数量、标注图像数量和耗时，并查看所选周期的总体完成图片数与总体耗时。完成 Part 和图片数量只归入最后一次提交所在周期，审核退回后的复修不会重复计数；跨周期的计时按时间范围截取并保留全部实际复修耗时。

## Part 状态

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> in_progress: 领取并开始计时
    in_progress --> paused: 暂停计时
    paused --> in_progress: 继续计时
    in_progress --> pending: 中途退还
    paused --> pending: 中途退还
    in_progress --> submitted: 提交备注并停止计时
    submitted --> completed: 发布者、协同审核人或管理员同意
    submitted --> rework: 发布者、协同审核人或管理员填写修改意见
    rework --> in_progress: 原标注者开始修改并重新计时
```

每次暂停、继续、退还、提交、审核和补充备注均保留历史。累计工时是首次标注与各次返工计时之和。左侧任务卡的进度条表示标注完成度，按“待审核 + 已通过”的 Part 数量计算，因此提交后立即增长，不需要等待集中审核；已通过数量单独显示。发布者还可以在任务创建后填写每个 Part 的预计耗时；实际耗时与预估相差达到 50% 时，平台显示偏差，并允许发布者确认预估是否合理。

## 数据关系

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : login
    TASKS ||--o{ PARTS : contains
    PARTS ||--o{ PART_WORK_SESSIONS : times
    PARTS ||--o{ PART_COMMENTS : records
```
