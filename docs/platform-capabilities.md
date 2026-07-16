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

发布时额外设置产品大标签、Part 数量和 Part 前缀。粘贴多行时，每行发布成一个独立任务。

## 权限

- 所有登录用户都能查看任务并领取待领取 Part；发布者也可以领取自己发布的任务。
- 标注者只能看到自己的 Part、备注和计时，以及任务整体数量进度。
- 只有任务发布者能编辑或删除任务、追加 Part、查看所有人的综合统计。
- 任务发布者和管理员都能查看任务的完整 Part 明细，并对待审核 Part 执行同意或退回修改。
- 发布者可以删除任意状态的自有任务，包括进行中、待审核和已完成的任务。
- 编辑任务时可更新产品标签、Part 前缀和任务表格中的十项信息；新前缀只用于之后追加的 Part。
- 管理员负责平台账号管理和 Part 审核；管理员不会自动获得其他发布者任务的个人统计或任务编辑权限。

## Part 状态

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> in_progress: 领取并开始计时
    in_progress --> submitted: 提交备注并停止计时
    submitted --> completed: 发布者或管理员同意
    submitted --> rework: 发布者或管理员填写修改意见
    rework --> in_progress: 原标注者开始修改并重新计时
```

每次提交、审核和补充备注均保留历史。累计工时是首次标注与各次返工计时之和。

## 数据关系

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : login
    TASKS ||--o{ PARTS : contains
    PARTS ||--o{ PART_WORK_SESSIONS : times
    PARTS ||--o{ PART_COMMENTS : records
```
