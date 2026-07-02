# Task Operating Procedure

## 1. Create The Task

Enter a clear task name, assignee, notes and class table. Keep class IDs stable for the whole task:

```text
0 person
1 car
2 bicycle
```

The same table is sent to multi-class LocateAnything jobs and used when exporting YOLO labels. Changing IDs after prelabels have been generated can make old labels inconsistent.

## 2. Upload Source Material

A task may contain multiple complete videos. Upload each original video without manually splitting it first. If an employee model already produced YOLO labels, upload one ZIP whose TXT filenames follow the source frame numbering.

Select the intended video in the task UI before starting video-specific operations.

## 3. Split Long Videos

Choose segment length in frames based on video duration, target density and expected review effort. The platform writes segment videos and remaps matching full-video YOLO files to segment-local frame numbers.

Do not split again after downstream work has started unless the old segment outputs have been deliberately archived. Segment identity is part of the review contract.

## 4. Choose A Prelabel Source

- **Uploaded YOLO**: use labels produced by an employee model.
- **LocateAnything**: run prompt-based inference on all segments. Multi-class tasks automatically use detection categories and map returned labels to task class IDs.
- **None**: start manual work without prelabels.

LocateAnything runs sequentially per segment because a model instance normally occupies most of one GPU. The platform polls the remote job instead of holding a long HTTP request open.

## 5. Build Initial Tracks

Run segment tracking after labels are available. The default pipeline performs forward/backward tracking and tracklet fusion. Inspect logs and generated overview media before assigning the task for review.

## 6. Review Locally

The employee downloads the package, opens its workspace in local Annotator, cleans IDs and boxes, and uses SAM3.1 only where a bbox-prompt continuation saves time. SAM3.1 credentials are supplied on the employee PC through the configured environment variable or SSH key.

## 7. Return And Export

Upload reviewed `tracking_results.json`, then export YOLO. Preserve the task record after completion for traceability; use soft delete only when it should disappear from the normal task list.

## Current MVP Caveat

The platform's full-video compatibility paths for package creation, reviewed upload and final export predate the segment hierarchy. Segment LocateAnything and segment tracking are available, but complete segment package/download/review/merge orchestration remains on the roadmap. Until that is implemented, verify package paths manually for segmented production work.
