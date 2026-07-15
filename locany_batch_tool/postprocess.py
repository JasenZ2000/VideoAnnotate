from __future__ import annotations

import filecmp
import shutil
import uuid
from pathlib import Path
from typing import Any

from locany_batch_tool.server import VIDEO_EXTENSIONS


def _annotation_directories(root: Path) -> dict[str, Path]:
    return {path.name.casefold(): path for path in root.iterdir() if path.is_dir()}


def _merge_preflight(source: Path, target: Path) -> list[str]:
    conflicts: list[str] = []
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        target_file = target / relative
        if target_file.exists() and (
            not target_file.is_file() or not filecmp.cmp(source_file, target_file, shallow=False)
        ):
            conflicts.append(str(relative))
    return conflicts


def _merge_directories(source: Path, target: Path) -> None:
    for source_file in sorted(source.rglob("*")):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if target_file.exists():
            source_file.unlink()
        else:
            shutil.move(str(source_file), str(target_file))
    for directory in sorted((path for path in source.rglob("*") if path.is_dir()), reverse=True):
        directory.rmdir()
    source.rmdir()


def organize_prelabels(video_dir: str, prelabel_dir: str, *, dry_run: bool = True) -> dict[str, Any]:
    videos_root = Path(video_dir).expanduser().resolve()
    prelabels_root = Path(prelabel_dir).expanduser().resolve()
    if not videos_root.is_dir():
        raise ValueError(f"视频目录不存在: {videos_root}")
    if not prelabels_root.is_dir():
        raise ValueError(f"预标注目录不存在: {prelabels_root}")

    videos = sorted(
        path for path in videos_root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    directories = _annotation_directories(prelabels_root)
    items: list[dict[str, Any]] = []

    stems: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for video in videos:
        key = video.stem.casefold()
        if key in stems:
            duplicate_stems.add(key)
        else:
            stems[key] = video

    for key, video in stems.items():
        item: dict[str, Any] = {"video": str(video), "name": video.name, "status": "ready", "actions": []}
        if key in duplicate_stems:
            item.update(status="error", error=f"存在多个同名视频文件: {video.stem}")
            items.append(item)
            continue
        workspace = directories.get(key)
        if workspace is None:
            item.update(status="skipped", error=f"没有找到同名预标注目录: {video.stem}")
            items.append(item)
            continue

        destination_video = workspace / video.name
        labels = workspace / "labels"
        renamed_labels = workspace / video.stem

        if destination_video.exists():
            if not destination_video.is_file() or destination_video.stat().st_size != video.stat().st_size:
                item.update(status="error", error=f"目标视频已存在但内容大小不同: {destination_video}")
                items.append(item)
                continue
            item["actions"].append("视频已存在，跳过复制")
        else:
            item["actions"].append(f"复制视频到 {destination_video}")

        if labels.is_dir() and not renamed_labels.exists():
            item["actions"].append(f"重命名 labels 为 {video.stem}")
            label_action = "rename"
        elif not labels.exists() and renamed_labels.is_dir():
            item["actions"].append(f"标注目录已是 {video.stem}，跳过改名")
            label_action = "done"
        elif labels.is_dir() and renamed_labels.is_dir():
            conflicts = _merge_preflight(labels, renamed_labels)
            if conflicts:
                preview = ", ".join(conflicts[:5])
                item.update(status="error", error=f"labels 与 {video.stem} 存在不同内容的同名文件: {preview}")
                items.append(item)
                continue
            item["actions"].append(f"合并 labels 到已有 {video.stem} 目录")
            label_action = "merge"
        elif labels.exists() or renamed_labels.exists():
            item.update(status="error", error="labels 或同名目标存在，但不是目录")
            items.append(item)
            continue
        else:
            item.update(status="error", error="既没有 labels 目录，也没有已改名的同名标注目录")
            items.append(item)
            continue

        if not dry_run:
            if not destination_video.exists():
                temporary = workspace / f".{video.name}.{uuid.uuid4().hex}.copying"
                try:
                    shutil.copy2(video, temporary)
                    temporary.replace(destination_video)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            if label_action == "rename":
                labels.rename(renamed_labels)
            elif label_action == "merge":
                _merge_directories(labels, renamed_labels)
            item["status"] = "done"
        items.append(item)

    counts = {
        status: sum(1 for item in items if item["status"] == status)
        for status in ("ready", "done", "skipped", "error")
    }
    return {
        "dry_run": dry_run,
        "video_dir": str(videos_root),
        "prelabel_dir": str(prelabels_root),
        "video_count": len(videos),
        "matched_count": len(items) - counts["skipped"],
        "counts": counts,
        "items": items,
    }
