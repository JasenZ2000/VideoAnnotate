from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MARKERS = ("images", "labels", "annotations")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass
class ScanItem:
    relative_path: str
    full_path: str
    depth: int
    matched_markers: list[str] = field(default_factory=list)
    image_count: int | None = None


@dataclass
class ScanResult:
    root: str
    items: list[ScanItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def manifest_lines(self, full_paths: bool = False, include_image_counts: bool = False) -> list[str]:
        lines = []
        for item in self.items:
            path = item.full_path if full_paths else item.relative_path
            lines.append(f"{path}\t{item.image_count or 0}" if include_image_counts else path)
        return lines


def _count_images(directory: Path) -> int:
    count = 0
    try:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
                count += 1
    except (OSError, PermissionError):
        return count
    return count


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return True


def scan_part_directories(
    root: str | Path,
    *,
    max_depth: int = 4,
    marker_directories: tuple[str, ...] | list[str] = DEFAULT_MARKERS,
    minimum_marker_count: int = 1,
) -> ScanResult:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"根目录不存在或不是目录：{root_path}")
    if max_depth < 0 or max_depth > 32:
        raise ValueError("最大扫描深度必须在 0 到 32 之间")
    markers = {str(marker).strip().casefold() for marker in marker_directories if str(marker).strip()}
    if not markers:
        raise ValueError("请至少填写一个识别标志目录")
    if minimum_marker_count < 1 or minimum_marker_count > len(markers):
        raise ValueError("最少命中数量不能超过识别标志目录数量")

    result = ScanResult(root=str(root_path))

    def visit(directory: Path, depth: int) -> None:
        try:
            children = sorted(
                (child for child in directory.iterdir() if child.is_dir()),
                key=lambda child: child.name.casefold(),
            )
        except (OSError, PermissionError) as exc:
            result.warnings.append(f"无法读取：{directory}（{exc}）")
            return
        child_names = {child.name.casefold(): child.name for child in children}
        matched = sorted(child_names[marker] for marker in markers if marker in child_names)
        if len(matched) >= minimum_marker_count:
            relative = os.path.relpath(directory, root_path).replace("\\", "/")
            result.items.append(ScanItem(
                relative_path=relative,
                full_path=str(directory),
                depth=depth,
                matched_markers=matched,
                image_count=_count_images(directory / child_names["images"])
                if any(name.casefold() == "images" for name in matched) else None,
            ))
            return
        if depth >= max_depth:
            return
        for child in children:
            if _is_reparse_point(child):
                continue
            visit(child, depth + 1)

    visit(root_path, 0)
    result.items.sort(key=lambda item: item.relative_path.casefold())
    return result
