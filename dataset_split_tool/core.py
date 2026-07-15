from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    stem: str
    image: Path
    label: Path
    annotation: Path
    dataset_root: Path


@dataclass(frozen=True)
class ScanResult:
    dataset_roots: tuple[Path, ...]
    samples: tuple[Sample, ...]
    warnings: tuple[str, ...]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _dataset_roots(ori_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in [ori_dir, *sorted(path for path in ori_dir.rglob("*") if path.is_dir())]:
        if all((candidate / name).is_dir() for name in ("images", "labels", "annotations")):
            roots.append(candidate)
    return roots


def _unique_files_by_stem(directory: Path, extensions: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError(f"同一目录存在重复文件名 stem：{result[key]} / {path}")
        result[key] = path
    return result


def scan_samples(ori_dir: str | Path) -> ScanResult:
    ori = Path(ori_dir).expanduser().resolve()
    if not ori.is_dir():
        raise ValueError(f"找不到 ori 目录：{ori}")

    roots = _dataset_roots(ori)
    if not roots:
        raise ValueError(f"没有找到同时包含 images、labels、annotations 的目录：{ori}")

    samples: list[Sample] = []
    warnings: list[str] = []
    seen: dict[str, Sample] = {}
    for root in roots:
        images = _unique_files_by_stem(root / "images", IMAGE_EXTENSIONS)
        labels = _unique_files_by_stem(root / "labels", {".txt"})
        annotations = _unique_files_by_stem(root / "annotations", {".xml"})

        for key, image in images.items():
            missing = []
            if key not in labels:
                missing.append("label TXT")
            if key not in annotations:
                missing.append("annotation XML")
            if missing:
                raise ValueError(f"{image} 缺少对应的 {'、'.join(missing)}")
            sample = Sample(image.stem, image, labels[key], annotations[key], root)
            if key in seen:
                raise ValueError(
                    f"不同数据集存在重复 stem，无法安全汇总：{seen[key].image} / {image}"
                )
            seen[key] = sample
            samples.append(sample)

        extra_labels = sorted(path.name for key, path in labels.items() if key not in images)
        extra_annotations = sorted(path.name for key, path in annotations.items() if key not in images)
        if extra_labels:
            warnings.append(f"{root}: 跳过 {len(extra_labels)} 个无对应图片的 TXT")
        if extra_annotations:
            warnings.append(f"{root}: 跳过 {len(extra_annotations)} 个无对应图片的 XML")

    if not samples:
        raise ValueError("找到数据集目录，但没有完整的 image/label/annotation 样本")
    samples.sort(key=lambda item: (item.stem.casefold(), str(item.image).casefold()))
    return ScanResult(tuple(roots), tuple(samples), tuple(warnings))


def split_dataset(ori_dir: str | Path, output_dir: str | Path, parts: int) -> dict[str, object]:
    scan = scan_samples(ori_dir)
    part_count = int(parts)
    if part_count < 1:
        raise ValueError("份数必须大于 0")
    if part_count > len(scan.samples):
        raise ValueError(f"份数 {part_count} 不能超过完整样本数 {len(scan.samples)}")

    ori = Path(ori_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == ori or _is_relative_to(output, ori):
        raise ValueError("输出目录不能位于 ori 目录内部")
    if output.exists():
        raise ValueError(f"输出目录已存在，请更换目录或先自行移走旧结果：{output}")

    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    counts = [0] * part_count
    try:
        for index in range(part_count):
            part_dir = staging / f"part_{index + 1:03d}"
            for name in ("images", "labels", "annotations"):
                (part_dir / name).mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(scan.samples):
            part_index = index % part_count
            part_dir = staging / f"part_{part_index + 1:03d}"
            shutil.copy2(sample.image, part_dir / "images" / sample.image.name)
            shutil.copy2(sample.label, part_dir / "labels" / f"{sample.stem}.txt")
            shutil.copy2(sample.annotation, part_dir / "annotations" / f"{sample.stem}.xml")
            counts[part_index] += 1

        staging.replace(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "output_dir": str(output),
        "dataset_roots": len(scan.dataset_roots),
        "samples": len(scan.samples),
        "parts": part_count,
        "counts": counts,
        "warnings": list(scan.warnings),
    }
