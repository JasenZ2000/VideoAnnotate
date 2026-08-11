#!/usr/bin/env python3
"""Reproducible LocateAnything video inference benchmark.

The parent process extracts one fixed frame set and runs every benchmark case in
an isolated child process.  A CUDA OOM, missing optional batch runtime, or a
hung generation therefore does not prevent the remaining cases from running.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


DEFAULT_CASES = (
    "standard-slow",
    "standard-hybrid",
    "standard-fast",
    "batch-hybrid-2",
    "batch-hybrid-4",
)
BOX_PATTERN = re.compile(
    r"<ref>(?P<label>.*?)</ref>"
    r"|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>",
    re.DOTALL,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    runtime: str
    generation_mode: str
    batch_size: int


def parse_case(raw: str) -> CaseSpec:
    value = raw.strip().lower()
    if value.startswith("standard-"):
        mode = value.removeprefix("standard-")
        if mode not in {"slow", "hybrid", "fast"}:
            raise ValueError(f"Unsupported standard generation mode: {mode}")
        return CaseSpec(value, "standard", mode, 1)
    match = re.fullmatch(r"batch-(slow|hybrid|fast)-(\d+)", value)
    if match:
        size = int(match.group(2))
        if size < 1:
            raise ValueError("Batch size must be positive")
        return CaseSpec(value, "batch", match.group(1), size)
    raise ValueError(
        f"Invalid case {raw!r}; expected standard-slow or batch-hybrid-4"
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * fraction)]


def parse_boxes(answer: str, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    current_label = ""
    for match in BOX_PATTERN.finditer(answer):
        if match.group("label") is not None:
            current_label = match.group("label").strip()
            continue
        coords = [int(match.group(key)) for key in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = coords
        boxes.append(
            {
                "label": current_label,
                "bbox_xyxy": [
                    x1 / 1000 * width,
                    y1 / 1000 * height,
                    x2 / 1000 * width,
                    y2 / 1000 * height,
                ],
                "normalized_1000": coords,
            }
        )
    return boxes


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    left_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    right_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def compare_box_sets(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]], threshold: float = 0.5
) -> tuple[int, int, int]:
    pairs: list[tuple[float, int, int]] = []
    for ref_index, ref_box in enumerate(reference):
        for cand_index, cand_box in enumerate(candidate):
            if ref_box.get("label") and cand_box.get("label"):
                if ref_box["label"].strip().casefold() != cand_box["label"].strip().casefold():
                    continue
            overlap = box_iou(ref_box["bbox_xyxy"], cand_box["bbox_xyxy"])
            if overlap >= threshold:
                pairs.append((overlap, ref_index, cand_index))
    matched_ref: set[int] = set()
    matched_candidate: set[int] = set()
    for _, ref_index, cand_index in sorted(pairs, reverse=True):
        if ref_index not in matched_ref and cand_index not in matched_candidate:
            matched_ref.add(ref_index)
            matched_candidate.add(cand_index)
    matches = len(matched_ref)
    return matches, len(candidate) - matches, len(reference) - matches


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resize_for_inference(image: Any, long_edge: int) -> tuple[Any, float]:
    from PIL import Image

    if long_edge <= 0:
        return image, 1.0
    ratio = min(1.0, float(long_edge) / max(image.size))
    if ratio >= 1.0:
        return image, 1.0
    size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(size, Image.Resampling.LANCZOS), ratio


def extract_frames(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    import cv2

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        requested = args.warmup_frames + args.frames
        indices = [args.start_frame + index * args.frame_step for index in range(requested)]
        indices = [index for index in indices if 0 <= index < total]
        if len(indices) <= args.warmup_frames:
            raise RuntimeError(
                f"Video has insufficient frames for {args.warmup_frames} warmups and "
                f"{args.frames} measured frames from start={args.start_frame}, step={args.frame_step}"
            )
        rows = []
        started = time.perf_counter()
        for sequence, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to decode frame {frame_index}")
            path = frame_dir / f"frame_{sequence:04d}_source_{frame_index:08d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Unable to write extracted frame: {path}")
            rows.append(
                {
                    "sequence": sequence,
                    "source_frame": frame_index,
                    "path": str(path.resolve()),
                    "warmup": sequence < args.warmup_frames,
                }
            )
        elapsed = time.perf_counter() - started
        return {
            "video": str(args.video.resolve()),
            "video_frames": total,
            "video_fps": fps,
            "video_width": width,
            "video_height": height,
            "extraction_seconds": elapsed,
            "frames": rows,
        }
    finally:
        capture.release()


def _load_images(frame_rows: list[dict[str, Any]], long_edge: int) -> tuple[list[Any], float]:
    from PIL import Image

    images = []
    started = time.perf_counter()
    for row in frame_rows:
        with Image.open(row["path"]) as opened:
            original = opened.convert("RGB")
        resized, ratio = resize_for_inference(original, long_edge)
        images.append(resized)
        row["original_size"] = list(original.size)
        row["inference_size"] = list(resized.size)
        row["resize_ratio"] = ratio
    return images, time.perf_counter() - started


def _clear_cuda(torch: Any, device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _load_standard_worker(
    worker_type: Any,
    model_path: Path,
    device: str,
    dtype: Any,
    attention: str,
    vision_attention: str = "auto",
) -> Any:
    """Load the public worker while explicitly controlling its LLM attention.

    The released model configuration prefers Magi and may auto-fallback to
    ``flash_attention_2`` merely because flash-attn is installed.  The stock
    LocateAnything Qwen2 forward does not implement that LLM mode; LA Flash is
    wired only by the optional batch runtime.  Build the same worker state as
    its constructor, but pass an explicit supported attention implementation.
    """
    if attention == "auto":
        return worker_type(str(model_path), device=device, dtype=dtype)

    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    attention_by_backbone = {"text_config": attention}
    if vision_attention != "auto":
        attention_by_backbone["vision_config"] = vision_attention

    # LocateAnything is a composite model.  A scalar attn_implementation may be
    # accepted by AutoModel but subsequently overwritten while its Qwen2
    # submodel is constructed.  Lock the nested config as well as using the
    # Transformers per-backbone API so this also works with older remote code.
    _set_config_attention(config, "text_config", attention)
    if vision_attention != "auto":
        _set_config_attention(config, "vision_config", vision_attention)

    worker = worker_type.__new__(worker_type)
    worker.device = device
    worker.dtype = dtype
    worker.use_batch_runtime = False
    worker.scheduler = "pipeline"
    worker.group_size = 0
    worker.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    worker.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    worker.model = AutoModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attention_by_backbone,
    ).to(device).eval()
    worker._benchmark_attention_backends = _standard_attention_snapshot(worker)
    actual_text_attention = worker._benchmark_attention_backends.get("language_model.model")
    if actual_text_attention is None:
        actual_text_attention = worker._benchmark_attention_backends.get("language_model.config")
    if actual_text_attention != attention:
        raise RuntimeError(
            "Standard LLM attention override did not take effect: "
            f"requested={attention!r}, actual={actual_text_attention!r}, "
            f"snapshot={worker._benchmark_attention_backends!r}"
        )
    return worker


def _set_config_attention(config: Any, subconfig_name: str, attention: str) -> None:
    """Pin an attention backend on one member of a composite HF config."""
    subconfig = getattr(config, subconfig_name, None)
    if subconfig is None:
        raise RuntimeError(f"Model config has no {subconfig_name!r} subconfig")
    # _attn_implementation is a property on recent Transformers versions and
    # an ordinary attribute on some model-shipped config implementations.
    setattr(subconfig, "_attn_implementation", attention)
    setattr(subconfig, "_attn_implementation_internal", attention)
    setattr(subconfig, "_attn_implementation_autoset", True)


def _standard_attention_snapshot(worker: Any) -> dict[str, Any]:
    """Return the actual standard-runtime attention settings after loading."""
    model = worker.model
    language_model = getattr(model, "language_model", None)
    language_core = getattr(language_model, "model", None)

    def implementation(value: Any) -> Any:
        if value is None:
            return None
        return getattr(value, "_attn_implementation", None)

    return {
        "model.config": implementation(getattr(model, "config", None)),
        "model.config.text_config": implementation(
            getattr(getattr(model, "config", None), "text_config", None)
        ),
        "model.config.vision_config": implementation(
            getattr(getattr(model, "config", None), "vision_config", None)
        ),
        "language_model.config": implementation(getattr(language_model, "config", None)),
        "language_model.model": implementation(language_core),
    }


def render_case_previews(report_path: Path, report: dict[str, Any]) -> None:
    """Render parsed boxes after timing so visual QA does not affect throughput."""
    if not report.get("frames"):
        return
    from PIL import Image, ImageDraw

    preview_dir = report_path.parent / (report_path.stem + "_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for row in report["frames"]:
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        line_width = max(2, round(max(image.size) / 500))
        for box in row["boxes"]:
            xyxy = [round(float(value)) for value in box["bbox_xyxy"]]
            draw.rectangle(xyxy, outline=(255, 64, 64), width=line_width)
            label = str(box.get("label", "")).strip()
            if label:
                left, top = xyxy[:2]
                draw.rectangle((left, max(0, top - 18), left + max(36, len(label) * 8), top), fill=(255, 64, 64))
                draw.text((left + 2, max(0, top - 17)), label, fill=(255, 255, 255))
        preview_path = preview_dir / f"source_{int(row['source_frame']):08d}.jpg"
        image.save(preview_path, quality=92)
        row["preview"] = str(preview_path.resolve())
    report["preview_dir"] = str(preview_dir.resolve())


def _standard_generate(
    worker: Any, image: Any, prompt: str, spec: CaseSpec, config: dict[str, Any]
) -> list[dict[str, Any]]:
    result = worker.ground_multi(
        image,
        prompt,
        generation_mode=spec.generation_mode,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        verbose=False,
    )
    return [result]


def _batch_generate(
    worker: Any, images: list[Any], prompt: str, spec: CaseSpec, config: dict[str, Any]
) -> list[dict[str, Any]]:
    question = f"Locate all the instances that match the following description: {prompt}."
    return worker.predict_batch(
        [(image, question) for image in images],
        generation_mode=spec.generation_mode,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        verbose=False,
    )


def run_child_case(config_path: Path, spec: CaseSpec, report_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "case": spec.name,
        "runtime": spec.runtime,
        "generation_mode": spec.generation_mode,
        "requested_batch_size": spec.batch_size,
        "effective_batch_size": spec.batch_size,
        "status": "starting",
        "started_at": datetime.now().astimezone().isoformat(),
        "oom_retries": [],
        "frames": [],
    }
    try:
        locate_root = Path(config["locate_root"]).resolve()
        model_path = Path(config["model_path"]).resolve()
        worker_file = locate_root / "locateanything_worker.py"
        if not worker_file.is_file():
            raise FileNotFoundError(f"locateanything_worker.py is missing from {locate_root}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        sys.path.insert(0, str(model_path))
        sys.path.insert(0, str(locate_root))

        import torch
        from locateanything_worker import LocateAnythingWorker

        device = config["device"]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        torch.cuda.set_device(device)
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[config["dtype"]]
        report["torch_version"] = torch.__version__
        report["cuda_version"] = torch.version.cuda
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["device"] = device
        report["dtype"] = config["dtype"]

        load_started = time.perf_counter()
        if spec.runtime == "batch":
            worker = LocateAnythingWorker(
                str(model_path),
                device=device,
                dtype=dtype,
                use_batch_runtime=True,
                attn=config["batch_attn"],
                vision_attn=config["vision_attn"],
                scheduler=config["scheduler"],
                group_size=config["group_size"],
                strict_attn=config["strict_attn"],
            )
        else:
            worker = _load_standard_worker(
                LocateAnythingWorker,
                model_path,
                device,
                dtype,
                config["standard_attn"],
                config["vision_attn"],
            )
            report["standard_attn"] = config["standard_attn"]
            report["standard_attention_backends"] = getattr(
                worker, "_benchmark_attention_backends", {}
            )
        report["model_load_seconds"] = time.perf_counter() - load_started
        report["memory_after_load_gib"] = torch.cuda.memory_reserved(device) / (1024**3)

        frame_rows = [dict(row) for row in config["frame_manifest"]["frames"]]
        images, image_load_seconds = _load_images(frame_rows, config["resize_long_edge"])
        report["image_load_resize_seconds"] = image_load_seconds
        warmup_count = sum(bool(row["warmup"]) for row in frame_rows)
        measured_rows = frame_rows[warmup_count:]
        measured_images = images[warmup_count:]
        warmup_images = images[:warmup_count]

        for image in warmup_images:
            if spec.runtime == "batch":
                _batch_generate(worker, [image], config["prompt"], spec, config)
            else:
                _standard_generate(worker, image, config["prompt"], spec, config)
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        inference_started = time.perf_counter()
        index = 0
        effective_batch = spec.batch_size
        latencies: list[float] = []
        while index < len(measured_images):
            current_size = 1 if spec.runtime == "standard" else min(
                effective_batch, len(measured_images) - index
            )
            batch_images = measured_images[index : index + current_size]
            batch_rows = measured_rows[index : index + current_size]
            try:
                started = time.perf_counter()
                if spec.runtime == "batch":
                    results = _batch_generate(worker, batch_images, config["prompt"], spec, config)
                else:
                    results = _standard_generate(
                        worker, batch_images[0], config["prompt"], spec, config
                    )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if len(results) != current_size:
                    raise RuntimeError(
                        f"Runtime returned {len(results)} answers for batch size {current_size}"
                    )
                per_image = elapsed / current_size
                latencies.extend([per_image] * current_size)
                for row, result in zip(batch_rows, results):
                    answer = str(result.get("answer", ""))
                    width, height = row["original_size"]
                    report["frames"].append(
                        {
                            **row,
                            "latency_seconds": per_image,
                            "batch_wall_seconds": elapsed,
                            "batch_size": current_size,
                            "answer": answer,
                            "boxes": parse_boxes(answer, width, height),
                        }
                    )
                index += current_size
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                report["oom_retries"].append(
                    {
                        "at_measured_index": index,
                        "failed_batch_size": current_size,
                        "error": str(exc),
                    }
                )
                _clear_cuda(torch, device)
                if spec.runtime == "batch" and effective_batch > 1:
                    effective_batch = max(1, effective_batch // 2)
                    report["effective_batch_size"] = effective_batch
                    continue
                report["status"] = "oom"
                report["error"] = str(exc)
                break

        inference_seconds = time.perf_counter() - inference_started
        completed = len(report["frames"])
        box_count = sum(len(row["boxes"]) for row in report["frames"])
        report.update(
            {
                "completed_frames": completed,
                "requested_frames": len(measured_images),
                "inference_seconds": inference_seconds,
                "frames_per_second": completed / inference_seconds if inference_seconds else 0.0,
                "boxes_per_second": box_count / inference_seconds if inference_seconds else 0.0,
                "total_boxes": box_count,
                "latency_mean_seconds": mean(latencies) if latencies else None,
                "latency_p50_seconds": median(latencies) if latencies else None,
                "latency_p95_seconds": percentile(latencies, 0.95),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
        )
        if report["status"] == "starting":
            report["status"] = "ok" if completed == len(measured_images) else "partial"
        del worker
        _clear_cuda(torch, device)
    except (ImportError, ModuleNotFoundError) as exc:
        report["status"] = "unsupported" if spec.runtime == "batch" else "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    except Exception as exc:
        batch_api_missing = spec.runtime == "batch" and (
            isinstance(exc, TypeError)
            or "batch_utils" in str(exc)
            or "kernel_utils" in str(exc)
        )
        if is_cuda_oom(exc):
            report["status"] = "oom"
        elif batch_api_missing:
            report["status"] = "unsupported"
        else:
            report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        try:
            render_case_previews(report_path, report)
        except Exception as preview_exc:
            report["preview_error"] = str(preview_exc)
        atomic_json(report_path, report)
    return 0 if report["status"] in {"ok", "partial", "oom", "unsupported"} else 1


def summarize_reports(output_dir: Path, reports: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(
        (report for report in reports if report["case"] == "standard-slow" and report["status"] == "ok"),
        None,
    )
    baseline_frames = {
        row["source_frame"]: row for row in baseline.get("frames", [])
    } if baseline else {}
    rows = []
    for report in reports:
        true_positive = false_positive = false_negative = 0
        for frame in report.get("frames", []):
            reference = baseline_frames.get(frame["source_frame"])
            if reference is None or report is baseline:
                continue
            matched, extra, missed = compare_box_sets(reference["boxes"], frame["boxes"])
            true_positive += matched
            false_positive += extra
            false_negative += missed
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        baseline_fps = baseline.get("frames_per_second", 0.0) if baseline else 0.0
        fps = report.get("frames_per_second", 0.0) or 0.0
        row = {
            "case": report["case"],
            "status": report["status"],
            "effective_batch_size": report.get("effective_batch_size"),
            "completed_frames": report.get("completed_frames", 0),
            "frames_per_second": fps,
            "speedup_vs_slow": fps / baseline_fps if baseline_fps and fps else None,
            "boxes_per_second": report.get("boxes_per_second"),
            "latency_p50_seconds": report.get("latency_p50_seconds"),
            "latency_p95_seconds": report.get("latency_p95_seconds"),
            "peak_reserved_gib": report.get("peak_reserved_gib"),
            "total_boxes": report.get("total_boxes"),
            "box_precision_vs_slow_iou50": precision,
            "box_recall_vs_slow_iou50": recall,
            "oom_retries": len(report.get("oom_retries", [])),
            "error": report.get("error", ""),
        }
        rows.append(row)

    summary = {"baseline": "standard-slow", "iou_threshold": 0.5, "cases": rows}
    atomic_json(output_dir / "summary.json", summary)
    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case", "status"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# LocateAnything benchmark summary",
        "",
        "`standard-slow` is treated as the output-comparison reference, not ground truth.",
        "",
        "| Case | Status | Batch | FPS | vs slow | P50 s | P95 s | Peak GiB | Boxes | P@IoU50 | R@IoU50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any, digits: int = 3) -> str:
            return "-" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"| {row['case']} | {row['status']} | {row['effective_batch_size'] or '-'} | "
            f"{fmt(row['frames_per_second'], 4)} | {fmt(row['speedup_vs_slow'], 2)} | "
            f"{fmt(row['latency_p50_seconds'])} | {fmt(row['latency_p95_seconds'])} | "
            f"{fmt(row['peak_reserved_gib'], 2)} | {row['total_boxes'] or 0} | "
            f"{fmt(row['box_precision_vs_slow_iou50'])} | {fmt(row['box_recall_vs_slow_iou50'])} |"
        )
    lines.extend(
        [
            "",
            "Inspect each `cases/<case>.json` for raw answers, boxes, OOM retries, and tracebacks.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_parent(args: argparse.Namespace) -> int:
    locate_root = args.locate_root.resolve()
    model_path = args.model_path.resolve()
    video = args.video.resolve()
    for path, description in (
        (locate_root / "locateanything_worker.py", "LocateAnything worker"),
        (model_path, "model"),
        (video, "video"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} path does not exist: {path}")

    output_dir = args.output.resolve() if args.output else Path.cwd() / (
        "locany-benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases").mkdir(exist_ok=True)

    frame_manifest = extract_frames(args, output_dir)
    atomic_json(output_dir / "frames.json", frame_manifest)
    cases = [parse_case(value) for value in args.cases.split(",") if value.strip()]
    config = {
        "locate_root": str(locate_root),
        "model_path": str(model_path),
        "video": str(video),
        "device": args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "resize_long_edge": args.resize_long_edge,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "batch_attn": args.batch_attn,
        "standard_attn": args.standard_attn,
        "vision_attn": args.vision_attn,
        "scheduler": args.scheduler,
        "group_size": args.group_size,
        "strict_attn": args.strict_attn,
        "timeout_per_case": args.timeout_per_case,
        "frame_manifest": frame_manifest,
        "cases": [spec.__dict__ for spec in cases],
    }
    config_path = output_dir / "benchmark_config.json"
    atomic_json(config_path, config)

    reports = []
    for spec in cases:
        report_path = output_dir / "cases" / f"{spec.name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_run-case",
            spec.name,
            "--_config",
            str(config_path),
            "--_report",
            str(report_path),
        ]
        print(f"[benchmark] starting {spec.name}", flush=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, timeout=args.timeout_per_case, check=False)
            if not report_path.is_file():
                atomic_json(
                    report_path,
                    {
                        "case": spec.name,
                        "status": "failed",
                        "error": f"Child exited {completed.returncode} without a report",
                    },
                )
        except subprocess.TimeoutExpired:
            atomic_json(
                report_path,
                {
                    "case": spec.name,
                    "runtime": spec.runtime,
                    "generation_mode": spec.generation_mode,
                    "requested_batch_size": spec.batch_size,
                    "status": "timeout",
                    "error": f"Exceeded timeout_per_case={args.timeout_per_case}s; child was terminated",
                },
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["parent_wall_seconds"] = time.perf_counter() - started
        atomic_json(report_path, report)
        reports.append(report)
        print(
            f"[benchmark] finished {spec.name}: {report['status']} "
            f"fps={report.get('frames_per_second', 0):.4f}",
            flush=True,
        )

    summarize_reports(output_dir, reports)
    print(f"[benchmark] summary: {output_dir / 'summary.md'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark LocateAnything generation modes and optional batch runtime on fixed video frames"
    )
    parser.add_argument("--locate-root", type=Path, help="Directory containing locateanything_worker.py")
    parser.add_argument("--model-path", type=Path, help="Local LocateAnything model directory")
    parser.add_argument("--video", type=Path, help="Test video")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="person</c>car")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--frames", type=int, default=8, help="Measured frames per case")
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--resize-long-edge", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--batch-attn", default="la_flash")
    parser.add_argument(
        "--standard-attn",
        choices=("sdpa", "magi", "auto"),
        default="sdpa",
        help="LLM attention for standard cases; sdpa avoids the unsupported flash_attention_2 auto-fallback",
    )
    parser.add_argument("--vision-attn", default="auto")
    parser.add_argument("--scheduler", default="pipeline")
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--strict-attn", action="store_true")
    parser.add_argument("--timeout-per-case", type=int, default=1800)
    parser.add_argument("--_run-case", help=argparse.SUPPRESS)
    parser.add_argument("--_config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_report", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args._run_case:
        if args._config is None or args._report is None:
            parser.error("Internal case mode requires --_config and --_report")
        return run_child_case(args._config, parse_case(args._run_case), args._report)
    missing = [name for name in ("locate_root", "model_path", "video") if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.frames < 1 or args.warmup_frames < 0 or args.frame_step < 1:
        parser.error("--frames and --frame-step must be positive; --warmup-frames cannot be negative")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
