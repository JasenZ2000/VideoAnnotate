from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from debug_infer import extract_items, parse_dtype, resize_for_inference
from locateanything_worker import LocateAnythingWorker


DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.bmp,.webp"


def parse_extensions(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def iter_images(root: Path, extensions: set[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in extensions:
                yield path


def output_path_for_image(image_path: Path, input_root: Path, output_dir: Path, include_root_name: bool) -> Path:
    rel = image_path.relative_to(input_root)
    if include_root_name:
        rel = Path(input_root.name) / rel
    return (output_dir / rel).with_suffix(".txt")


def clip_box(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_to_yolo_line(
    box: list[float],
    width: int,
    height: int,
    class_id: int,
    score: float,
) -> str | None:
    clipped = clip_box(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (
        f"{class_id} "
        f"{cx / width:.6f} {cy / height:.6f} "
        f"{bw / width:.6f} {bh / height:.6f} "
        f"{score:.6f}"
    )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_model(worker: LocateAnythingWorker, image: Image.Image, args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "use_cache": not args.no_use_cache,
        "verbose": args.verbose,
    }
    if args.question:
        return worker.predict(image, args.question, **common)
    if args.task == "detect":
        return worker.detect(image, [args.target], **common)
    if args.task == "ground_single":
        return worker.ground_single(image, args.target, **common)
    return worker.ground_multi(image, args.target, **common)


def process_image(
    worker: LocateAnythingWorker,
    image_path: Path,
    txt_path: Path,
    args: argparse.Namespace,
    raw_jsonl: Path | None,
) -> dict[str, Any]:
    original = Image.open(image_path).convert("RGB")
    inference_image, resize_ratio = resize_for_inference(
        original,
        long_edge=args.resize_long_edge,
        scale=args.resize_scale,
    )

    start = time.perf_counter()
    result = run_model(worker, inference_image, args)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    answer = str(result.get("answer", ""))
    items = extract_items(answer, original.width, original.height)
    lines = []
    boxes = []
    for item in items:
        if item.get("type") != "box":
            continue
        box = [float(value) for value in item["bbox_xyxy"]]
        line = box_to_yolo_line(box, original.width, original.height, args.class_id, args.score)
        if line is None:
            continue
        lines.append(line)
        boxes.append(
            {
                "label": item.get("label", ""),
                "bbox_xyxy": box,
                "normalized_token_box": item.get("normalized"),
            }
        )

    write_text_atomic(txt_path, "\n".join(lines) + ("\n" if lines else ""))

    summary = {
        "image": str(image_path),
        "output": str(txt_path),
        "image_size": [original.width, original.height],
        "inference_image_size": [inference_image.width, inference_image.height],
        "inference_resize_ratio": resize_ratio,
        "num_boxes": len(lines),
        "seconds": seconds,
    }
    if raw_jsonl is not None:
        append_jsonl(
            raw_jsonl,
            {
                **summary,
                "answer": answer,
                "boxes": boxes,
                "stats": result.get("stats"),
            },
        )
    return summary


def collect_work(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    extensions = parse_extensions(args.extensions)
    output_dir = args.output_dir.expanduser().resolve()
    work: list[tuple[Path, Path, Path]] = []
    for raw_root in args.input_roots:
        input_root = raw_root.expanduser().resolve()
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input root is not a directory: {input_root}")
        for image_path in iter_images(input_root, extensions):
            txt_path = output_path_for_image(
                image_path=image_path,
                input_root=input_root,
                output_dir=output_dir,
                include_root_name=not args.no_root_name,
            )
            work.append((input_root, image_path, txt_path))
    if args.num_shards > 1:
        work = [item for idx, item in enumerate(work) if idx % args.num_shards == args.shard_index]
    if args.limit > 0:
        work = work[: args.limit]
    return work


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "LocateAnything remote processor code uses Python 3.10+ type syntax. "
            f"Current Python is {sys.version.split()[0]}; please use Python 3.10 or newer."
        )

    parser = argparse.ArgumentParser(description="Batch LocateAnything inference and write YOLO txt files.")
    parser.add_argument("--input-roots", nargs="+", required=True, type=Path, help="One or more image folder roots.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--task", choices=["ground_multi", "ground_single", "detect"], default="ground_multi")
    parser.add_argument("--target", default="person", help="Target phrase/category. Class id is still written as --class-id.")
    parser.add_argument("--question", default="", help="Raw prompt override.")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--score", type=float, default=1.0, help="LocateAnything has no confidence score; this value is written.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--generation-mode", choices=["fast", "slow", "hybrid"], default="slow")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-use-cache", action="store_true", help="Lower KV-cache memory at the cost of speed.")
    parser.add_argument("--resize-long-edge", type=int, default=1024)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-root-name", action="store_true", help="Do not include input root folder name under output-dir.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--status-every", type=int, default=50)
    parser.add_argument("--raw-jsonl", type=Path, default=None, help="Optional JSONL log with raw answers and parsed boxes.")
    parser.add_argument("--error-jsonl", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    output_dir = args.output_dir.expanduser().resolve()
    raw_jsonl = args.raw_jsonl.expanduser().resolve() if args.raw_jsonl else None
    error_jsonl = (
        args.error_jsonl.expanduser().resolve()
        if args.error_jsonl
        else output_dir / f"errors_shard{args.shard_index}.jsonl"
    )

    work = collect_work(args)
    total = len(work)
    print(f"inputs: {len(args.input_roots)} root(s)")
    print(f"images_this_shard: {total}")
    print(f"output_dir: {output_dir}")
    print(f"task: {args.task}, target: {args.target!r}, class_id: {args.class_id}, score: {args.score}")
    print(f"resize_long_edge: {args.resize_long_edge}, max_new_tokens: {args.max_new_tokens}")
    print(f"device: {args.device}, dtype: {args.dtype}, generation_mode: {args.generation_mode}")

    load_start = time.perf_counter()
    worker = LocateAnythingWorker(args.model, device=args.device, dtype=parse_dtype(args.dtype))
    print(f"model_loaded_seconds: {time.perf_counter() - load_start:.3f}")

    done = 0
    skipped = 0
    failed = 0
    start_all = time.perf_counter()
    for index, (_, image_path, txt_path) in enumerate(work, start=1):
        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            summary = process_image(worker, image_path, txt_path, args, raw_jsonl)
            done += 1
            if args.verbose:
                print(f"[{index}/{total}] boxes={summary['num_boxes']} {image_path} -> {txt_path}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failed += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            append_jsonl(
                error_jsonl,
                {
                    "image": str(image_path),
                    "output": str(txt_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[error] {image_path}: {exc}", file=sys.stderr)

        if args.status_every > 0 and index % args.status_every == 0:
            elapsed = time.perf_counter() - start_all
            rate = index / elapsed if elapsed > 0 else 0.0
            print(
                f"[status] {index}/{total} seen, done={done}, skipped={skipped}, "
                f"failed={failed}, rate={rate:.3f} img/s"
            )

    elapsed = time.perf_counter() - start_all
    print(
        f"finished: total={total}, done={done}, skipped={skipped}, "
        f"failed={failed}, seconds={elapsed:.1f}"
    )
    if failed:
        print(f"errors: {error_jsonl}")


if __name__ == "__main__":
    main()
