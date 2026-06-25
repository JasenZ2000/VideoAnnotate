from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from locateanything_worker import LocateAnythingWorker


def parse_dtype(raw: str) -> torch.dtype:
    value = raw.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {raw}")


def print_environment() -> None:
    print("== Environment ==")
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_version: {torch.version.cuda}")
        print(f"gpu_count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / 1024**3
            print(f"gpu[{idx}]: {props.name}, {total_gb:.1f} GiB")
    print()


def resize_for_inference(
    image: Image.Image,
    long_edge: int = 0,
    scale: float = 1.0,
) -> tuple[Image.Image, float]:
    if long_edge > 0:
        ratio = min(1.0, float(long_edge) / float(max(image.size)))
    else:
        ratio = float(scale)

    if ratio <= 0:
        raise ValueError("Resize scale must be > 0")
    if ratio >= 1.0:
        return image, 1.0

    new_size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS), ratio


def extract_items(answer: str, image_width: int, image_height: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(?:<ref>(?P<label>.*?)</ref>)?"
        r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
    )
    for match in pattern.finditer(answer):
        x1, y1, x2, y2 = [int(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        label = (match.group("label") or "").strip()
        items.append(
            {
                "type": "box",
                "label": label,
                "normalized": [x1, y1, x2, y2],
                "bbox_xyxy": [
                    x1 / 1000 * image_width,
                    y1 / 1000 * image_height,
                    x2 / 1000 * image_width,
                    y2 / 1000 * image_height,
                ],
            }
        )

    point_pattern = re.compile(r"<box><(?P<x>\d+)><(?P<y>\d+)></box>")
    for match in point_pattern.finditer(answer):
        x, y = int(match.group("x")), int(match.group("y"))
        items.append(
            {
                "type": "point",
                "normalized": [x, y],
                "point_xy": [x / 1000 * image_width, y / 1000 * image_height],
            }
        )
    return items


def draw_items(image: Image.Image, items: list[dict[str, Any]], output_path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, item in enumerate(items, start=1):
        if item["type"] == "box":
            x1, y1, x2, y2 = item["bbox_xyxy"]
            label = item.get("label") or f"box {idx}"
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            text = str(label)
            left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
            pad = 4
            bg = [x1, max(0, y1 - (bottom - top) - pad * 2), x1 + (right - left) + pad * 2, y1]
            draw.rectangle(bg, fill=(0, 255, 0))
            draw.text((bg[0] + pad, bg[1] + pad), text, fill=(0, 0, 0), font=font)
        elif item["type"] == "point":
            x, y = item["point_xy"]
            radius = 8
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 40, 40))
            draw.text((x + 10, y + 10), f"point {idx}", fill=(255, 40, 40), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_question(args: argparse.Namespace) -> str | None:
    if args.question:
        return args.question
    if args.task == "detect":
        categories = [item.strip() for item in args.categories.split(",") if item.strip()]
        if not categories:
            raise ValueError("--categories is required for task=detect")
        return None
    if args.task in {"ground_single", "ground_multi", "ground_text", "ground_gui", "point"}:
        if not args.prompt:
            raise ValueError(f"--prompt is required for task={args.task}")
        return None
    return None


def run_task(worker: LocateAnythingWorker, image: Image.Image, args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "verbose": True,
    }
    question = build_question(args)
    if question:
        return worker.predict(image, question, **common)
    if args.task == "detect":
        categories = [item.strip() for item in args.categories.split(",") if item.strip()]
        return worker.detect(image, categories, **common)
    if args.task == "ground_single":
        return worker.ground_single(image, args.prompt, **common)
    if args.task == "ground_multi":
        return worker.ground_multi(image, args.prompt, **common)
    if args.task == "ground_text":
        return worker.ground_text(image, args.prompt, **common)
    if args.task == "detect_text":
        return worker.detect_text(image, **common)
    if args.task == "ground_gui":
        return worker.ground_gui(image, args.prompt, output_type=args.output_type, **common)
    if args.task == "point":
        return worker.point(image, args.prompt, **common)
    raise ValueError(f"Unsupported task: {args.task}")


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "LocateAnything remote processor code uses Python 3.10+ type syntax. "
            f"Current Python is {sys.version.split()[0]}; please use Python 3.10 or newer."
        )

    parser = argparse.ArgumentParser(description="Debug LocateAnything inference end to end.")
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--task",
        choices=["detect", "ground_single", "ground_multi", "ground_text", "detect_text", "ground_gui", "point"],
        default="ground_multi",
    )
    parser.add_argument("--prompt", default="", help="Natural language target for grounding tasks.")
    parser.add_argument("--categories", default="person,car,bicycle", help="Comma-separated categories for detect.")
    parser.add_argument("--question", default="", help="Raw question. Overrides --task prompt construction.")
    parser.add_argument("--output-type", choices=["box", "point"], default="box")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("locateanything_debug_out"))
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=0,
        help="Resize the image for inference so the long edge is at most this value. Default keeps original size.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=1.0,
        help="Uniform inference resize scale used when --resize-long-edge is not set.",
    )
    parser.add_argument(
        "--save-resized-image",
        action="store_true",
        help="Save the resized image used for inference.",
    )
    parser.add_argument("--no-draw", action="store_true")
    args = parser.parse_args()

    print_environment()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    original_image = Image.open(image_path).convert("RGB")
    inference_image, resize_ratio = resize_for_inference(
        original_image,
        long_edge=args.resize_long_edge,
        scale=args.resize_scale,
    )
    print(f"image: {image_path} ({original_image.width}x{original_image.height})")
    if inference_image.size != original_image.size:
        print(f"inference_image: {inference_image.width}x{inference_image.height} (scale={resize_ratio:.6f})")
    print(f"model: {args.model}")
    print(f"device: {args.device}, dtype: {args.dtype}, generation_mode: {args.generation_mode}")
    print()

    load_start = time.perf_counter()
    worker = LocateAnythingWorker(args.model, device=args.device, dtype=parse_dtype(args.dtype))
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds: {load_seconds:.3f}")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        before_mem = torch.cuda.max_memory_allocated() / 1024**3
    else:
        before_mem = 0.0

    infer_start = time.perf_counter()
    result = run_task(worker, inference_image, args)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_start

    answer = str(result.get("answer", ""))
    items = extract_items(answer, original_image.width, original_image.height)
    payload = {
        "image": str(image_path),
        "image_size": [original_image.width, original_image.height],
        "inference_image_size": [inference_image.width, inference_image.height],
        "inference_resize_ratio": resize_ratio,
        "model": args.model,
        "task": args.task,
        "prompt": args.prompt,
        "categories": args.categories,
        "question": args.question,
        "generation_mode": args.generation_mode,
        "dtype": args.dtype,
        "device": args.device,
        "model_load_seconds": load_seconds,
        "infer_seconds": infer_seconds,
        "cuda_max_memory_gib_before_infer": before_mem,
        "answer": answer,
        "items": items,
        "stats": result.get("stats"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{image_path.stem}_{args.task}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("== Answer ==")
    print(answer)
    print()
    print(f"parsed_items: {len(items)}")
    print(f"infer_seconds: {infer_seconds:.3f}")
    print(f"json: {json_path.resolve()}")

    if not args.no_draw:
        draw_path = args.out_dir / f"{image_path.stem}_{args.task}_vis.jpg"
        draw_items(original_image, items, draw_path)
        print(f"vis: {draw_path.resolve()}")
    if args.save_resized_image and inference_image.size != original_image.size:
        resized_path = args.out_dir / f"{image_path.stem}_inference_{inference_image.width}x{inference_image.height}.jpg"
        inference_image.save(resized_path)
        print(f"resized_image: {resized_path.resolve()}")


if __name__ == "__main__":
    main()
