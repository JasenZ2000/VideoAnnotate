from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_COMFY_ROOT = Path("/data2/DET_Group/ZZS/generate/update/ComfyUI")
DEFAULT_CHECKPOINT = Path("/data2/DET_Group/ZZS/my_sam3/sam3.1_multiplex_fp16.safetensors")
IMAGE_SIZE = 1008


def log(message: str) -> None:
    print(f"[sam31] {message}", flush=True)


def parse_bbox(raw: str) -> List[float]:
    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'x1,y1,x2,y2' or 'x1 y1 x2 y2'")
    bbox = [float(v) for v in parts]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise argparse.ArgumentTypeError("bbox must satisfy x2>x1 and y2>y1")
    return bbox


def add_comfy_root(comfy_root: Path) -> None:
    root = comfy_root.expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Comfy root does not exist: {root}")
    sys.path.insert(0, str(root))
    log(f"Comfy root: {root}")


def find_sam31_model(root: Any) -> Any:
    candidates = [
        root,
        getattr(root, "model", None),
        getattr(root, "diffusion_model", None),
        getattr(getattr(root, "model", None), "diffusion_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "forward_video"):
            return candidate
    if hasattr(root, "named_modules"):
        for name, module in root.named_modules():
            if hasattr(module, "forward_video"):
                log(f"found SAM3.1 module: {name} ({type(module)})")
                return module
    raise RuntimeError("Could not find a SAM3.1 module with forward_video in the Comfy model wrapper.")


def load_sam31(comfy_root: Path, checkpoint: Path, device: str, need_clip: bool):
    add_comfy_root(comfy_root)
    import torch
    from comfy.sd import load_checkpoint_guess_config

    log(f"checkpoint: {checkpoint}")
    model_patcher, clip, _, _ = load_checkpoint_guess_config(
        str(checkpoint),
        output_vae=False,
        output_clip=need_clip,
        output_clipvision=False,
        output_model=True,
    )
    if model_patcher is None:
        raise RuntimeError("Checkpoint did not produce a SAM3.1 model.")
    dev = torch.device(device)
    log(f"patching model to {dev}")
    patched = model_patcher.patch_model(device_to=dev)
    model = find_sam31_model(patched)
    model.eval()
    log(f"using model: {type(model)}")
    return torch, model_patcher, model, clip, dev


def encode_prompt_pack(clip, prompt: str) -> Dict[str, Any]:
    log(f"encoding prompt: {prompt!r}")
    tokens = clip.tokenize(prompt)
    encoded = clip.encode_from_tokens(tokens, return_dict=True)
    multi = encoded.get("sam3_multi_cond") or [{
        "cond": encoded["cond"],
        "attention_mask": encoded.get("attention_mask"),
        "max_detections": 1,
    }]
    text_prompts = []
    max_objects = 0
    for item in multi:
        cond = item["cond"].detach().cpu()
        mask = item.get("attention_mask")
        if mask is not None:
            mask = mask.detach().cpu().bool()
        text_prompts.append((cond, mask))
        max_objects += int(item.get("max_detections", 1))
    return {"prompt": prompt, "text_prompts": text_prompts, "max_objects": max_objects}


def load_video_tensor(torch, cv2, np, video_path: Path, start_frame: int, max_frames: int, device, dtype, log_every: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 0 or start_frame >= frame_count:
        raise RuntimeError(f"start_frame {start_frame} out of range for {frame_count} frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = frame_count - start_frame if max_frames <= 0 else min(frame_count - start_frame, max_frames)
    images = torch.empty((limit, 3, IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=dtype)
    used = 0
    for idx in range(limit):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)
        images[idx].copy_(tensor)
        used += 1
        if log_every > 0 and (idx + 1) % log_every == 0:
            log(f"preprocessed frames: {idx + 1}/{limit}")
    cap.release()
    if used == 0:
        raise RuntimeError("No frames read from video.")
    images = images[:used]
    images.sub_(0.5).div_(0.5)
    meta = {
        "video_path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "start_frame": start_frame,
        "used_frames": used,
    }
    return images, meta


def scale_bbox_to_model(bbox: Sequence[float], width: int, height: int) -> List[float]:
    sx = IMAGE_SIZE / float(width)
    sy = IMAGE_SIZE / float(height)
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]


def rectangle_mask(torch, bbox: Sequence[float], width: int, height: int, device, dtype):
    x1, y1, x2, y2 = [int(round(v)) for v in scale_bbox_to_model(bbox, width, height)]
    x1 = max(0, min(IMAGE_SIZE - 1, x1))
    y1 = max(0, min(IMAGE_SIZE - 1, y1))
    x2 = max(0, min(IMAGE_SIZE, x2))
    y2 = max(0, min(IMAGE_SIZE, y2))
    mask = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=dtype)
    mask[:, y1:y2, x1:x2] = 1.0
    return mask


def initial_masks_from_boxes(torch, model, first_image, bboxes, meta, device, dtype, use_rect_mask: bool):
    masks = []
    for bbox in bboxes:
        if use_rect_mask:
            masks.append(rectangle_mask(torch, bbox, meta["width"], meta["height"], device, dtype))
            continue
        if not hasattr(model, "forward_segment"):
            raise RuntimeError("Loaded SAM3.1 object has no forward_segment; rerun with --use-rect-mask.")
        x1, y1, x2, y2 = scale_bbox_to_model(bbox, meta["width"], meta["height"])
        box = torch.tensor([[[x1, y1], [x2, y2]]], device=device, dtype=dtype)
        mask_logits = model.forward_segment(first_image, box_inputs=box)
        masks.append((mask_logits[0] > 0).to(dtype=dtype))
    return torch.stack(masks, dim=0)


def mask_to_bbox(np, mask, min_area: int) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_area:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def rows_from_packed_masks(np, cv2, packed, meta, min_area: int, class_id: int) -> List[Dict[str, Any]]:
    rows = []
    h, w = int(meta["height"]), int(meta["width"])
    start = int(meta["start_frame"])
    packed_np = packed.detach().cpu().numpy()
    n_frames = min(int(meta["used_frames"]), packed_np.shape[0])
    log(f"processing packed masks on CPU: shape={packed_np.shape}")
    for local_idx in range(n_frames):
        frame_idx = start + local_idx
        for obj_idx in range(packed_np.shape[1]):
            unpacked = np.unpackbits(packed_np[local_idx, obj_idx], axis=-1, bitorder="little")
            mask = unpacked.astype(bool)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            bbox = mask_to_bbox(np, mask, min_area)
            if bbox is None:
                continue
            rows.append({
                "frame_idx": frame_idx,
                "local_frame_idx": local_idx,
                "object_idx": obj_idx,
                "track_id": obj_idx + 1,
                "class_id": class_id,
                "bbox_xyxy": bbox,
            })
    return rows


def rows_from_dense_masks(torch, np, cv2, masks, meta, min_area: int, class_id: int) -> List[Dict[str, Any]]:
    rows = []
    h, w = int(meta["height"]), int(meta["width"])
    start = int(meta["start_frame"])
    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 3:
        masks_np = masks_np[:, None, :, :]
    if masks_np.ndim == 5:
        masks_np = masks_np[:, 0]
    for local_idx in range(min(int(meta["used_frames"]), masks_np.shape[0])):
        frame_idx = start + local_idx
        for obj_idx in range(masks_np.shape[1]):
            mask = masks_np[local_idx, obj_idx]
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            bbox = mask_to_bbox(np, mask > 0, min_area)
            if bbox is None:
                continue
            rows.append({
                "frame_idx": frame_idx,
                "local_frame_idx": local_idx,
                "object_idx": obj_idx,
                "track_id": obj_idx + 1,
                "class_id": class_id,
                "bbox_xyxy": bbox,
            })
    return rows


def rows_from_result(torch, np, cv2, result, meta, min_area: int, class_id: int) -> List[Dict[str, Any]]:
    if isinstance(result, dict):
        if result.get("packed_masks") is not None:
            return rows_from_packed_masks(np, cv2, result["packed_masks"], meta, min_area, class_id)
        for key in ("masks", "pred_masks", "pred_masks_high_res"):
            if key in result and torch.is_tensor(result[key]):
                log(f"processing dense masks from result[{key!r}] on CPU")
                return rows_from_dense_masks(torch, np, cv2, result[key], meta, min_area, class_id)
    if torch.is_tensor(result):
        log("processing dense tensor result on CPU")
        return rows_from_dense_masks(torch, np, cv2, result, meta, min_area, class_id)
    raise RuntimeError(f"Unsupported forward_video result type: {type(result)}")


def rows_to_tracking_results(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    by_track: Dict[int, List[Dict[str, Any]]] = {}
    class_by_track: Dict[int, int] = {}
    for row in rows:
        tid = int(row.get("track_id", int(row["object_idx"]) + 1))
        by_track.setdefault(tid, []).append(row)
        class_by_track.setdefault(tid, int(row.get("class_id", 0)))

    tracks = []
    for tid in sorted(by_track):
        track_rows = sorted(by_track[tid], key=lambda item: int(item["frame_idx"]))
        frames = [
            {
                "frame_id": int(row["frame_idx"]),
                "video_frame_idx": int(row["frame_idx"]),
                "bbox_xyxy": [int(v) for v in row["bbox_xyxy"]],
            }
            for row in track_rows
        ]
        tracks.append({
            "track_id": tid,
            "class_id": class_by_track[tid],
            "num_frames": len(frames),
            "start_frame_id": frames[0]["frame_id"],
            "end_frame_id": frames[-1]["frame_id"],
            "start_video_frame_idx": frames[0]["video_frame_idx"],
            "end_video_frame_idx": frames[-1]["video_frame_idx"],
            "clip_size": [0, 0],
            "clip_path": None,
            "frames": frames,
        })
    return {
        "metadata": {
            "video_path": meta.get("video_path", ""),
            "annotation_dir": None,
            "fps": float(meta["fps"]),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "frame_count": int(meta["frame_count"]),
            "frame_offset": 0,
            "num_tracks": len(tracks),
        },
        "tracks": tracks,
    }


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sam31_detections.json").write_text(json.dumps({"metadata": meta, "detections": rows}, indent=2), encoding="utf-8")
    tracking = rows_to_tracking_results(rows, meta)
    (out_dir / "tracking_results.json").write_text(json.dumps(tracking, indent=2), encoding="utf-8")
    log(f"wrote {out_dir / 'tracking_results.json'} ({len(tracking['tracks'])} tracks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3.1 assisted tracking and write annotator-compatible tracking_results.json.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help='Text prompt, e.g. "person:5, car:3".')
    input_group.add_argument("--prompt-pt", type=Path, help="Prompt pack generated by encode_sam31_prompt.py.")
    input_group.add_argument("--bbox", action="append", type=parse_bbox, help="Initial bbox: x1,y1,x2,y2. Repeat for multiple boxes.")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means track until video end.")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--preprocess-log-every", type=int, default=100)
    parser.add_argument("--new-det-thresh", type=float, default=0.5)
    parser.add_argument("--detect-interval", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--use-rect-mask", action="store_true")
    args = parser.parse_args()

    import cv2
    import numpy as np

    need_clip = args.prompt is not None
    torch, model_patcher, model, clip, device = load_sam31(args.comfy_root, args.checkpoint, args.device, need_clip=need_clip)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    images, meta = load_video_tensor(torch, cv2, np, args.video, args.start_frame, args.max_frames, device, dtype, args.preprocess_log_every)
    log(f"video: {meta}")
    log(f"images: shape={tuple(images.shape)} dtype={images.dtype} device={images.device}")

    try:
        with torch.inference_mode():
            if args.bbox:
                initial_masks = initial_masks_from_boxes(torch, model, images[0:1], args.bbox, meta, device, dtype, args.use_rect_mask)
                log(f"initial masks: shape={tuple(initial_masks.shape)}")
                result = model.forward_video(images, initial_masks=initial_masks)
            else:
                if args.prompt is not None:
                    prompt_pack = encode_prompt_pack(clip, args.prompt)
                else:
                    prompt_pack = torch.load(args.prompt_pt, map_location="cpu")
                text_prompts = []
                for cond, mask in prompt_pack["text_prompts"]:
                    cond = cond.to(device=device, dtype=dtype)
                    mask = None if mask is None else mask.to(device=device).bool()
                    text_prompts.append((cond, mask))
                max_objects = args.max_objects or int(prompt_pack.get("max_objects", len(text_prompts)))
                log(f"prompt={prompt_pack.get('prompt')} text_prompts={len(text_prompts)} max_objects={max_objects}")
                result = model.forward_video(
                    images,
                    initial_masks=None,
                    text_prompts=text_prompts,
                    new_det_thresh=args.new_det_thresh,
                    max_objects=max_objects,
                    detect_interval=args.detect_interval,
                )
        log(f"result type: {type(result)}")
        if isinstance(result, dict):
            log(f"result keys: {list(result.keys())}")
        rows = rows_from_result(torch, np, cv2, result, meta, args.min_mask_area, args.class_id)
        write_outputs(args.out_dir, rows, meta)
    finally:
        model_patcher.unpatch_model()


if __name__ == "__main__":
    main()
