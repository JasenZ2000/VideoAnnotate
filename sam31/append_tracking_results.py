from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def max_track_id(payload: Dict[str, Any]) -> int:
    tracks = payload.get("tracks", [])
    return max((int(track["track_id"]) for track in tracks), default=0)


def normalize_track(track: Dict[str, Any]) -> Dict[str, Any]:
    frames = sorted(track.get("frames", []), key=lambda item: int(item.get("video_frame_idx", item.get("frame_id", 0))))
    out = copy.deepcopy(track)
    out["frames"] = frames
    out["num_frames"] = len(frames)
    if frames:
        out["start_frame_id"] = int(frames[0]["frame_id"])
        out["end_frame_id"] = int(frames[-1]["frame_id"])
        out["start_video_frame_idx"] = int(frames[0]["video_frame_idx"])
        out["end_video_frame_idx"] = int(frames[-1]["video_frame_idx"])
    else:
        out["start_frame_id"] = 0
        out["end_frame_id"] = 0
        out["start_video_frame_idx"] = 0
        out["end_video_frame_idx"] = 0
    out.setdefault("clip_size", [0, 0])
    out.setdefault("clip_path", None)
    return out


def append_tracking_results(base: Dict[str, Any], new: Dict[str, Any], class_id_override: int | None = None) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    merged.setdefault("tracks", [])
    offset = max_track_id(merged)

    appended = []
    for track in new.get("tracks", []):
        remapped = normalize_track(track)
        remapped["track_id"] = offset + int(track["track_id"])
        if class_id_override is not None:
            remapped["class_id"] = int(class_id_override)
        appended.append(remapped)

    merged["tracks"].extend(appended)
    merged["tracks"] = sorted(merged["tracks"], key=lambda item: int(item["track_id"]))
    merged.setdefault("metadata", {})
    merged["metadata"]["num_tracks"] = len(merged["tracks"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one tracking_results.json into another while avoiding track_id conflicts.")
    parser.add_argument("--base", required=True, type=Path, help="Existing tracking_results.json.")
    parser.add_argument("--new", required=True, type=Path, help="New SAM31 tracking_results.json to append.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-id", type=int, help="Optional class_id override for appended tracks.")
    args = parser.parse_args()

    base = load_json(args.base)
    new = load_json(args.new)
    merged = append_tracking_results(base, new, class_id_override=args.class_id)
    write_json(args.output, merged)
    print(
        f"wrote {args.output} "
        f"base_tracks={len(base.get('tracks', []))} "
        f"new_tracks={len(new.get('tracks', []))} "
        f"merged_tracks={len(merged.get('tracks', []))}"
    )


if __name__ == "__main__":
    main()
