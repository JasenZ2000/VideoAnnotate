from __future__ import annotations

import copy
from typing import Any, Optional


def _normalize_track(track: dict[str, Any]) -> dict[str, Any]:
    frames = sorted(
        track.get("frames", []),
        key=lambda item: int(item.get("video_frame_idx", item.get("frame_id", 0))),
    )
    normalized = copy.deepcopy(track)
    normalized["frames"] = frames
    normalized["num_frames"] = len(frames)
    if frames:
        normalized["start_frame_id"] = int(frames[0].get("frame_id", 0))
        normalized["end_frame_id"] = int(frames[-1].get("frame_id", 0))
        normalized["start_video_frame_idx"] = int(frames[0].get("video_frame_idx", 0))
        normalized["end_video_frame_idx"] = int(frames[-1].get("video_frame_idx", 0))
    else:
        normalized.update({
            "start_frame_id": 0,
            "end_frame_id": 0,
            "start_video_frame_idx": 0,
            "end_video_frame_idx": 0,
        })
    normalized.setdefault("clip_size", [0, 0])
    normalized.setdefault("clip_path", None)
    return normalized


def append_tracking_results(
    base: dict[str, Any],
    additional: dict[str, Any],
    *,
    class_id_override: Optional[int] = None,
) -> dict[str, Any]:
    """Append tracks without colliding with IDs already present in *base*."""
    merged = copy.deepcopy(base)
    existing = merged.setdefault("tracks", [])
    next_track_id = max((int(track.get("track_id", 0)) for track in existing), default=0) + 1

    appended = []
    for track in sorted(additional.get("tracks", []), key=lambda item: int(item.get("track_id", 0))):
        remapped = _normalize_track(track)
        remapped["track_id"] = next_track_id
        next_track_id += 1
        if class_id_override is not None:
            remapped["class_id"] = int(class_id_override)
        appended.append(remapped)

    existing.extend(appended)
    existing.sort(key=lambda item: int(item["track_id"]))
    merged.setdefault("metadata", {})["num_tracks"] = len(existing)
    return merged
