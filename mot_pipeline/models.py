from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BBox = List[float]


@dataclass
class Detection:
    frame_id: int
    video_frame_idx: int
    class_id: int
    bbox: BBox


@dataclass
class FrameDetections:
    frame_id: int
    video_frame_idx: int
    txt_path: Path
    detections: List[Detection] = field(default_factory=list)


@dataclass
class Tracklet:
    local_track_id: int
    direction: str
    class_id: int
    frames: Dict[int, BBox]
    video_frames: Dict[int, int]
    frame_classes: Dict[int, int]
    hits: int
    misses: int
    jitter: float
    quality: float

    @property
    def start_frame(self) -> int:
        return min(self.frames)

    @property
    def end_frame(self) -> int:
        return max(self.frames)

    @property
    def length(self) -> int:
        return len(self.frames)


@dataclass
class FinalTrack:
    track_id: int
    class_id: int
    frames: Dict[int, BBox]
    video_frames: Dict[int, int]
    clip_size: Tuple[int, int] = (0, 0)
    clip_path: Optional[Path] = None

    @property
    def start_frame(self) -> int:
        return min(self.frames)

    @property
    def end_frame(self) -> int:
        return max(self.frames)

    @property
    def length(self) -> int:
        return len(self.frames)

