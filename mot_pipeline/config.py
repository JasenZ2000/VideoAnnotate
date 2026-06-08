from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "tracking": {
        "method": "iou_kalman",
        "iou_match": 0.3,
        "max_missed": 15,
        "disable_kalman": False,
    },
    "fusion": {
        "method": "bidirectional_iou",
        "iou_fuse": 0.5,
        "min_track_len": 10,
        "smooth_window": 5,
    },
    "clips": {
        "pad_frames": 10,
        "crop_margin": 1.2,
        "crop_min_size": 128,
        "codec": "mp4v",
        "overview_filename": "tracking_overview.mp4",
        "overview_box_thickness": 2,
        "overview_font_scale": 0.7,
    },
    "exports": {
        "export_yolo_from_tracking": True,
        "yolo_dirname": "tracking_yolo",
        "export_label_studio": True,
        "label_studio_filename": "tracking_results.label_studio.json",
        "label_studio_local_files_prefix": "/data/local-files/?d=",
        "class_labels": {
            "0": "person",
            "1": "car",
        },
    },
    "annotator": {
        "frame_buffer_ahead": 30,
        "frame_batch_size": 15,
        "frame_cache_limit": 80,
        "frame_batch_max": 30,
        "annotation_buffer_ahead": 60,
        "annotation_batch_size": 60,
        "annotation_cache_limit": 300,
        "annotation_batch_max": 200,
    },
    "quality_control": {
        "area_anomaly": {
            "enabled": True,
            "high_area_ratio": 3.0,
            "low_area_ratio": 0.25,
            "robust_z_threshold": 6.0,
            "min_track_frames": 8,
            "min_area": 1.0,
            "max_gap": 1,
            "filename": "tracking_area_anomalies.json",
        },
    },
    "sam31": {
        "runner": "remote",
        "server_url": "",
        "video_transfer": "path",
        "local_path_prefix": "",
        "remote_path_prefix": "",
        "request_timeout": 30,
        "poll_interval": 2,
        "sftp_host": "",
        "sftp_port": 22,
        "sftp_username": "",
        "sftp_password_env": "SAM31_SFTP_PASSWORD",
        "sftp_key_path": "",
        "sftp_remote_dir": "",
        "sftp_reuse_existing": True,
        "comfy_root": "/data2/DET_Group/ZZS/generate/update/ComfyUI",
        "checkpoint": "/data2/DET_Group/ZZS/my_sam3/sam3.1_multiplex_fp16.safetensors",
        "device": "cuda",
        "dtype": "fp16",
        "min_mask_area": 64,
        "use_rect_mask": False,
        "postprocess_spikes": True,
        "spike_area_ratio": 4.0,
        "spike_size_ratio": 3.0,
        "spike_history": 10,
        "spike_min_history": 3,
        "spike_max_run": 10,
    },
}


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    # 递归合并用户配置，只覆盖显式提供的字段，未提供的参数保持默认值。
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | None) -> Dict[str, Any]:
    # 先加载默认配置，再按需叠加外部 JSON，保证所有模块都有完整参数。
    config = deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return config
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    return deep_update(config, user_config)
