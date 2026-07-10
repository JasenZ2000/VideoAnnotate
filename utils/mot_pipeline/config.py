from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "tracking": {
        "method": "sparse_track",
        "iou_match": 0.3,
        "max_missed": 15,
        "disable_kalman": False,
        "class_agnostic": False,
        "score_high": 0.5,
        "score_low": 0.1,
        "new_track_score": 0.6,
        "match_iou": 0.3,
        "low_iou_match": 0.15,
        "recover_iou_match": 0.05,
        "velocity_weight": 0.2,
        "velocity_min_score": 0.6,
        "height_weight": 0.08,
        "score_weight": 0.06,
        "fuse_score": True,
        "cbiou_small_buffer": 0.08,
        "cbiou_large_buffer": 0.28,
        "cbiou_second_iou": 0.15,
        "sparse_depth_bins": 4,
        "sparse_neighbor_bins": 1,
        "sparse_depth_weight": 0.15,
        "sparse_height_weight": 0.5,
        "sparse_cross_depth_iou": 0.75,
    },
    "fusion": {
        "method": "bidirectional_iou_all_pairs",
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
            "frame_area_change_ratio": 2.5,
            "frame_area_change_abs": 800.0,
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
        "comfy_root": "/opt/ComfyUI",
        "checkpoint": "/models/sam3.1_multiplex_fp16.safetensors",
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
    "locateanything": {
        "server_url": "",
        "video_transfer": "path",
        "local_path_prefix": "",
        "remote_path_prefix": "",
        "request_timeout": 30,
        "download_timeout": 300,
        "poll_interval": 5,
        "sftp_host": "",
        "sftp_port": 22,
        "sftp_username": "",
        "sftp_password_env": "LOCANY_SFTP_PASSWORD",
        "sftp_key_path": "",
        "sftp_remote_dir": "",
        "sftp_reuse_existing": True,
        "device": "cuda",
        "dtype": "bf16",
        "task": "ground_multi",
        "class_id": 0,
        "score": 1.0,
        "frame_offset": 1,
        "resize_long_edge": 1024,
        "generation_mode": "slow",
        "max_new_tokens": 512,
        "temperature": 0.0,
        "use_cache": True,
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
