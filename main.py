#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mot_pipeline.config import load_config
from mot_pipeline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    # 最小 CLI：只暴露输入、输出和可选配置文件。
    parser = argparse.ArgumentParser(
        description="Run dual-pass MOT and extract one cropped clip per track."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--label-dir",
        required=True,
        help="Directory containing YOLO txt files, one per frame.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file overriding default parameters.",
    )
    return parser.parse_args()


def main() -> None:
    # 入口只负责拼装参数和配置，具体流程交给 pipeline 模块。
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(
        video_path=Path(args.video),
        ann_dir=Path(args.label_dir),
        out_dir=Path(args.out_dir),
        config=config,
    )


if __name__ == "__main__":
    main()
