#!/usr/bin/env python3
"""Quick smoke test: pipeline on a short clip from data/test_video.mp4."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
TEST_VIDEO = DATA / "test_video.mp4"
SHORT = DATA / "short.mp4"


def ensure_short_clip() -> Path:
    if not TEST_VIDEO.exists():
        raise FileNotFoundError(
            f"Положите тестовое видео: {TEST_VIDEO}"
        )
    if SHORT.exists():
        return SHORT
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(TEST_VIDEO),
            "-t",
            "2",
            "-c",
            "copy",
            str(SHORT),
        ],
        check=True,
        capture_output=True,
    )
    return SHORT


def main() -> None:
    from src.config import load_config
    from src.pipeline import VehicleCounterPipeline

    clip = ensure_short_clip()
    info = __import__("supervision", fromlist=["sv"]).VideoInfo.from_video_path(str(clip))
    mid_y = info.height // 2

    cfg = load_config()
    out = ROOT / "output" / "smoke_result.mp4"
    pipeline = VehicleCounterPipeline(cfg)
    result = pipeline.run(
        clip,
        out,
        line_start=(0, mid_y),
        line_end=(info.width, mid_y),
    )
    print("OK")
    print(f"  in={result.in_count} out={result.out_count}")
    print(f"  video={result.target_path}")


if __name__ == "__main__":
    main()
