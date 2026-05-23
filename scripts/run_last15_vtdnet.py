#!/usr/bin/env python3
"""Последние 15 секунд тестового видео → пайплайн с VTDNet → output/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import VehicleCounterPipeline

SOURCE = ROOT / "data" / "test_video_remote.mp4"
CLIP = ROOT / "data" / "last_15s.mp4"
CONFIG = ROOT / "config" / "demo_vtdnet_last15.yaml"
CLIP_SECONDS = 15


def extract_last_seconds(source: Path, target: Path, seconds: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        f"-{seconds}",
        "-i",
        str(source),
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(target),
    ]
    print("ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if not SOURCE.exists():
        print(f"Нет исходного видео: {SOURCE}", file=sys.stderr)
        sys.exit(1)

    if not (ROOT / "models" / "vtdnet_640.onnx").exists():
        print("Нет models/vtdnet_640.onnx — сначала обучите VTDNet.", file=sys.stderr)
        sys.exit(1)

    print(f"Вырезаем последние {CLIP_SECONDS} с из {SOURCE}")
    extract_last_seconds(SOURCE, CLIP, CLIP_SECONDS)

    config = load_config(CONFIG)
    pipeline = VehicleCounterPipeline(config)
    target = config.resolve_path(config.target_video)
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Обработка VTDNet: {CLIP} -> {target}")
    result = pipeline.run(
        source_path=CLIP,
        target_path=target,
        line_start=config.line.start,
        line_end=config.line.end,
        swap_directions=config.swap_directions,
    )

    print(f"Готово: {result.target_path}")
    print(f"  {config.labels.in_label}: {result.in_count}")
    print(f"  {config.labels.out_label}: {result.out_count}")
    print(f"  JSON: {target.with_suffix('.json')}")


if __name__ == "__main__":
    main()
