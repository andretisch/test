#!/usr/bin/env python3
"""
Обучение кастомного детектора дорожного транспорта на YOLO11x.

Сейчас — заготовка: полный пайплайн обучения настроим на следующем этапе.
Перед запуском соберите датасет: python Train/create_dataset.py --source <путь>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_YAML = ROOT / "Dataset" / "data.yaml"
DEFAULT_MODEL = "yolo11x.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение YOLO11x на Dataset/ (будет реализовано на следующем шаге).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_YAML,
        help="Путь к data.yaml датасета.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Базовые веса (по умолчанию yolo11x.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data

    if not data_yaml.exists():
        print(
            "Файл датасета не найден. Сначала выполните:\n"
            "  python Train/create_dataset.py --source data/test_video.mp4",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Обучение YOLO11x пока не настроено — обсудим параметры на следующем шаге.")
    print(f"  data:  {data_yaml}")
    print(f"  model: {args.model}")
    sys.exit(0)


if __name__ == "__main__":
    main()
