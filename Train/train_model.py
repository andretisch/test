#!/usr/bin/env python3
"""
Обучение кастомного детектора дорожного транспорта (Ultralytics YOLO11).

Формат датасета — стандартный YOLO (Dataset/data.yaml).
После обучения экспортирует ONNX 640x640, пригодный для Rockchip NPU / RKNN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from Train.constants import (
    DEFAULT_DATASET_YAML,
    DEFAULT_ONNX_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_TRAIN_WEIGHTS,
    IMGSZ,
)
from Train.export_onnx import export_for_rockchip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение YOLO11-детектора ТС и экспорт ONNX для Rockchip.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_YAML,
        help="Путь к data.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TRAIN_WEIGHTS,
        help=(
            "Базовые веса Ultralytics (yolo11n/s/m/l/x.pt). "
            "Для NPU по умолчанию yolo11s — меньше и быстрее, чем x."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80, help="Число эпох.")
    parser.add_argument("--batch", type=int, default=8, help="Размер батча.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMGSZ,
        help="Сторона входа (640 — для ONNX/RKNN).",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu, cuda:0, ...",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Каталог экспериментов Ultralytics.",
    )
    parser.add_argument("--name", type=str, default="vehicles", help="Имя run.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить последний run в project/name.",
    )
    parser.add_argument(
        "--no-export-onnx",
        action="store_true",
        help="Не экспортировать ONNX после обучения.",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Путь к выходному ONNX.",
    )
    return parser.parse_args()


def validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        print(
            "Файл датасета не найден. Сначала:\n"
            "  python3 Train/create_dataset.py --source <видео или папка>",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_root = data_yaml.parent
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    n_train = len(list(train_images.glob("*.jpg"))) if train_images.exists() else 0
    n_val = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0

    if n_train == 0:
        print(f"Нет изображений в {train_images}", file=sys.stderr)
        sys.exit(1)
    if n_val == 0:
        print(
            f"Предупреждение: нет val-изображений в {val_images}; "
            "обучение возможно, но валидация будет пустой.",
            file=sys.stderr,
        )


def write_train_summary(
    run_dir: Path,
    data_yaml: Path,
    args: argparse.Namespace,
    best_weights: Path,
    onnx_path: Path | None,
) -> Path:
    summary = {
        "data": str(data_yaml.resolve()),
        "base_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "best_weights": str(best_weights.resolve()),
        "onnx": str(onnx_path.resolve()) if onnx_path else None,
        "deploy": {
            "imgsz": args.imgsz,
            "onnx_opset": 12,
            "onnx_dynamic": False,
            "onnx_nms": False,
            "rockchip": "Конвертация: RKNN Toolkit, вход 1x3x640x640",
        },
    }
    out = run_dir / "train_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    project = args.project if args.project.is_absolute() else ROOT / args.project
    onnx_out = args.onnx_out if args.onnx_out.is_absolute() else ROOT / args.onnx_out

    validate_dataset(data_yaml)

    print(f"Датасет: {data_yaml}")
    print(f"Базовая модель: {args.model}, imgsz={args.imgsz}, epochs={args.epochs}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(project),
        name=args.name,
        resume=args.resume,
        exist_ok=not args.resume,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        # Аугментации по умолчанию Ultralytics; для видеонаблюдения обычно подходят
        mosaic=1.0,
        close_mosaic=10,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project / args.name
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights" / "last.pt"
    if not best_weights.exists():
        print(f"Не найдены веса в {run_dir / 'weights'}", file=sys.stderr)
        sys.exit(1)

    print(f"\nЛучшие веса: {best_weights}")

    onnx_path: Path | None = None
    if not args.no_export_onnx:
        print(f"Экспорт ONNX для Rockchip → {onnx_out}")
        onnx_path = export_for_rockchip(best_weights, onnx_out, imgsz=args.imgsz)
        print(f"ONNX готов: {onnx_path}")

    summary_path = write_train_summary(run_dir, data_yaml, args, best_weights, onnx_path)
    print(f"Сводка: {summary_path}")


if __name__ == "__main__":
    main()
