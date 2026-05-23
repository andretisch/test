#!/usr/bin/env python3
"""
Экспорт обученного YOLO в ONNX для Rockchip NPU (RKNN).

Фиксированный вход 1x3x640x640, FP32, без dynamic axes и без встроенного NMS
(NMS удобнее выполнять на CPU при конвертации в RKNN или в постобработке).
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
    CLASS_NAMES,
    DEFAULT_ONNX_PATH,
    IMGSZ,
    NUM_CLASSES,
    ONNX_BATCH,
    ONNX_OPSET,
)


def export_for_rockchip(
    weights: Path,
    onnx_path: Path,
    imgsz: int = IMGSZ,
) -> Path:
    weights = weights.resolve()
    onnx_path = onnx_path.resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=ONNX_OPSET,
        simplify=True,
        dynamic=False,
        half=False,
        batch=ONNX_BATCH,
        nms=False,
        device="cpu",
    )

    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported_path.read_bytes())
        if exported_path.parent.resolve() == Path.cwd().resolve():
            exported_path.unlink(missing_ok=True)

    meta = {
        "format": "onnx",
        "imgsz": imgsz,
        "batch": ONNX_BATCH,
        "opset": ONNX_OPSET,
        "nms": False,
        "dynamic": False,
        "half": False,
        "classes": CLASS_NAMES,
        "nc": NUM_CLASSES,
        "weights": str(weights),
        "onnx": str(onnx_path),
        "rockchip_notes": (
            "Конвертируйте в RKNN с тем же imgsz=640. "
            "При необходимости квантуйте INT8 на репрезентативной выборке кадров."
        ),
    }
    meta_path = onnx_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return onnx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт YOLO → ONNX (Rockchip NPU).")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Путь к best.pt после обучения.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=f"Куда сохранить ONNX (по умолчанию: {DEFAULT_ONNX_PATH}).",
    )
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Сторона входа.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    output = args.output if args.output.is_absolute() else ROOT / args.output

    if not weights.exists():
        print(f"Веса не найдены: {weights}", file=sys.stderr)
        sys.exit(1)

    onnx_path = export_for_rockchip(weights, output, imgsz=args.imgsz)
    print(f"ONNX: {onnx_path}")
    print(f"Meta: {onnx_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
