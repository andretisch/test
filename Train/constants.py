"""Общие константы пайплайна Train (датасет, VTDNet, ONNX для Rockchip)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMGSZ = 640

# PyTorch 2.x экспортирует стабильнее с opset 18; RKNN Toolkit 2.x обычно принимает
ONNX_OPSET = 18
ONNX_BATCH = 1

CLASS_NAMES: list[str] = ["car", "motorcycle", "bus", "truck"]
NUM_CLASSES = len(CLASS_NAMES)

DEFAULT_DATASET_YAML = ROOT / "Dataset" / "data.yaml"
DEFAULT_ONNX_PATH = ROOT / "models" / "vtdnet_640.onnx"
DEFAULT_RUNS_DIR = ROOT / "Train" / "runs"

# Только для псевдоразметки датасета (create_dataset.py), не для обучения детектора
DEFAULT_LABEL_MODEL = "yolo11x.pt"
