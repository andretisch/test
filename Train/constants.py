"""Общие константы пайплайна Train (датасет, VTDNet, ONNX для Rockchip)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMGSZ = 640

# Multi-scale training (как в YOLO: случайный размер батча, кратный 32)
MULTISCALE_MIN = 512
MULTISCALE_MAX = 768
MULTISCALE_STEP = 32
MULTISCALE_BASE = 768  # загрузка кадров с запасом для upscale в диапазоне

DEFAULT_MODELS_PT = ROOT / "models" / "vtdnet_640.pt"

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
