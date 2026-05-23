"""Общие константы пайплайна Train (датасет, обучение, ONNX для Rockchip)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Универсальный вход для инференса и RKNN-конвертации
IMGSZ = 640

# ONNX: фиксированный NCHW, opset 12 — совместим с ONNX Runtime и RKNN Toolkit
ONNX_OPSET = 12
ONNX_BATCH = 1

CLASS_NAMES: list[str] = ["car", "motorcycle", "bus", "truck"]
NUM_CLASSES = len(CLASS_NAMES)

DEFAULT_DATASET_YAML = ROOT / "Dataset" / "data.yaml"
DEFAULT_ONNX_PATH = ROOT / "models" / "vehicles_640.onnx"
DEFAULT_RUNS_DIR = ROOT / "Train" / "runs"

# yolo11s — баланс точности и размера для NPU; для разметки датасета — yolo11x
DEFAULT_TRAIN_WEIGHTS = "yolo11s.pt"
