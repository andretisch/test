#!/usr/bin/env python3
"""Опционально: принудительный экспорт по текущему config/default.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.model_loader import export_onnx, _pt_name_from_onnx


def main() -> None:
    config = load_config()
    onnx_path = config.resolve_path(config.model_path)
    pt_name = _pt_name_from_onnx(onnx_path)
    export_onnx(pt_name, onnx_path, config.imgsz)
    print(f"OK: {onnx_path}")


if __name__ == "__main__":
    main()
