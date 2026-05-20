"""Ensure ONNX model exists and matches config.imgsz (auto-export if needed)."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from src.config import AppConfig

logger = logging.getLogger(__name__)


def _pt_name_from_onnx(onnx_path: Path) -> str:
    stem = onnx_path.stem
    if stem.endswith(".onnx"):
        stem = Path(stem).stem
    return f"{stem}.pt"


def get_onnx_input_imgsz(onnx_path: Path) -> int | None:
    """Read H/W from ONNX input tensor (YOLO: NCHW, square)."""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
    except Exception as exc:
        logger.warning("Не удалось прочитать ONNX %s: %s", onnx_path, exc)
        return None

    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        dims = []
        for d in shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                return None
        if len(dims) == 4 and dims[1] == 3:
            h, w = dims[2], dims[3]
            if h == w and h > 0:
                return h
    return None


def export_onnx(pt_name: str, onnx_path: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Экспорт %s -> %s (imgsz=%s)", pt_name, onnx_path, imgsz)

    try:
        model = YOLO(pt_name)
        exported = model.export(
            format="onnx",
            simplify=True,
            opset=12,
            imgsz=imgsz,
        )
    except Exception:
        logger.error("Ошибка экспорта модели %s", pt_name)
        traceback.print_exc()
        raise

    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported_path.read_bytes())
        if exported_path.parent.resolve() == Path.cwd().resolve():
            exported_path.unlink(missing_ok=True)

    actual = get_onnx_input_imgsz(onnx_path)
    logger.info("Готово: %s (ONNX imgsz=%s)", onnx_path, actual)
    return onnx_path


def ensure_onnx_model(config: AppConfig) -> Path:
    """
    Возвращает путь к ONNX. Если файла нет или imgsz не совпадает с конфигом — экспорт.
    """
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(message)s",
            stream=sys.stderr,
        )

    onnx_path = config.resolve_path(config.model_path)
    expected = int(config.imgsz)

    if not onnx_path.suffix.lower() == ".onnx":
        raise ValueError(
            f"model_path должен указывать на .onnx, получено: {onnx_path}"
        )

    need_export = False
    reason = ""

    if not onnx_path.exists():
        need_export = True
        reason = f"файл не найден: {onnx_path}"
    else:
        actual = get_onnx_input_imgsz(onnx_path)
        if actual is None:
            need_export = True
            reason = "не удалось определить imgsz в ONNX"
        elif actual != expected:
            need_export = True
            reason = f"imgsz в ONNX={actual}, в конфиге={expected}"

    if need_export:
        logger.warning("Пересборка ONNX: %s", reason)
        pt_name = _pt_name_from_onnx(onnx_path)
        try:
            export_onnx(pt_name, onnx_path, expected)
        except Exception as exc:
            logger.error(
                "Не удалось получить модель. Проверьте имя в model_path "
                "(например models/yolov8s.onnx -> скачается yolov8s.pt). "
                "Ошибка: %s",
                exc,
            )
            raise
    else:
        logger.info("Модель OK: %s (imgsz=%s)", onnx_path, expected)

    return onnx_path
