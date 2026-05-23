#!/usr/bin/env python3
"""
Экспорт VTDNet в ONNX для Rockchip NPU (RKNN).

Вход:  1×3×640×640
Выход: 1×N×(5+nc) — сырые логиты; decode + NMS на CPU/хосте.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Train.constants import (
    CLASS_NAMES,
    DEFAULT_ONNX_PATH,
    IMGSZ,
    NUM_CLASSES,
    ONNX_OPSET,
)
from Train.vtdnet.model import VTDNet, VTDNetConfig


class VTDNetExportWrapper(torch.nn.Module):
    def __init__(self, model: VTDNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_export(x)


def load_checkpoint(weights: Path, device: torch.device) -> tuple[VTDNet, VTDNetConfig]:
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg") or {"num_classes": NUM_CLASSES, "imgsz": IMGSZ}
    cfg = VTDNetConfig(
        num_classes=int(cfg_dict.get("num_classes", NUM_CLASSES)),
        imgsz=int(cfg_dict.get("imgsz", IMGSZ)),
    )
    model = VTDNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def export_for_rockchip(
    weights: Path,
    onnx_path: Path,
    imgsz: int = IMGSZ,
) -> Path:
    device = torch.device("cpu")
    model, cfg = load_checkpoint(weights, device)
    if cfg.imgsz != imgsz:
        print(f"Предупреждение: checkpoint imgsz={cfg.imgsz}, экспорт imgsz={imgsz}")

    wrapper = VTDNetExportWrapper(model).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    onnx_path = onnx_path.resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["predictions"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    n_cells = sum((imgsz // s) ** 2 for s in cfg.strides)
    out_dim = 5 + cfg.num_classes

    meta = {
        "architecture": "VTDNet",
        "format": "onnx",
        "imgsz": imgsz,
        "input": {"name": "images", "shape": [1, 3, imgsz, imgsz], "dtype": "float32"},
        "output": {
            "name": "predictions",
            "shape": [1, n_cells, out_dim],
            "layout": "obj, cx, cy, w, h, class_logits...",
        },
        "strides": list(cfg.strides),
        "classes": CLASS_NAMES,
        "nc": cfg.num_classes,
        "opset": ONNX_OPSET,
        "dynamic": False,
        "postprocess": "sigmoid + decode_predictions + NMS (Train/vtdnet/decode.py)",
        "rockchip_notes": (
            "Конвертация RKNN: статический вход 640. "
            "NMS рекомендуется на ARM CPU после NPU-инференса."
        ),
        "weights": str(weights.resolve()),
    }
    meta_path = onnx_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return onnx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт VTDNet → ONNX (Rockchip).")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
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
