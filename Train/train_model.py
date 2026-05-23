#!/usr/bin/env python3
"""
Обучение VTDNet — собственной anchor-free архитектуры (не YOLO).

Датасет: формат разметки YOLO (Dataset/data.yaml).
Экспорт: ONNX 1×3×640×640 для Rockchip NPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Train.constants import (
    CLASS_NAMES,
    DEFAULT_DATASET_YAML,
    DEFAULT_ONNX_PATH,
    DEFAULT_RUNS_DIR,
    IMGSZ,
    NUM_CLASSES,
)
from Train.export_onnx import export_for_rockchip
from Train.vtdnet.loss import VTDNetLoss
from Train.vtdnet.model import VTDNet, VTDNetConfig
from Train.vtdnet.targets import assign_targets
from Train.yolo_dataset import build_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение VTDNet (своя архитектура) + ONNX для Rockchip.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_YAML)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=0, help="0 надёжнее на CPU.")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--name", type=str, default="vtdnet")
    parser.add_argument("--resume", type=Path, default=None, help="Путь к checkpoint .pt")
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX_PATH)
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def train_one_epoch(
    model: VTDNet,
    loader: torch.utils.data.DataLoader,
    criterion: VTDNetLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: VTDNetConfig,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0, "pos": 0}
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        preds = model(images)
        obj_t, cls_t, box_t, pos_mask = assign_targets(targets, cfg, device)
        loss, stats = criterion(preds, obj_t, cls_t, box_t, pos_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k in totals:
            totals[k] += stats.get(k, 0)
        n_batches += 1

    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: VTDNet,
    loader: torch.utils.data.DataLoader,
    criterion: VTDNetLoss,
    device: torch.device,
    cfg: VTDNetConfig,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0, "pos": 0}
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        preds = model(images)
        obj_t, cls_t, box_t, pos_mask = assign_targets(targets, cfg, device)
        _, stats = criterion(preds, obj_t, cls_t, box_t, pos_mask)
        for k in totals:
            totals[k] += stats.get(k, 0)
        n_batches += 1

    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: VTDNet,
    cfg: VTDNetConfig,
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "class_names": CLASS_NAMES,
            "architecture": "VTDNet",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    run_dir = (args.project if args.project.is_absolute() else ROOT / args.project) / args.name
    onnx_out = args.onnx_out if args.onnx_out.is_absolute() else ROOT / args.onnx_out
    device = resolve_device(args.device)

    if not data_yaml.exists():
        print("Нет data.yaml. Сначала: python3 Train/create_dataset.py --source ...", file=sys.stderr)
        sys.exit(1)

    cfg = VTDNetConfig(num_classes=NUM_CLASSES, imgsz=args.imgsz)
    model = VTDNet(cfg).to(device)
    criterion = VTDNetLoss(num_classes=NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 0
    best_val = float("inf")
    patience_left = args.patience

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", best_val))
        print(f"Resume с {args.resume}, epoch={start_epoch}")

    train_loader, val_loader, _ = build_loaders(
        data_yaml, imgsz=args.imgsz, batch_size=args.batch, workers=args.workers
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VTDNet — собственная архитектура (не YOLO)")
    print(f"  параметров: {n_params:,}")
    print(f"  классы: {CLASS_NAMES}")
    print(f"  imgsz: {args.imgsz}, device: {device}")
    print(f"  train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg)
        val_stats = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_stats['loss']:.4f} | val loss {val_stats['loss']:.4f} | "
            f"pos {train_stats['pos']:.1f}"
        )

        last_path = weights_dir / "last.pt"
        save_checkpoint(last_path, model, cfg, epoch, val_stats["loss"])

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            patience_left = args.patience
            best_path = weights_dir / "best.pt"
            save_checkpoint(best_path, model, cfg, epoch, val_stats["loss"])
            print(f"  → best.pt (val_loss={best_val:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        best_weights = weights_dir / "last.pt"

    summary = {
        "architecture": "VTDNet",
        "description": "Anchor-free DW-CSP backbone + FPN, 3 scale heads",
        "class_names": CLASS_NAMES,
        "imgsz": args.imgsz,
        "best_weights": str(best_weights.resolve()),
        "deploy": {
            "onnx_input": "1x3x640x640",
            "onnx_output": "1xNx(5+nc) raw logits",
            "postprocess": "decode_predictions + NMS на CPU (см. Train/vtdnet/decode.py)",
            "rockchip": "RKNN Toolkit, opset 12, static shape",
        },
    }
    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    onnx_path = None
    if not args.no_export_onnx and best_weights.exists():
        print(f"Экспорт ONNX → {onnx_out}")
        onnx_path = export_for_rockchip(best_weights, onnx_out, imgsz=args.imgsz)
        summary["onnx"] = str(onnx_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ONNX: {onnx_path}")

    print(f"Готово. Веса: {best_weights}")


if __name__ == "__main__":
    main()
