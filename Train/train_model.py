#!/usr/bin/env python3
"""
Обучение VTDNet — собственной anchor-free архитектуры (не YOLO).

Multi-scale training: случайный размер батча 512–768 (шаг 32), val/export — 640.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Train.constants import (
    CLASS_NAMES,
    DEFAULT_DATASET_YAML,
    DEFAULT_MODELS_PT,
    DEFAULT_ONNX_PATH,
    DEFAULT_RUNS_DIR,
    IMGSZ,
    MULTISCALE_BASE,
    MULTISCALE_MAX,
    MULTISCALE_MIN,
    NUM_CLASSES,
)
from Train.export_onnx import export_for_rockchip
from Train.history import append_history_row
from Train.multiscale import cfg_for_imgsz, resize_batch, sample_train_imgsz
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
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Размер val и ONNX.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--name", type=str, default="vtdnet")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--no-multiscale",
        action="store_true",
        help="Отключить multi-scale (только фикс. imgsz).",
    )
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument(
        "--models-pt",
        type=Path,
        default=DEFAULT_MODELS_PT,
        help="Куда скопировать лучшие веса для GitHub.",
    )
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
    base_cfg: VTDNetConfig,
    multiscale: bool,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0, "pos": 0}
    imgsz_sum = 0
    n_batches = 0

    for images, targets in loader:
        batch_imgsz = sample_train_imgsz() if multiscale else base_cfg.imgsz
        images = resize_batch(images.to(device), batch_imgsz)
        targets = [t.to(device) for t in targets]
        batch_cfg = cfg_for_imgsz(base_cfg, batch_imgsz)

        preds = model(images)
        obj_t, cls_t, box_t, pos_mask = assign_targets(targets, batch_cfg, device)
        loss, stats = criterion(preds, obj_t, cls_t, box_t, pos_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k in totals:
            totals[k] += stats.get(k, 0)
        imgsz_sum += batch_imgsz
        n_batches += 1

    if n_batches == 0:
        return totals
    out = {k: v / n_batches for k, v in totals.items()}
    out["train_imgsz_avg"] = imgsz_sum / n_batches
    return out


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
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "cfg": asdict(cfg),
        "class_names": CLASS_NAMES,
        "architecture": "VTDNet",
        "multiscale_train": [MULTISCALE_MIN, MULTISCALE_MAX],
        "export_imgsz": IMGSZ,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def publish_to_models(
    best_weights: Path,
    models_pt: Path,
    onnx_out: Path,
    imgsz: int,
) -> tuple[Path, Path | None]:
    models_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, models_pt)

    onnx_path = export_for_rockchip(best_weights, onnx_out, imgsz=imgsz)
    return models_pt, onnx_path


def main() -> None:
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    run_dir = (args.project if args.project.is_absolute() else ROOT / args.project) / args.name
    onnx_out = args.onnx_out if args.onnx_out.is_absolute() else ROOT / args.onnx_out
    models_pt = args.models_pt if args.models_pt.is_absolute() else ROOT / args.models_pt
    device = resolve_device(args.device)
    multiscale = not args.no_multiscale

    if not data_yaml.exists():
        print("Нет data.yaml. Сначала: python3 Train/create_dataset.py --source ...", file=sys.stderr)
        sys.exit(1)

    val_cfg = VTDNetConfig(num_classes=NUM_CLASSES, imgsz=args.imgsz)
    model = VTDNet(val_cfg).to(device)
    criterion = VTDNetLoss(num_classes=NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 0
    best_val = float("inf")
    patience_left = args.patience
    history_rows: list[dict[str, Any]] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", best_val))
        hist_path = run_dir / "history.json"
        if hist_path.exists():
            history_rows = json.loads(hist_path.read_text(encoding="utf-8"))
        print(f"Resume с {args.resume}, epoch={start_epoch}")

    train_base = MULTISCALE_BASE if multiscale else args.imgsz
    train_loader, _, _ = build_loaders(
        data_yaml,
        imgsz=train_base,
        batch_size=args.batch,
        workers=args.workers,
    )
    _, val_loader, _ = build_loaders(
        data_yaml,
        imgsz=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print("VTDNet — собственная архитектура (не YOLO)")
    print(f"  параметров: {n_params:,}")
    print(f"  классы: {CLASS_NAMES}")
    print(f"  multi-scale train: {multiscale} [{MULTISCALE_MIN}..{MULTISCALE_MAX}]")
    print(f"  val / ONNX imgsz: {args.imgsz}, device: {device}")
    print(f"  train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            val_cfg,
            multiscale=multiscale,
        )
        val_stats = validate(model, val_loader, criterion, device, val_cfg)
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": round(train_stats["loss"], 6),
            "train_obj": round(train_stats["obj"], 6),
            "train_cls": round(train_stats["cls"], 6),
            "train_box": round(train_stats["box"], 6),
            "train_pos": round(train_stats["pos"], 2),
            "train_imgsz_avg": round(train_stats.get("train_imgsz_avg", args.imgsz), 1),
            "val_loss": round(val_stats["loss"], 6),
            "val_obj": round(val_stats["obj"], 6),
            "val_cls": round(val_stats["cls"], 6),
            "val_box": round(val_stats["box"], 6),
            "val_pos": round(val_stats["pos"], 2),
            "val_imgsz": args.imgsz,
        }
        history_rows = append_history_row(run_dir, row, history_rows)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train {train_stats['loss']:.4f} (imgsz≈{train_stats.get('train_imgsz_avg', args.imgsz):.0f}) | "
            f"val {val_stats['loss']:.4f}"
        )

        last_path = weights_dir / "last.pt"
        save_checkpoint(last_path, model, val_cfg, epoch, val_stats["loss"])

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            patience_left = args.patience
            best_path = weights_dir / "best.pt"
            save_checkpoint(best_path, model, val_cfg, epoch, val_stats["loss"])
            print(f"  → best.pt (val_loss={best_val:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        best_weights = weights_dir / "last.pt"

    summary: dict[str, Any] = {
        "architecture": "VTDNet",
        "epochs_run": len(history_rows),
        "multiscale_train": [MULTISCALE_MIN, MULTISCALE_MAX] if multiscale else None,
        "val_imgsz": args.imgsz,
        "class_names": CLASS_NAMES,
        "best_val_loss": best_val,
        "best_weights": str(best_weights.resolve()),
        "history_json": str((run_dir / "history.json").resolve()),
        "history_csv": str((run_dir / "history.csv").resolve()),
    }

    onnx_path = None
    if best_weights.exists():
        if not args.no_export_onnx:
            print(f"Публикация в models/: {models_pt}, {onnx_out}")
            pub_pt, onnx_path = publish_to_models(
                best_weights, models_pt, onnx_out, imgsz=args.imgsz
            )
            summary["models_pt"] = str(pub_pt.resolve())
            summary["onnx"] = str(onnx_path.resolve()) if onnx_path else None
        else:
            shutil.copy2(best_weights, models_pt)
            summary["models_pt"] = str(models_pt.resolve())

    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"История: {run_dir / 'history.json'}")
    print(f"Сводка: {summary_path}")
    print(f"Готово. Веса: {best_weights}")


if __name__ == "__main__":
    main()
