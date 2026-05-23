#!/usr/bin/env python3
"""Сравнение YOLO11x и VTDNet на последних 15 с тестового видео."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.vtdnet_detector import VTDNetDetector

CLIP = ROOT / "data" / "last_15s.mp4"
COCO_VEHICLES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}


def count_yolo11x(
    video_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> dict:
    model = YOLO("yolo11x.pt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    per_frame: list[dict] = []
    total_raw = 0
    total_vehicles = 0
    by_class: dict[str, int] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(
            frame,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device="cpu",
        )[0]
        dets = sv.Detections.from_ultralytics(results)
        n_raw = len(dets)

        if n_raw > 0:
            mask = np.isin(dets.class_id, list(COCO_VEHICLES))
            veh = dets[mask]
        else:
            veh = dets

        n_veh = len(veh)
        total_raw += n_raw
        total_vehicles += n_veh

        frame_classes: dict[str, int] = {}
        for cid in veh.class_id:
            name = results.names[int(cid)]
            frame_classes[name] = frame_classes.get(name, 0) + 1
            by_class[name] = by_class.get(name, 0) + 1

        per_frame.append(
            {
                "frame": frame_idx,
                "raw": n_raw,
                "vehicles": n_veh,
                "classes": frame_classes,
            }
        )
        frame_idx += 1

    cap.release()
    n_frames = frame_idx
    return {
        "model": "yolo11x",
        "conf": conf,
        "frames": n_frames,
        "total_detections_all_classes": total_raw,
        "total_vehicle_detections": total_vehicles,
        "avg_vehicles_per_frame": total_vehicles / n_frames if n_frames else 0,
        "frames_with_vehicles": sum(1 for p in per_frame if p["vehicles"] > 0),
        "by_class": by_class,
        "per_frame": per_frame,
    }


def count_vtdnet(
    video_path: Path,
    onnx_path: Path,
    conf: float,
    iou: float,
    imgsz: int = 640,
) -> dict:
    det = VTDNetDetector(onnx_path, conf=conf, iou=iou, imgsz=imgsz)
    cap = cv2.VideoCapture(str(video_path))

    per_frame: list[dict] = []
    total = 0
    by_class: dict[str, int] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = det.predict(frame)
        n = len(dets)
        total += n

        frame_classes: dict[str, int] = {}
        for cid in dets.class_id:
            name = CLASS_NAMES.get(int(cid), str(int(cid)))
            frame_classes[name] = frame_classes.get(name, 0) + 1
            by_class[name] = by_class.get(name, 0) + 1

        per_frame.append(
            {
                "frame": frame_idx,
                "vehicles": n,
                "classes": frame_classes,
            }
        )
        frame_idx += 1

    cap.release()
    n_frames = frame_idx
    return {
        "model": "vtdnet_640",
        "conf": conf,
        "frames": n_frames,
        "total_vehicle_detections": total,
        "avg_vehicles_per_frame": total / n_frames if n_frames else 0,
        "frames_with_vehicles": sum(1 for p in per_frame if p["vehicles"] > 0),
        "by_class": by_class,
        "per_frame": per_frame,
    }


def estimate_unique_vehicles(per_frame: list[dict], key: str = "vehicles") -> int:
    """Грубая оценка: макс. число машин на одном кадре (нижняя граница присутствия)."""
    if not per_frame:
        return 0
    return max(p[key] for p in per_frame)


def main() -> None:
    if not CLIP.exists():
        print(f"Нет клипа: {CLIP}. Запустите scripts/run_last15_vtdnet.py", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(ROOT / "config" / "demo_vtdnet_last15.yaml")
    onnx = cfg.resolve_path(cfg.model_path)

    print("Запуск YOLO11x (COCO ТС, conf=0.25)...")
    yolo = count_yolo11x(CLIP, conf=0.25, iou=0.45, imgsz=640)

    print("Запуск VTDNet (conf=0.03)...")
    vtd = count_vtdnet(CLIP, onnx, conf=cfg.vtdnet_confidence, iou=cfg.iou, imgsz=640)

    # также VTDNet со старым порогом 0.25 для сравнения
    print("Запуск VTDNet (conf=0.25, старый порог)...")
    vtd_old = count_vtdnet(CLIP, onnx, conf=0.25, iou=cfg.iou, imgsz=640)

    report = {
        "clip": str(CLIP),
        "duration_sec": 15,
        "frames": yolo["frames"],
        "yolo11x": {
            "conf": yolo["conf"],
            "total_vehicle_box_detections": yolo["total_vehicle_detections"],
            "avg_per_frame": round(yolo["avg_vehicles_per_frame"], 2),
            "frames_with_detections": yolo["frames_with_vehicles"],
            "max_on_single_frame": estimate_unique_vehicles(yolo["per_frame"], "vehicles"),
            "by_class": yolo["by_class"],
        },
        "vtdnet_conf_003": {
            "conf": vtd["conf"],
            "total_vehicle_box_detections": vtd["total_vehicle_detections"],
            "avg_per_frame": round(vtd["avg_vehicles_per_frame"], 2),
            "frames_with_detections": vtd["frames_with_vehicles"],
            "max_on_single_frame": estimate_unique_vehicles(vtd["per_frame"], "vehicles"),
            "by_class": vtd["by_class"],
        },
        "vtdnet_conf_025_old": {
            "conf": vtd_old["conf"],
            "total_vehicle_box_detections": vtd_old["total_vehicle_detections"],
            "avg_per_frame": round(vtd_old["avg_vehicles_per_frame"], 2),
            "frames_with_detections": vtd_old["frames_with_vehicles"],
            "max_on_single_frame": estimate_unique_vehicles(vtd_old["per_frame"], "vehicles"),
        },
        "interpretation": (
            "total_vehicle_box_detections — сумма bbox по всем кадрам (одна машина "
            "считается многократно). max_on_single_frame — сколько машин видно "
            "одновременно на самом «загруженном» кадре."
        ),
    }

    out = ROOT / "output" / "compare_last15_yolo_vs_vtdnet.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ: последние 15 с")
    print("=" * 60)
    print(f"Кадров: {yolo['frames']}")
    print()
    print("YOLO11x (conf=0.25, классы COCO 2,3,5,7):")
    print(f"  Сумма детекций ТС:     {yolo['total_vehicle_detections']}")
    print(f"  В среднем на кадр:     {yolo['avg_vehicles_per_frame']:.2f}")
    print(f"  Кадров с детекциями:   {yolo['frames_with_vehicles']}/{yolo['frames']}")
    print(f"  Макс. на одном кадре:  {report['yolo11x']['max_on_single_frame']}")
    print(f"  По классам:            {yolo['by_class']}")
    print()
    print("VTDNet (conf=0.03, после исправления):")
    print(f"  Сумма детекций ТС:     {vtd['total_vehicle_detections']}")
    print(f"  В среднем на кадр:     {vtd['avg_vehicles_per_frame']:.2f}")
    print(f"  Кадров с детекциями:   {vtd['frames_with_vehicles']}/{vtd['frames']}")
    print(f"  Макс. на одном кадре:  {report['vtdnet_conf_003']['max_on_single_frame']}")
    print(f"  По классам:            {vtd['by_class']}")
    print()
    print("VTDNet (conf=0.25, как было в первом демо):")
    print(f"  Сумма детекций ТС:     {vtd_old['total_vehicle_detections']}")
    print(f"  В среднем на кадр:     {vtd_old['avg_vehicles_per_frame']:.2f}")
    print(f"  Макс. на одном кадре:  {report['vtdnet_conf_025_old']['max_on_single_frame']}")
    print()
    ratio = vtd["total_vehicle_detections"] / max(yolo["total_vehicle_detections"], 1)
    print(f"VTDNet / YOLO11x по сумме bbox: {ratio:.1%}")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
