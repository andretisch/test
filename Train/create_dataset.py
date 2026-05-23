#!/usr/bin/env python3
"""
Сбор датасета в формате YOLO из видео или изображений.

YOLO11x (COCO) размечает кадры; остаются только классы дорожного транспорта.
Результат: Dataset/images/{train,val}, Dataset/labels/{train,val}, Dataset/data.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Train.constants import DEFAULT_LABEL_MODEL as DEFAULT_MODEL

DEFAULT_DATASET_DIR = ROOT / "Dataset"

# COCO -> индексы в нашем датасете (4 класса)
COCO_TO_DATASET: dict[int, int] = {
    2: 0,  # car
    3: 1,  # motorcycle
    5: 2,  # bus
    7: 3,  # truck
}
CLASS_NAMES: list[str] = ["car", "motorcycle", "bus", "truck"]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка YOLO-датасета транспорта (псевдоразметка YOLO11x).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Видео, папка с видео или папка с изображениями.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Куда сохранить датасет (по умолчанию: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Веса Ultralytics для разметки (по умолчанию: yolo11x.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence.")
    parser.add_argument("--iou", type=float, default=0.45, help="Порог NMS IoU.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Сторона входа при инференсе (больше — точнее на дальних объектах).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Брать каждый N-й кадр из видео (1 = все кадры, без пропусков).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Доля примеров в val (0..1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Лимит сохранённых кадров (0 = без лимита).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для train/val split.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство инференса (cpu, cuda:0, ...).",
    )
    return parser.parse_args()


def collect_media_paths(source: Path) -> tuple[list[Path], list[Path]]:
    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return [source], []
        if suffix in IMAGE_EXTENSIONS:
            return [], [source]
        raise ValueError(f"Неподдерживаемый файл: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"Источник не найден: {source}")

    videos: list[Path] = []
    images: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            videos.append(path)
        elif suffix in IMAGE_EXTENSIONS:
            images.append(path)
    return videos, images


def iter_frames_from_videos(
    video_paths: list[Path],
    frame_step: int,
) -> list[tuple[np.ndarray, str]]:
    if frame_step < 1:
        raise ValueError("frame-step должен быть >= 1")

    samples: list[tuple[np.ndarray, str]] = []
    for video_path in video_paths:
        stem = video_path.stem
        generator = sv.get_video_frames_generator(str(video_path))
        for index, frame in enumerate(generator):
            if index % frame_step != 0:
                continue
            sample_id = f"{stem}_f{index:06d}"
            samples.append((frame, sample_id))
    return samples


def iter_frames_from_images(image_paths: list[Path]) -> list[tuple[np.ndarray, str]]:
    samples: list[tuple[np.ndarray, str]] = []
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Пропуск (не удалось прочитать): {image_path}", file=sys.stderr)
            continue
        samples.append((frame, image_path.stem))
    return samples


def xyxy_to_yolo_line(
    xyxy: np.ndarray,
    class_id: int,
    image_width: int,
    image_height: int,
) -> str:
    x1, y1, x2, y2 = xyxy
    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return (
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    )


def filter_vehicle_detections(
    detections: sv.Detections,
) -> tuple[sv.Detections, list[int]]:
    if len(detections) == 0:
        return detections, []

    mapped_ids: list[int] = []
    keep_indices: list[int] = []
    for i, coco_id in enumerate(detections.class_id):
        dataset_id = COCO_TO_DATASET.get(int(coco_id))
        if dataset_id is None:
            continue
        keep_indices.append(i)
        mapped_ids.append(dataset_id)

    if not keep_indices:
        return sv.Detections.empty(), []

    filtered = detections[keep_indices]
    return filtered, mapped_ids


def label_frame(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[str]:
    results = model(
        frame,
        verbose=False,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    detections, class_ids = filter_vehicle_detections(detections)

    if len(detections) == 0:
        return []

    h, w = frame.shape[:2]
    lines: list[str] = []
    for xyxy, class_id in zip(detections.xyxy, class_ids):
        lines.append(xyxy_to_yolo_line(xyxy, class_id, w, h))
    return lines


def prepare_output_dirs(dataset_dir: Path) -> None:
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_generated_dataset(dataset_dir: Path) -> None:
    for sub in ("images", "labels"):
        target = dataset_dir / sub
        if target.exists():
            shutil.rmtree(target)


def write_data_yaml(dataset_dir: Path) -> Path:
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def save_split(
    samples: list[tuple[np.ndarray, str, list[str]]],
    dataset_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, int | dict[str, int]]:
    random.Random(seed).shuffle(samples)
    if not samples:
        return {"total_saved": 0, "train": 0, "val": 0, "skipped_empty": 0}

    val_count = int(len(samples) * val_ratio)
    if val_ratio > 0 and val_count == 0 and len(samples) > 1:
        val_count = 1

    stats = {"total_saved": 0, "train": 0, "val": 0, "by_class": {n: 0 for n in CLASS_NAMES}}

    for index, (frame, sample_id, lines) in enumerate(samples):
        split = "val" if index < val_count else "train"
        image_name = f"{sample_id}.jpg"
        label_name = f"{sample_id}.txt"

        image_path = dataset_dir / "images" / split / image_name
        label_path = dataset_dir / "labels" / split / label_name

        cv2.imwrite(str(image_path), frame)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats["total_saved"] += 1
        stats[split] += 1
        for line in lines:
            class_id = int(line.split()[0])
            stats["by_class"][CLASS_NAMES[class_id]] += 1

    return stats


def main() -> None:
    args = parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    dataset_dir = (
        args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    )

    videos, images = collect_media_paths(source)
    if not videos and not images:
        print(f"Нет видео/изображений в {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Загрузка модели: {args.model}")
    model = YOLO(args.model)

    raw_samples: list[tuple[np.ndarray, str]] = []
    raw_samples.extend(iter_frames_from_videos(videos, args.frame_step))
    raw_samples.extend(iter_frames_from_images(images))

    if not raw_samples:
        print("Нет кадров для обработки.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Кадров к разметке: {len(raw_samples)} "
        f"(видео: {len(videos)}, изображений: {len(images)})"
    )

    labeled: list[tuple[np.ndarray, str, list[str]]] = []
    skipped_empty = 0

    for index, (frame, sample_id) in enumerate(raw_samples, start=1):
        if args.max_samples > 0 and len(labeled) >= args.max_samples:
            break

        lines = label_frame(
            model,
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
        if not lines:
            skipped_empty += 1
            continue

        labeled.append((frame, sample_id, lines))
        if index % 50 == 0 or index == len(raw_samples):
            print(f"  обработано {index}/{len(raw_samples)}, сохранено {len(labeled)}")

    clear_generated_dataset(dataset_dir)
    prepare_output_dirs(dataset_dir)
    yaml_path = write_data_yaml(dataset_dir)
    stats = save_split(labeled, dataset_dir, args.val_ratio, args.seed)
    stats["skipped_empty"] = skipped_empty
    stats["frames_processed"] = len(raw_samples)
    stats["frame_step"] = args.frame_step
    stats["imgsz"] = args.imgsz
    stats["source"] = str(source.resolve())
    stats["model"] = args.model
    stats["class_names"] = CLASS_NAMES

    meta_path = dataset_dir / "dataset_meta.json"
    meta_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nГотово.")
    print(f"  data.yaml: {yaml_path}")
    print(f"  train: {stats['train']}, val: {stats['val']}")
    print(f"  пропущено без ТС: {skipped_empty}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
