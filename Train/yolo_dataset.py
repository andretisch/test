from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def load_data_yaml(path: Path) -> dict:
  with open(path, encoding="utf-8") as f:
    return yaml.safe_load(f)


class YoloFolderDataset(Dataset):
  """Читает разметку YOLO (txt) — формат датасета тот же, архитектура своя."""

  def __init__(
    self,
    images_dir: Path,
    labels_dir: Path,
    imgsz: int,
    augment: bool = False,
  ) -> None:
    self.images_dir = images_dir
    self.labels_dir = labels_dir
    self.imgsz = imgsz
    self.augment = augment
    self.image_paths = sorted(images_dir.glob("*.jpg"))
    if not self.image_paths:
      self.image_paths = sorted(images_dir.glob("*.png"))

  def __len__(self) -> int:
    return len(self.image_paths)

  def _load_labels(self, label_path: Path) -> torch.Tensor:
    if not label_path.exists():
      return torch.zeros((0, 5), dtype=torch.float32)
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
      if not line.strip():
        continue
      parts = line.split()
      cls_id = float(parts[0])
      cx, cy, w, h = map(float, parts[1:5])
      rows.append([cls_id, cx, cy, w, h])
    if not rows:
      return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    img_path = self.image_paths[index]
    image = cv2.imread(str(img_path))
    if image is None:
      raise RuntimeError(f"Не удалось прочитать: {img_path}")

    h0, w0 = image.shape[:2]
    image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    label_path = self.labels_dir / f"{img_path.stem}.txt"
    targets = self._load_labels(label_path)

    if self.augment and len(targets) > 0 and np.random.random() < 0.5:
      image = np.ascontiguousarray(image[:, ::-1, :])
      targets = targets.clone()
      targets[:, 1] = 1.0 - targets[:, 1]

    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor, targets


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
  images, targets = zip(*batch)
  return torch.stack(images, dim=0), list(targets)


def build_loaders(
  data_yaml: Path,
  imgsz: int,
  batch_size: int,
  workers: int = 2,
) -> tuple[DataLoader, DataLoader, dict]:
  cfg = load_data_yaml(data_yaml)
  root = Path(cfg["path"])
  if not root.is_absolute():
    root = data_yaml.parent / root
    if not root.exists():
      root = Path(cfg["path"])

  train_ds = YoloFolderDataset(
    root / "images" / "train",
    root / "labels" / "train",
    imgsz=imgsz,
    augment=True,
  )
  val_ds = YoloFolderDataset(
    root / "images" / "val",
    root / "labels" / "val",
    imgsz=imgsz,
    augment=False,
  )

  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    collate_fn=collate_fn,
    pin_memory=False,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    collate_fn=collate_fn,
    pin_memory=False,
  )
  return train_loader, val_loader, cfg
