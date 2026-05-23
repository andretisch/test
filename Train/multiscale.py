from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from Train.constants import MULTISCALE_MAX, MULTISCALE_MIN, MULTISCALE_STEP
from Train.vtdnet.model import VTDNetConfig


def sample_train_imgsz() -> int:
    """Случайная сторона входа в [512, 768], шаг 32 (как у Ultralytics YOLO)."""
    steps = (MULTISCALE_MAX - MULTISCALE_MIN) // MULTISCALE_STEP + 1
    idx = random.randint(0, steps - 1)
    return MULTISCALE_MIN + idx * MULTISCALE_STEP


def resize_batch(images: torch.Tensor, imgsz: int) -> torch.Tensor:
    h, w = images.shape[-2:]
    if h == imgsz and w == imgsz:
        return images
    return F.interpolate(
        images,
        size=(imgsz, imgsz),
        mode="bilinear",
        align_corners=False,
    )


def cfg_for_imgsz(base_cfg: VTDNetConfig, imgsz: int) -> VTDNetConfig:
    return VTDNetConfig(
        num_classes=base_cfg.num_classes,
        imgsz=imgsz,
        channels=base_cfg.channels,
        strides=base_cfg.strides,
    )
