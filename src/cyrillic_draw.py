"""Draw Cyrillic text on OpenCV frames (supervision uses Hershey fonts without Cyrillic)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

FONT_CANDIDATES = [
    ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
]


@lru_cache(maxsize=1)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def draw_text_box(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    *,
    font_size: int = 28,
    text_color: tuple[int, int, int] = (0, 0, 0),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    padding: int = 10,
    above: bool = True,
) -> np.ndarray:
    """Draw text with background near anchor (x, y) — center of line zone."""
    font = _load_font(font_size)
    rgb = _bgr_to_rgb(frame)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = anchor[0] - tw // 2
    y = anchor[1] - th - padding * 2 if above else anchor[1] + padding

    box = (x - padding, y - padding, x + tw + padding, y + th + padding)
    draw.rectangle(box, fill=bg_color)
    draw.text((x, y), text, font=font, fill=text_color)

    return _rgb_to_bgr(np.asarray(pil))


def draw_line_zone_labels(
    frame: np.ndarray,
    line_zone: sv.LineZone,
    in_label: str,
    out_label: str,
    *,
    font_size: int = 28,
    text_offset_factor: float = 1.5,
) -> np.ndarray:
    """Draw in/out counters in Cyrillic near the counting line."""
    center = line_zone.vector.center
    cx, cy = int(center.x), int(center.y)
    offset = int(font_size * text_offset_factor)

    frame = draw_text_box(
        frame,
        f"{in_label}: {line_zone.in_count}",
        (cx, cy - offset),
        font_size=font_size,
        above=True,
    )
    frame = draw_text_box(
        frame,
        f"{out_label}: {line_zone.out_count}",
        (cx, cy + offset),
        font_size=font_size,
        above=False,
    )
    return frame
