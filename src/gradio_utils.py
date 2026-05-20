from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import supervision as sv


def extract_first_frame(video_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = str(video_path)
    generator = sv.get_video_frames_generator(path)
    frame = next(generator)
    info = sv.VideoInfo.from_video_path(path)
    state: dict[str, Any] = {
        "points": [],
        "video_path": path,
        "frame_shape": (info.height, info.width),
        "base_frame": frame.copy(),
    }
    return frame, state


def draw_line_on_frame(
    frame: np.ndarray,
    points: list[tuple[int, int]],
) -> np.ndarray:
    out = frame.copy()
    for pt in points:
        cv2.circle(out, pt, 12, (0, 255, 0), -1)
        cv2.circle(out, pt, 14, (255, 255, 255), 2)
    if len(points) == 2:
        cv2.line(out, points[0], points[1], (0, 255, 255), 3)
    return out


def _video_path(video: object) -> str | None:
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, Path):
        return str(video)
    if isinstance(video, dict):
        return video.get("name") or video.get("path") or video.get("video")
    return getattr(video, "name", None) or getattr(video, "path", None)


def on_video_upload(video: object) -> tuple[np.ndarray | None, dict[str, Any]]:
    empty: dict[str, Any] = {
        "points": [],
        "video_path": None,
        "frame_shape": None,
        "base_frame": None,
    }
    path = _video_path(video)
    if not path or not Path(path).exists():
        return None, empty

    frame, state = extract_first_frame(path)
    return frame, state


def on_image_click(
    frame: np.ndarray | None,
    state: dict[str, Any],
    evt: gr.SelectData,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if frame is None or state.get("base_frame") is None:
        return frame, state

    base = state["base_frame"]
    x, y = int(evt.index[0]), int(evt.index[1])
    points: list[tuple[int, int]] = list(state.get("points", []))

    if len(points) >= 2:
        points = [(x, y)]
    else:
        points.append((x, y))

    state = {**state, "points": points}
    annotated = draw_line_on_frame(base, points)
    return annotated, state


def reset_line(state: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    base = state.get("base_frame")
    if base is None:
        return None, {"points": [], "video_path": None, "frame_shape": None, "base_frame": None}
    state = {**state, "points": []}
    return base.copy(), state


def format_stats(in_count: int, out_count: int, in_label: str, out_label: str) -> str:
    return (
        f"### Результаты подсчёта\n\n"
        f"| Направление | Количество |\n"
        f"|-------------|------------|\n"
        f"| **{in_label}** | **{in_count}** |\n"
        f"| **{out_label}** | **{out_count}** |\n"
    )


def validate_line_state(state: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    points = state.get("points", [])
    if len(points) != 2:
        return None
    return (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1]))
