#!/usr/bin/env python3
"""Gradio UI for vehicle counting."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.gradio_utils import (
    format_stats,
    on_image_click,
    on_video_upload,
    reset_line,
    validate_line_state,
)
from src.pipeline import VehicleCounterPipeline

STORAGE_OUTPUTS = ROOT / "storage" / "outputs"
STORAGE_OUTPUTS.mkdir(parents=True, exist_ok=True)

config = load_config()
pipeline = VehicleCounterPipeline(config)


def run_processing(
    video: dict | str | None,
    state: dict,
    swap_directions: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str]:
    line = validate_line_state(state)
    if line is None:
        return None, "⚠️ Задайте линию: кликните **две точки** на превью кадра."

    from src.gradio_utils import _video_path

    video_path = state.get("video_path") or _video_path(video)
    if not video_path or not Path(video_path).exists():
        return None, "⚠️ Сначала загрузите видео."

    out_path = STORAGE_OUTPUTS / f"{uuid.uuid4().hex}.mp4"
    start, end = line

    def update_progress(value: float) -> None:
        progress(value, desc="Обработка видео...")

    try:
        result = pipeline.run(
            source_path=video_path,
            target_path=out_path,
            line_start=start,
            line_end=end,
            progress_callback=update_progress,
            swap_directions=swap_directions,
        )
    except Exception as exc:
        return None, f"❌ Ошибка обработки: {exc}"

    stats = format_stats(
        result.in_count,
        result.out_count,
        config.labels.in_label,
        config.labels.out_label,
    )
    return result.target_path, stats


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Подсчёт транспортных средств") as demo:
        gr.Markdown(
            """
            # Подсчёт ТС по линии

            1. Загрузите видео с камеры видеонаблюдения
            2. На превью кадра **кликните дважды** — начало и конец линии подсчёта
            3. Нажмите **Запустить обработку**
            """
        )

        with gr.Row():
            video_in = gr.Video(label="Входное видео", sources=["upload"])
            frame_view = gr.Image(
                label="Линия подсчёта (2 клика)",
                type="numpy",
                interactive=True,
            )

        line_state = gr.State(
            {"points": [], "video_path": None, "frame_shape": None, "base_frame": None}
        )

        swap_dirs = gr.Checkbox(
            label="Поменять направления (если «к камере» считает уезжающих)",
            value=config.swap_directions,
        )

        with gr.Row():
            btn_reset = gr.Button("Сбросить линию")
            btn_run = gr.Button("Запустить обработку", variant="primary")

        with gr.Row():
            video_out = gr.Video(label="Результат")
            stats = gr.Markdown()

        video_in.change(
            fn=on_video_upload,
            inputs=[video_in],
            outputs=[frame_view, line_state],
        )
        frame_view.select(
            fn=on_image_click,
            inputs=[frame_view, line_state],
            outputs=[frame_view, line_state],
        )
        btn_reset.click(
            fn=reset_line,
            inputs=[line_state],
            outputs=[frame_view, line_state],
        )
        btn_run.click(
            fn=run_processing,
            inputs=[video_in, line_state, swap_dirs],
            outputs=[video_out, stats],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=config.gradio.server_name,
        server_port=config.gradio.server_port,
    )
