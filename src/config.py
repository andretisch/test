from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"


@dataclass
class LineConfig:
    start: tuple[int, int]
    end: tuple[int, int]


@dataclass
class LabelsConfig:
    in_label: str = "к камере"
    out_label: str = "от камеры"


@dataclass
class AnnotatorConfig:
    box_thickness: int = 4
    trace_length: int = 30
    label_text_thickness: int = 4
    label_text_scale: float = 2.0
    line_thickness: int = 4


@dataclass
class GradioConfig:
    server_name: str = "0.0.0.0"
    server_port: int = 7860


@dataclass
class AppConfig:
    source_video: str = "data/test_video.mp4"
    target_video: str = "output/result.mp4"
    model_backend: str = "yolo"  # yolo | vtdnet
    model_path: str = "models/yolov8s.onnx"
    confidence: float = 0.25
    vtdnet_confidence: float = 0.06
    iou: float = 0.45
    imgsz: int = 1280
    vehicle_class_ids: list[int] = field(default_factory=lambda: [2, 3, 5, 7])
    line: LineConfig = field(default_factory=lambda: LineConfig((0, 540), (1920, 540)))
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    swap_directions: bool = True
    annotators: AnnotatorConfig = field(default_factory=AnnotatorConfig)
    gradio: GradioConfig = field(default_factory=GradioConfig)

    def resolve_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else ROOT / p


def _parse_line(data: dict[str, Any]) -> LineConfig:
    start = tuple(data["start"])
    end = tuple(data["end"])
    return LineConfig((int(start[0]), int(start[1])), (int(end[0]), int(end[1])))


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    labels_raw = raw.get("labels", {})
    annot_raw = raw.get("annotators", {})
    gradio_raw = raw.get("gradio", {})

    return AppConfig(
        source_video=raw.get("source_video", "data/test_video.mp4"),
        target_video=raw.get("target_video", "output/result.mp4"),
        model_backend=str(raw.get("model_backend", "yolo")).lower(),
        model_path=raw.get("model_path", "models/yolov8s.onnx"),
        confidence=float(raw.get("confidence", 0.25)),
        vtdnet_confidence=float(raw.get("vtdnet_confidence", raw.get("confidence", 0.06))),
        iou=float(raw.get("iou", 0.45)),
        imgsz=int(raw.get("imgsz", 1280)),
        vehicle_class_ids=list(raw.get("vehicle_class_ids", [2, 3, 5, 7])),
        line=_parse_line(raw.get("line", {"start": [0, 540], "end": [1920, 540]})),
        labels=LabelsConfig(
            in_label=labels_raw.get("in", "к камере"),
            out_label=labels_raw.get("out", "от камеры"),
        ),
        swap_directions=bool(raw.get("swap_directions", True)),
        annotators=AnnotatorConfig(
            box_thickness=int(annot_raw.get("box_thickness", 4)),
            trace_length=int(annot_raw.get("trace_length", 30)),
            label_text_thickness=int(annot_raw.get("label_text_thickness", 4)),
            label_text_scale=float(annot_raw.get("label_text_scale", 2.0)),
            line_thickness=int(annot_raw.get("line_thickness", 4)),
        ),
        gradio=GradioConfig(
            server_name=str(gradio_raw.get("server_name", "0.0.0.0")),
            server_port=int(gradio_raw.get("server_port", 7860)),
        ),
    )
