from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.config import AppConfig, load_config
from src.cyrillic_draw import draw_line_zone_labels
from src.model_loader import ensure_onnx_model


@dataclass
class JobResult:
    in_count: int
    out_count: int
    target_path: str


class VehicleCounterPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        model_path = ensure_onnx_model(config)
        self.model = YOLO(str(model_path), task="detect")
        self.tracker = sv.ByteTrack()
        self._class_names: dict[int, str] | None = None

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> VehicleCounterPipeline:
        return cls(load_config(config_path))

    def _filter_vehicles(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        mask = np.isin(
            detections.class_id, np.array(self.config.vehicle_class_ids, dtype=int)
        )
        return detections[mask]

    def _build_annotators(
        self,
        line_zone: sv.LineZone,
    ) -> tuple[
        sv.TraceAnnotator,
        sv.BoxAnnotator,
        sv.LabelAnnotator,
        sv.LineZoneAnnotator,
    ]:
        ann = self.config.annotators
        trace = sv.TraceAnnotator(
            thickness=ann.box_thickness,
            trace_length=ann.trace_length,
        )
        box = sv.BoxAnnotator(thickness=ann.box_thickness)
        label = sv.LabelAnnotator(
            text_thickness=ann.label_text_thickness,
            text_scale=ann.label_text_scale,
        )
        line_ann = sv.LineZoneAnnotator(
            thickness=ann.line_thickness,
            text_thickness=ann.label_text_thickness,
            text_scale=ann.label_text_scale,
            display_in_count=False,
            display_out_count=False,
        )
        return trace, box, label, line_ann

    def run(
        self,
        source_path: str | Path,
        target_path: str | Path,
        line_start: tuple[int, int],
        line_end: tuple[int, int],
        progress_callback: Callable[[float], None] | None = None,
        swap_directions: bool | None = None,
    ) -> JobResult:
        source = Path(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if swap_directions is None:
            swap_directions = self.config.swap_directions
        if swap_directions:
            line_start, line_end = line_end, line_start

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        line_zone = sv.LineZone(
            start=sv.Point(line_start[0], line_start[1]),
            end=sv.Point(line_end[0], line_end[1]),
        )
        trace_ann, box_ann, label_ann, line_ann = self._build_annotators(line_zone)

        video_info = sv.VideoInfo.from_video_path(str(source))
        total_frames = video_info.total_frames or 1
        frame_index = 0

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            nonlocal frame_index
            frame_index = index

            results = self.model(
                frame,
                verbose=False,
                conf=self.config.confidence,
                iou=self.config.iou,
                imgsz=self.config.imgsz,
                device="cpu",
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self._filter_vehicles(detections)
            detections = self.tracker.update_with_detections(detections)

            labels = []
            for conf, class_id, tracker_id in zip(
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            ):
                name = results.names[int(class_id)]
                labels.append(f"#{tracker_id} {name} {conf:.2f}")

            annotated = frame.copy()
            annotated = trace_ann.annotate(annotated, detections)
            annotated = box_ann.annotate(annotated, detections)
            annotated = label_ann.annotate(annotated, detections, labels)
            line_zone.trigger(detections)

            if progress_callback and total_frames > 0:
                progress_callback(min(1.0, (index + 1) / total_frames))

            annotated = line_ann.annotate(annotated, line_counter=line_zone)
            annotated = draw_line_zone_labels(
                annotated,
                line_zone,
                self.config.labels.in_label,
                self.config.labels.out_label,
                font_size=max(20, int(self.config.annotators.label_text_scale * 14)),
            )
            return annotated

        sv.process_video(
            source_path=str(source),
            target_path=str(target),
            callback=callback,
        )

        counts_path = target.with_suffix(".json")
        counts_data = {
            "in_count": line_zone.in_count,
            "out_count": line_zone.out_count,
            "labels": {
                "in": self.config.labels.in_label,
                "out": self.config.labels.out_label,
            },
            "line": {"start": list(line_start), "end": list(line_end)},
            "swap_directions": swap_directions,
        }
        counts_path.write_text(
            json.dumps(counts_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return JobResult(
            in_count=line_zone.in_count,
            out_count=line_zone.out_count,
            target_path=str(target),
        )
