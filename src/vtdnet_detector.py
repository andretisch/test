"""Инференс VTDNet (ONNX) → supervision.Detections для пайплайна подсчёта."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Train.constants import CLASS_NAMES, IMGSZ
from Train.vtdnet.decode import decode_predictions


class VTDNetDetector:
    def __init__(
        self,
        onnx_path: Path | str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = IMGSZ,
    ) -> None:
        path = Path(onnx_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"VTDNet ONNX не найден: {path}")

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = {i: n for i, n in enumerate(CLASS_NAMES)}
        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self.session.get_inputs()[0].name

    def predict(self, frame_bgr: np.ndarray) -> sv.Detections:
        h, w = frame_bgr.shape[:2]
        resized = cv2.resize(
            frame_bgr,
            (self.imgsz, self.imgsz),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        raw = self.session.run(None, {self._input_name: tensor})[0]
        decoded = decode_predictions(
            torch.from_numpy(raw),
            conf_thres=self.conf,
            iou_thres=self.iou,
        )[0]

        if decoded.numel() == 0:
            return sv.Detections.empty()

        det = decoded.cpu().numpy()
        xyxy = det[:, :4].copy()
        xyxy[:, [0, 2]] *= w
        xyxy[:, [1, 3]] *= h

        return sv.Detections(
            xyxy=xyxy,
            confidence=det[:, 4],
            class_id=det[:, 5].astype(int),
        )
