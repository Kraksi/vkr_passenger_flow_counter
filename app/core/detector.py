# Обёртка над YOLOv11 — инференс и фильтрация по conf
from __future__ import annotations
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from app.config import (
    MODEL_PATH,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_DEVICE,
    DETECTOR_CLASS_PERSON,
)


class Detector:
    """Загружает YOLOv11n и выполняет детекцию людей на кадре."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        conf: float = DETECTOR_CONF_THRESHOLD,
        iou: float = DETECTOR_IOU_THRESHOLD,
        device: str = DETECTOR_DEVICE,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model: YOLO | None = None

    def load(self) -> None:
        """Загрузить веса модели в память GPU."""
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Принять кадр (H x W x 3, BGR), вернуть список детекций.

        Каждая детекция: {'bbox': [x1, y1, x2, y2], 'conf': float, 'class': int}
        Возвращает только класс DETECTOR_CLASS_PERSON.
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=[DETECTOR_CLASS_PERSON],
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                detections.append({
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(box.conf[0].cpu()),
                    "class": int(box.cls[0].cpu()),
                })
        return detections
