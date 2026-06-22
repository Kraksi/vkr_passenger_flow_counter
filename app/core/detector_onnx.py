from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.config import (
    MODEL_ONNX_PATH,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_CLASS_PERSON,
)


class DetectorONNX:
    """детектор людей на ONNX Runtime, интерфейс как у Detector (load/detect).
    провайдер CUDA на десктопе, CPU как fallback. на RK3588 - RKNN"""

    INPUT_SIZE = 640

    def __init__(
        self,
        model_path: Path = MODEL_ONNX_PATH,
        conf: float = DETECTOR_CONF_THRESHOLD,
        iou: float = DETECTOR_IOU_THRESHOLD,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self._session = None
        self._input_name: str = ""
        self._orig_w: int = 0
        self._orig_h: int = 0

    def load(self) -> None:
        """грузим ONNX через onnxruntime"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "Пакет onnxruntime не установлен. Установите его командой: "
                "uv pip install onnxruntime-gpu"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX-модель не найдена: {self.model_path}\n"
                "Экспортируйте её: python scripts/export.py --format onnx"
            )

        available = [p.lower() for p in ort.get_available_providers()]
        if self.device == "cuda" and "cudaexecutionprovider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        active = self._session.get_providers()[0]
        print(f"[DetectorONNX] Загружена: {self.model_path.name} | провайдер: {active}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """кадр BGR - список детекций {bbox, conf, class}, только person"""
        if self._session is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")

        self._orig_h, self._orig_w = frame.shape[:2]

        img, ratio, (pad_w, pad_h) = self._letterbox(frame, self.INPUT_SIZE)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis]

        outputs = self._session.run(None, {self._input_name: img})

        return self._postprocess(outputs[0], ratio, pad_w, pad_h)


    def _letterbox(
        self,
        img: np.ndarray,
        target: int,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """letterbox: масштаб с сохранением пропорций - (img, ratio, (pad_w, pad_h))"""
        h, w = img.shape[:2]
        ratio = min(target / h, target / w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target - new_w) // 2
        pad_h = (target - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target - new_h - pad_h,
            pad_w, target - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, ratio, (pad_w, pad_h)

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
    ) -> list[dict]:
        """сырой выход ONNX - детекции в координатах исходного кадра.
        shape (1, 4+nc, anchors): cx,cy,w,h + conf"""
        pred = output[0]

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        nc = pred.shape[1] - 4
        boxes_raw = pred[:, :4]
        scores_raw = pred[:, 4:]

        if nc == 1:
            confs = scores_raw[:, 0]
        else:
            confs = scores_raw[:, DETECTOR_CLASS_PERSON]

        mask = confs >= self.conf
        if not mask.any():
            return []

        boxes = boxes_raw[mask]
        confs = confs[mask]

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        x1 = np.clip((x1 - pad_w) / ratio, 0, self._orig_w)
        y1 = np.clip((y1 - pad_h) / ratio, 0, self._orig_h)
        x2 = np.clip((x2 - pad_w) / ratio, 0, self._orig_w)
        y2 = np.clip((y2 - pad_h) / ratio, 0, self._orig_h)

        indices = self._nms(
            np.stack([x1, y1, x2, y2], axis=1),
            confs,
            self.iou,
        )

        detections = []
        for i in indices:
            detections.append({
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "conf": float(confs[i]),
                "class": DETECTOR_CLASS_PERSON,
            })
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        """жадный NMS - индексы выживших боксов"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-9)
            order = order[1:][iou < iou_thresh]

        return keep

    def benchmark(self, frame: np.ndarray, n_runs: int = 100) -> dict:
        """средняя латентность на n_runs кадрах (для ноутбука 05, PT vs ONNX)"""
        for _ in range(5):
            self.detect(frame)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
            "fps": round(1000 / float(np.mean(times)), 1),
        }
