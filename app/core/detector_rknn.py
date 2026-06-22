from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.config import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_CLASS_PERSON,
)

_DEFAULT_RKNN_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "yolo11n_mot20_v2.rknn"

IMGSZ = 640


class DetectorRKNN:
    """детектор людей через NPU RK3588 (rknnlite2), интерфейс как у Detector/ONNX
    (load/detect/release).

    нюансы RK3588: до 6 потоков на NPU (core_mask), FP16 (без INT8), batch=1,
    только aarch64 (Orange Pi 5, Radxa Rock 5)"""

    def __init__(
        self,
        model_path: Path = _DEFAULT_RKNN_MODEL,
        conf: float = DETECTOR_CONF_THRESHOLD,
        iou: float = DETECTOR_IOU_THRESHOLD,
        core_mask: int = 0,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.core_mask = core_mask
        self._rknn = None
        self._orig_w: int = 0
        self._orig_h: int = 0

    def load(self) -> None:
        """грузим RKNN в NPU"""
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise RuntimeError(
                "rknnlite2 не установлен или запущен не на RK3588.\n"
                "На RK3588: pip install rknn-toolkit-lite2\n"
                "На десктопе используйте DetectorONNX."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RKNN-модель не найдена: {self.model_path}\n"
                "Сконвертируйте ONNX→RKNN: python scripts/export.py --format rknn"
            )

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn() failed: {ret}")

        ret = self._rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime() failed: {ret}")

        print(f"[DetectorRKNN] Загружена: {self.model_path.name} | NPU core_mask={self.core_mask}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """кадр BGR - список детекций {bbox, conf, class}"""
        if self._rknn is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")

        self._orig_h, self._orig_w = frame.shape[:2]

        img, ratio, (pad_w, pad_h) = self._letterbox(frame, IMGSZ)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = self._rknn.inference(inputs=[img_rgb])

        return self._postprocess(outputs[0], ratio, pad_w, pad_h)

    def release(self) -> None:
        """освободить NPU, звать при выходе"""
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def benchmark(self, frame: np.ndarray, n_runs: int = 100) -> dict:
        """средняя латентность инференса на NPU"""
        for _ in range(5):
            self.detect(frame)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms":  round(float(np.std(times)),  2),
            "min_ms":  round(float(np.min(times)),  2),
            "fps":     round(1000 / float(np.mean(times)), 1),
        }


    def _letterbox(
        self,
        img: np.ndarray,
        target: int,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
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
        """постпроцессинг как в DetectorONNX"""
        pred = output
        if pred.ndim == 3:
            pred = pred[0]

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        nc = pred.shape[1] - 4
        boxes_raw = pred[:, :4]
        confs = pred[:, 4] if nc == 1 else pred[:, 4 + DETECTOR_CLASS_PERSON]

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

        indices = self._nms(np.stack([x1, y1, x2, y2], axis=1), confs, self.iou)

        return [
            {
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "conf": float(confs[i]),
                "class": DETECTOR_CLASS_PERSON,
            }
            for i in indices
        ]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
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
