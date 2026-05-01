# Оркестратор: кадр → детекция → трекинг → подсчёт
from __future__ import annotations
import time
import numpy as np
import cv2
from pathlib import Path
from app.core.detector import Detector
from app.core.tracker import Tracker
from app.core.counter import LineCounter
from app.storage.db import EventStore


class Pipeline:
    """Связывает детектор, трекер и счётчик в единый поток обработки."""

    def __init__(self) -> None:
        self.detector = Detector()
        self.tracker = Tracker()
        self.counter = LineCounter()
        self.store = EventStore()
        self._initialized = False
        self._source: str | None = None
        self._running = False
        self._last_fps: float = 0.0

    def initialize(self) -> None:
        """Загрузить модель и инициализировать трекер."""
        self.detector.load()
        self.tracker.init()
        self.store.connect()
        self._initialized = True

    def shutdown(self) -> None:
        """Освободить ресурсы."""
        self.store.close()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Обработать один кадр.

        Возвращает:
        {
            'tracks': [...],
            'events': [...],
            'stats': {...},
            'latency_ms': float,
        }
        """
        if not self._initialized:
            raise RuntimeError("Pipeline не инициализирован. Вызовите initialize().")

        t0 = time.perf_counter()

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        events = self.counter.update(tracks)

        # Сохраняем события в БД
        if events:
            self.store.save_events(events)

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "tracks": tracks,
            "events": events,
            "stats": self.counter.stats,
            "latency_ms": round(latency_ms, 1),
        }

    def process_video(self, video_path: str | Path) -> dict:
        """
        Обработать видеофайл целиком. Возвращает итоговую статистику.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline не инициализирован. Вызовите initialize().")

        self._source = str(video_path)
        self._running = True

        self.tracker.reset()
        self.counter.reset()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.counter.set_frame_size(w, h)

        frame_count = 0
        total_latency = 0.0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame)
                frame_count += 1
                total_latency += result["latency_ms"]
        finally:
            cap.release()
            self._running = False

        avg_latency = total_latency / max(frame_count, 1)
        self._last_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        self._source = None

        return {
            "frames_processed": frame_count,
            "avg_latency_ms": round(avg_latency, 1),
            "fps": round(self._last_fps, 1),
            "stats": self.counter.stats,
        }

    def stop(self) -> None:
        """Остановить обработку видео."""
        self._running = False

    @property
    def status(self) -> dict:
        return {
            "initialized": self._initialized,
            "running": self._running,
            "source": self._source,
            "fps": self._last_fps,
        }
