from __future__ import annotations
import time
import numpy as np
import cv2
from pathlib import Path
from app.core.counter import LineCounter, TripwireCounter, ZoneCounter
from app.config import (
    DETECTOR_BACKEND, TRACKER_BACKEND, COUNTER_MODE,
    ZONE_RECT, ZONE_DIRECTION, ZONE_TRAVEL_REL, ZONE_MIN_FRAMES,
)
from app.storage.db import event_store


def make_detector():
    """детектор по DETECTOR_BACKEND (pytorch|onnx|rknn), единый интерфейс load/detect"""
    if DETECTOR_BACKEND == "rknn":
        from app.core.detector_rknn import DetectorRKNN
        return DetectorRKNN()
    if DETECTOR_BACKEND == "onnx":
        from app.core.detector_onnx import DetectorONNX
        return DetectorONNX()
    from app.core.detector import Detector
    return Detector()


def make_tracker():
    """трекер по TRACKER_BACKEND (osnet|bytetrack)"""
    if TRACKER_BACKEND == "osnet":
        from app.core.tracker_osnet import TrackerOSNet
        return TrackerOSNet()
    from app.core.tracker import Tracker
    return Tracker()


def make_counter():
    """счётчик по COUNTER_MODE (zone|tripwire|lifecycle)"""
    if COUNTER_MODE == "zone":
        return ZoneCounter(zone=ZONE_RECT, direction=ZONE_DIRECTION,
                           travel_rel=ZONE_TRAVEL_REL, min_frames=ZONE_MIN_FRAMES)
    if COUNTER_MODE == "tripwire":
        return TripwireCounter()
    return LineCounter()


class Pipeline:
    """детектор + трекер + счётчик в одном потоке обработки"""

    def __init__(self) -> None:
        self.detector = make_detector()
        self.tracker = make_tracker()
        self.counter = make_counter()
        self.store = event_store
        self._initialized = False
        self._source: str | None = None
        self._running = False
        self._last_fps: float = 0.0

    def initialize(self) -> None:
        """грузим модель, инициализируем трекер и БД"""
        self.detector.load()
        self.tracker.init()
        if self.store._conn is None:
            self.store.connect()
        self._initialized = True

    def shutdown(self) -> None:
        """освободить ресурсы"""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def process_frame(self, frame: np.ndarray) -> dict:
        """один кадр - {tracks, events, stats, latency_ms}"""
        if not self._initialized:
            raise RuntimeError("Pipeline не инициализирован. Вызовите initialize().")

        t0 = time.perf_counter()

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        events = self.counter.update(tracks)

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
        """прогнать видео целиком - итоговая статистика"""
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

        final_events = self.counter.finalize_remaining()
        if final_events:
            self.store.save_events(final_events)

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
        """стоп обработки видео"""
        self._running = False

    @property
    def status(self) -> dict:
        return {
            "initialized": self._initialized,
            "running": self._running,
            "source": self._source,
            "fps": self._last_fps,
        }
