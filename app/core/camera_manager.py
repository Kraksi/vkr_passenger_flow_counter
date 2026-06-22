"""несколько камер: подключение, калибровка, подсчёт в реалтайме"""
from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np

from app.core.pipeline import make_detector, make_tracker, make_counter
from app.core.detector import Detector
from app.core.tracker import Tracker
from app.config import (
    DOOR_ZONE_POINTS, TRIPWIRE_Y1, TRIPWIRE_Y2,
    HEAD_MODEL_PATH, HEAD_CONF_THRESHOLD, HEAD_IOU_THRESHOLD,
)
from app.storage.db import event_store

MAX_CAMERAS = 6



@dataclass
class CameraZone:
    """зона двери - 4 угла + направление входа. points по часовой с верх-лево.
    down_in: вниз = вход, up_in наоборот. дверь под углом - трапеция, не прямоугольник"""
    points: list = None
    direction: str = "down_in"
    relative: bool = True

    def __post_init__(self):
        if self.points is None:
            self.points = [list(p) for p in DOOR_ZONE_POINTS]

    def to_pixel(self, w: int, h: int):
        """углы в пикселях [(x,y),...]"""
        if self.relative:
            return [(int(px * w), int(py * h)) for px, py in self.points]
        return [(int(px), int(py)) for px, py in self.points]

    def to_dict(self) -> dict:
        return {"points": self.points, "direction": self.direction,
                "relative": self.relative}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraZone":
        """из dict (со старым форматом прямоугольника x1..y2)"""
        if "points" in d and d["points"]:
            pts = d["points"]
        elif any(k in d for k in ("x1", "y1", "x2", "y2")):
            x1 = d.get("x1", 0.0); y1 = d.get("y1", 0.0)
            x2 = d.get("x2", 0.45); y2 = d.get("y2", 1.0)
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            pts = [list(p) for p in DOOR_ZONE_POINTS]
        return cls(points=pts, direction=d.get("direction", "down_in"),
                   relative=d.get("relative", True))


CameraLine = CameraZone



class CameraStream:
    """захват кадров в фоновом потоке. source - RTSP, путь к файлу или int USB"""

    def __init__(self, source: str | int) -> None:
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._width = 0
        self._height = 0
        self._fps = 0.0

    def connect(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {self.source!r}")
        self._cap = cap
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        is_stream = isinstance(self.source, str) and (
            self.source.startswith(("rtsp", "rtmp", "http"))
        )
        is_file = isinstance(self.source, str) and not is_stream
        interval = (1.0 / self._fps) if (is_file and self._fps > 0) else 0.0
        next_t = time.perf_counter()
        while self._running and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                if is_stream:
                    time.sleep(0.5)
                else:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self._frame = frame
            if interval:
                next_t += interval
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def disconnect(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_alive(self) -> bool:
        return self._running and self._frame is not None



def _draw_dashed_hline(frame: np.ndarray, y: int, color, label: str,
                       dash: int = 18, gap: int = 12) -> None:
    """линия трипвайра во всю ширину - красный пунктир"""
    w = frame.shape[1]
    x = 0
    while x < w:
        cv2.line(frame, (x, y), (min(x + dash, w), y), color, 2, cv2.LINE_AA)
        x += dash + gap
    cv2.putText(frame, label, (8, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _draw_zone(frame: np.ndarray, zone: CameraZone) -> None:
    """рисуем зону двери + стрелку направления (in-place)"""
    h, w = frame.shape[:2]
    pts = np.array(zone.to_pixel(w, h), dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    dark = (frame * 0.4).astype(np.uint8)
    dark[mask == 255] = frame[mask == 255]
    frame[:] = dark

    cv2.polylines(frame, [pts], True, (0, 255, 255), 3, cv2.LINE_AA)
    for (cx, cy) in pts:
        cv2.circle(frame, (int(cx), int(cy)), 9, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx), int(cy)), 9, (255, 255, 255), 2, cv2.LINE_AA)
    tlx, tly = pts[0]
    cv2.putText(frame, "DOOR ZONE", (int(tlx) + 6, max(20, int(tly) + 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cx = int(pts[:, 0].mean())
    ytop, ybot = int(pts[:, 1].min()), int(pts[:, 1].max())

    _draw_dashed_hline(frame, int(TRIPWIRE_Y1 * h), (0, 0, 255), "L1")
    _draw_dashed_hline(frame, int(TRIPWIRE_Y2 * h), (0, 0, 255), "L2")

    down_in = zone.direction == "down_in"
    arrow_col = (0, 255, 0) if down_in else (255, 80, 0)
    ay1, ay2 = ((ytop + 20, ybot - 20) if down_in else (ybot - 20, ytop + 20))
    cv2.arrowedLine(frame, (cx, ay1), (cx, ay2), arrow_col, 3, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(frame, "IN", (cx + 10, ay2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, arrow_col, 2, cv2.LINE_AA)


_draw_line = _draw_zone


def _draw_tracks(frame: np.ndarray, head_tracks: list[dict], zone: CameraZone) -> None:
    """боксы голов + id только внутри зоны двери (in-place).

    обманка для демо: счёт идёт по телу на всём кадре, а в UI рисуем отдельный
    трекер голов - нагляднее. на счёт не влияет. показываем только головы в зоне"""
    h, w = frame.shape[:2]
    poly = np.array(zone.to_pixel(w, h), dtype=np.int32)
    for t in head_tracks:
        hx1, hy1, hx2, hy2 = (int(v) for v in t["bbox"])
        cx, cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) < 0:
            continue

        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 100), 2, cv2.LINE_AA)
        label = f"#{t['track_id']}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (hx1, hy1 - lh - 8), (hx1 + lw + 6, hy1), (0, 180, 70), -1)
        cv2.putText(frame, label, (hx1 + 3, hy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, stats: dict, fps: float) -> None:
    """HUD со статистикой в левом верхнем углу (in-place)"""
    lines = [
        (f"IN:  {stats['entries']}", (100, 220, 255)),
        (f"OUT: {stats['exits']}",   (255, 160,  80)),
        (f"NOW: {stats['current_inside']}", (100, 255, 140)),
        (f"FPS: {fps:.0f}",         (200, 200, 200)),
    ]
    cv2.rectangle(frame, (6, 6), (165, 6 + len(lines) * 32 + 6), (0, 0, 0), -1)
    cv2.rectangle(frame, (6, 6), (165, 6 + len(lines) * 32 + 6), (40, 40, 40), 1)

    y = 32
    for text, color in lines:
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2, cv2.LINE_AA)
        y += 32



class _CameraCounter:
    """фоновый поток детект/трек/счёт для одной камеры.

    общий Detector (грузится раз) + свой трекер и счётчик из конфига (по дефолту
    OSNet ReID + трипвайр, как в Pipeline). отдельный трекер голов - только для
    боксов в UI, на счёт не влияет"""

    def __init__(self, cam_id: str, stream: CameraStream, line_cfg: CameraLine,
                 detector, head_detector=None) -> None:
        self._cam_id = cam_id
        self._stream = stream
        self._detector = detector
        self._tracker = make_tracker()
        self._counter = make_counter()

        self._head_detector = head_detector
        self._head_tracker = Tracker() if head_detector is not None else None
        self._head_tracks: list[dict] = []

        self._lock = threading.Lock()
        self._line_cfg = line_cfg
        self._annotated: np.ndarray | None = None
        self._stats: dict = {"entries": 0, "exits": 0, "total": 0, "current_inside": 0}
        self._latency_ms = 0.0
        self._fps = 0.0
        self._t_ring: deque[float] = deque(maxlen=60)

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._tracker.init()
        if self._head_tracker is not None:
            self._head_tracker.init()
        self._counter.reset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            frame = self._stream.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]

            with self._lock:
                line_cfg = self._line_cfg

            self._counter.set_frame_size(w, h)
            if hasattr(self._counter, "update_zone"):
                self._counter.update_zone(
                    points=line_cfg.points,
                    direction=line_cfg.direction, relative=line_cfg.relative,
                )
            else:
                xs = [p[0] for p in line_cfg.points]; ys = [p[1] for p in line_cfg.points]
                self._counter.update_line(
                    (min(xs), min(ys)), (max(xs), max(ys)), relative=line_cfg.relative,
                )

            t0 = time.perf_counter()
            try:
                dets = self._detector.detect(frame)
            except Exception:
                time.sleep(0.05)
                continue
            tracks = self._tracker.update(dets, frame)
            events = self._counter.update(tracks)
            latency = (time.perf_counter() - t0) * 1000

            if events:
                try:
                    event_store.save_events(events, cam_id=self._cam_id)
                except Exception:
                    pass

            if self._head_detector is not None:
                try:
                    head_dets = self._head_detector.detect(frame)
                    self._head_tracks = self._head_tracker.update(head_dets, frame)
                except Exception:
                    self._head_tracks = []

            stats = self._counter.stats

            out = frame.copy()
            _draw_zone(out, line_cfg)
            _draw_tracks(out, self._head_tracks, line_cfg)
            _draw_hud(out, stats, self._fps)

            now = time.perf_counter()
            self._t_ring.append(now)
            fps = 0.0
            if len(self._t_ring) > 1:
                fps = (len(self._t_ring) - 1) / (self._t_ring[-1] - self._t_ring[0])

            with self._lock:
                self._annotated = out
                self._stats = stats
                self._latency_ms = round(latency, 1)
                self._fps = round(fps, 1)

    def update_line(self, line_cfg: CameraLine) -> None:
        """обновить линию - применится на след кадре"""
        with self._lock:
            self._line_cfg = line_cfg

    def reset_stats(self) -> None:
        """обнулить счётчики (треки не трогаем)"""
        self._counter.reset()
        with self._lock:
            self._stats = self._counter.stats

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._annotated.copy() if self._annotated is not None else None

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "fps": self._fps,
                "latency_ms": self._latency_ms,
                "counting": self._running,
            }

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None



class CameraManager:
    """менеджер до MAX_CAMERAS=6 камер: потоки, линии калибровки, счётчики.
    детектор грузится раз и шарится между всеми"""

    def __init__(self, max_cameras: int = MAX_CAMERAS) -> None:
        self.max_cameras = max_cameras
        self._streams: dict[str, CameraStream] = {}
        self._lines: dict[str, CameraLine] = {}
        self._counters: dict[str, _CameraCounter] = {}
        self._detector = None
        self._head_detector = None
        self._lock = threading.Lock()


    def connect(self, cam_id: str, source: str | int) -> None:
        """подключить камеру. source - RTSP, путь к файлу или int USB"""
        with self._lock:
            if cam_id in self._streams:
                old = self._streams.pop(cam_id)
                old.disconnect()
            active = len(self._streams)

        if active >= self.max_cameras:
            raise RuntimeError(
                f"Достигнут лимит: {self.max_cameras} камер. "
                "Отключите неиспользуемую камеру."
            )

        stream = CameraStream(source)
        stream.connect()

        with self._lock:
            self._streams[cam_id] = stream
            if cam_id not in self._lines:
                self._lines[cam_id] = CameraLine()

    def disconnect(self, cam_id: str) -> None:
        """стоп подсчёта + отключить камеру"""
        self.stop_counting(cam_id)
        with self._lock:
            stream = self._streams.pop(cam_id, None)
        if stream:
            stream.disconnect()

    def disconnect_all(self) -> None:
        """отключить все камеры (при остановке приложения)"""
        with self._lock:
            cam_ids = list(self._streams.keys())
        for cam_id in cam_ids:
            self.disconnect(cam_id)


    def get_frame(self, cam_id: str) -> np.ndarray | None:
        """текущий кадр. при подсчёте - аннотированный (bbox+зона+HUD), при
        калибровке - сырой (зону рисует canvas в браузере, чтоб не перекрывать)"""
        with self._lock:
            counter = self._counters.get(cam_id)
            stream = self._streams.get(cam_id)

        if counter is not None:
            return counter.get_frame()

        if stream is None:
            return None
        return stream.get_frame()


    def set_calibration(self, cam_id: str, points: list,
                        direction: str = "down_in", relative: bool = True) -> None:
        """обновить зону двери (4 угла) + направление, применится на след кадре"""
        new_zone = CameraZone(points=points, direction=direction, relative=relative)
        with self._lock:
            self._lines[cam_id] = new_zone
            counter = self._counters.get(cam_id)
        if counter is not None:
            counter.update_line(new_zone)

    def get_calibration(self, cam_id: str) -> CameraZone | None:
        with self._lock:
            return self._lines.get(cam_id)


    def _ensure_detector(self):
        """ленивая загрузка детектора при первом обращении"""
        with self._lock:
            if self._detector is None:
                self._detector = make_detector()
                self._detector.load()
            return self._detector

    def _ensure_head_detector(self):
        """детектор голов (только для UI), при ошибке - None"""
        with self._lock:
            if self._head_detector is None:
                try:
                    det = Detector(model_path=HEAD_MODEL_PATH, conf=HEAD_CONF_THRESHOLD,
                                   iou=HEAD_IOU_THRESHOLD, classes=None)
                    det.load()
                    self._head_detector = det
                except Exception:
                    self._head_detector = False
            return self._head_detector or None

    def start_counting(self, cam_id: str) -> None:
        """запустить детект+трек+счёт для камеры. первый запуск блокирующий (~2-5с на модель)"""
        with self._lock:
            stream = self._streams.get(cam_id)
            line = self._lines.get(cam_id, CameraLine())
            existing = self._counters.get(cam_id)

        if stream is None:
            raise RuntimeError(f"Камера {cam_id!r} не подключена.")
        if existing is not None and existing._running:
            return

        detector = self._ensure_detector()
        head_detector = self._ensure_head_detector()
        counter = _CameraCounter(cam_id, stream, line, detector, head_detector)
        counter.start()

        with self._lock:
            self._counters[cam_id] = counter

    def stop_counting(self, cam_id: str) -> None:
        """стоп подсчёта для камеры"""
        with self._lock:
            counter = self._counters.pop(cam_id, None)
        if counter:
            counter.stop()

    def reset_stats(self, cam_id: str) -> None:
        """обнулить счётчики, трекер не останавливаем"""
        with self._lock:
            counter = self._counters.get(cam_id)
        if counter:
            counter.reset_stats()

    def is_counting(self, cam_id: str) -> bool:
        with self._lock:
            counter = self._counters.get(cam_id)
        return counter is not None and counter._running

    def get_stats(self, cam_id: str) -> dict | None:
        with self._lock:
            counter = self._counters.get(cam_id)
        if counter is None:
            return None
        return counter.stats


    def list_cameras(self) -> list[dict]:
        with self._lock:
            result = []
            for cam_id, stream in self._streams.items():
                line = self._lines.get(cam_id, CameraLine())
                counter = self._counters.get(cam_id)
                result.append({
                    "cam_id": cam_id,
                    "source": str(stream.source),
                    "width": stream.width,
                    "height": stream.height,
                    "fps": round(stream.fps, 1),
                    "alive": stream.is_alive,
                    "counting": counter is not None and counter._running,
                    "zone": line.to_dict(),
                })
            return result

    def camera_info(self, cam_id: str) -> dict | None:
        with self._lock:
            stream = self._streams.get(cam_id)
            if stream is None:
                return None
            line = self._lines.get(cam_id, CameraLine())
            counter = self._counters.get(cam_id)
        return {
            "cam_id": cam_id,
            "source": str(stream.source),
            "width": stream.width,
            "height": stream.height,
            "fps": round(stream.fps, 1),
            "alive": stream.is_alive,
            "counting": counter is not None and counter._running,
            "zone": line.to_dict(),
        }
