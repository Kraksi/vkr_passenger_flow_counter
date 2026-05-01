# Подсчёт событий вход/выход через виртуальную линию с гистерезисом
from __future__ import annotations
import time
from app.config import (
    LINE_START,
    LINE_END,
    LINE_COORDS_RELATIVE,
    COUNTER_HYSTERESIS_MIN_SEC,
    COUNTER_HYSTERESIS_MAX_SEC,
)


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Знак определяет по какую сторону от отрезка AB лежит точка O."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class LineCounter:
    """
    Считает пересечения треков через заданную линию.

    Направление определяется знаком cross product при пересечении:
    положительный → entry (вход), отрицательный → exit (выход).
    Ориентация настраивается через порядок точек линии.
    """

    def __init__(
        self,
        line_start: tuple[float, float] = LINE_START,
        line_end: tuple[float, float] = LINE_END,
        relative: bool = LINE_COORDS_RELATIVE,
        hysteresis_min: float = COUNTER_HYSTERESIS_MIN_SEC,
        hysteresis_max: float = COUNTER_HYSTERESIS_MAX_SEC,
    ) -> None:
        self._line_start_raw = line_start
        self._line_end_raw = line_end
        self._relative = relative
        self._frame_w: int = 0
        self._frame_h: int = 0

        self.line_start: tuple[float, float] = line_start
        self.line_end: tuple[float, float] = line_end

        self.hysteresis_min = hysteresis_min
        self.hysteresis_max = hysteresis_max

        self.entries: int = 0
        self.exits: int = 0

        self._track_state: dict[int, dict] = {}

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Сообщить счётчику размер кадра.
        Если координаты относительные — масштабирует их в пиксели.
        Вызывается из pipeline при открытии видео/камеры.
        """
        self._frame_w = width
        self._frame_h = height
        if self._relative:
            self.line_start = (
                self._line_start_raw[0] * width,
                self._line_start_raw[1] * height,
            )
            self.line_end = (
                self._line_end_raw[0] * width,
                self._line_end_raw[1] * height,
            )

    def update_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        relative: bool = False,
    ) -> None:
        """
        Обновить координаты виртуальной линии (через API /calibration).
        relative=True — координаты в диапазоне 0.0–1.0.
        """
        self._relative = relative
        self._line_start_raw = start
        self._line_end_raw = end
        if relative and self._frame_w > 0:
            self.line_start = (start[0] * self._frame_w, start[1] * self._frame_h)
            self.line_end = (end[0] * self._frame_w, end[1] * self._frame_h)
        else:
            self.line_start = start
            self.line_end = end

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        """Центр bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _side(self, point: tuple[float, float]) -> int:
        """С какой стороны от линии лежит точка: +1, -1 или 0."""
        c = _cross(point, self.line_start, self.line_end)
        if c > 0:
            return 1
        elif c < 0:
            return -1
        return 0

    def update(self, tracks: list[dict]) -> list[dict]:
        """
        Принять треки кадра, зафиксировать новые события пересечения.

        Возвращает список новых событий: [{'track_id': int, 'event': 'entry'|'exit'}]
        """
        events = []
        now = time.monotonic()
        active_ids = set()

        for track in tracks:
            tid = track["track_id"]
            center = self._center(track["bbox"])
            side = self._side(center)
            active_ids.add(tid)

            if tid not in self._track_state:
                self._track_state[tid] = {
                    "prev_side": side,
                    "last_event_time": 0.0,
                }
                continue

            state = self._track_state[tid]
            prev_side = state["prev_side"]

            if prev_side != 0 and side != 0 and prev_side != side:
                elapsed = now - state["last_event_time"]
                if state["last_event_time"] == 0.0 or elapsed >= self.hysteresis_min:
                    if side > 0:
                        event_type = "entry"
                        self.entries += 1
                    else:
                        event_type = "exit"
                        self.exits += 1
                    events.append({"track_id": tid, "event": event_type})
                    state["last_event_time"] = now

            state["prev_side"] = side

        stale = [
            tid for tid, st in self._track_state.items()
            if tid not in active_ids and (now - st["last_event_time"]) > self.hysteresis_max
        ]
        for tid in stale:
            del self._track_state[tid]

        return events

    def reset(self) -> None:
        """Сбросить счётчики и состояния треков."""
        self.entries = 0
        self.exits = 0
        self._track_state.clear()

    @property
    def stats(self) -> dict:
        return {
            "entries": self.entries,
            "exits": self.exits,
            "total": self.entries + self.exits,
            "current_inside": max(0, self.entries - self.exits),
        }
