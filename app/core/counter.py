from __future__ import annotations

from app.config import (
    LINE_START,
    LINE_END,
    LINE_COORDS_RELATIVE,
    COUNTER_HYSTERESIS_MIN_SEC,
    COUNTER_HYSTERESIS_MAX_SEC,
    TRIPWIRE_Y1,
    TRIPWIRE_Y2,
    TRIPWIRE_ROI_X,
    TRIPWIRE_WINDOW_FRAMES,
    TRIPWIRE_MIN_FRAMES,
    TRIPWIRE_LOST_FRAMES,
    TRIPWIRE_DEDUP_FRAMES,
    TRIPWIRE_DEDUP_X_FRAC,
    TRIPWIRE_CONVENTION,
)


class LineCounter:
    """подсчёт по жизненному циклу трека: трек пропал - направление по знаку
    вертикального смещения. min_travel отсекает стоячих"""

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

        self.min_travel_px: float = 0.0
        self.min_travel_rel: float = 0.03

        self.lost_frames: int = 5

        self.entries: int = 0
        self.exits: int = 0

        self._tracks: dict[int, dict] = {}
        self._frame_idx: int = 0

    def set_frame_size(self, width: int, height: int) -> None:
        """размер кадра"""
        self._frame_w = width
        self._frame_h = height
        self.min_travel_px = self.min_travel_rel * height
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
        """обновить линию (из /calibration)"""
        self._relative = relative
        self._line_start_raw = start
        self._line_end_raw = end
        if relative and self._frame_w > 0:
            self.line_start = (start[0] * self._frame_w, start[1] * self._frame_h)
            self.line_end = (end[0] * self._frame_w, end[1] * self._frame_h)
        else:
            self.line_start = start
            self.line_end = end

    def _center_y(self, bbox: list[float]) -> float:
        """верт центр bbox"""
        return (bbox[1] + bbox[3]) / 2

    def _finalize_track(self, tid: int, state: dict) -> dict | None:
        """определить направление - событие, или None если не прошёл фильтры"""
        travel = state["max_y"] - state["min_y"]
        if travel < self.min_travel_px:
            return None

        dy = state["last_y"] - state["first_y"]
        if dy < 0:
            self.entries += 1
            return {"track_id": tid, "event": "entry"}
        elif dy > 0:
            self.exits += 1
            return {"track_id": tid, "event": "exit"}
        return None

    def update(self, tracks: list[dict]) -> list[dict]:
        """принять треки кадра, финализировать пропавшие - список новых событий"""
        events = []
        self._frame_idx += 1
        active_ids = set()

        for track in tracks:
            tid = track["track_id"]
            cy = self._center_y(track["bbox"])
            active_ids.add(tid)

            if tid not in self._tracks:
                self._tracks[tid] = {
                    "first_y": cy,
                    "last_y": cy,
                    "min_y": cy,
                    "max_y": cy,
                    "last_seen": self._frame_idx,
                }
            else:
                s = self._tracks[tid]
                s["last_y"] = cy
                s["min_y"] = min(s["min_y"], cy)
                s["max_y"] = max(s["max_y"], cy)
                s["last_seen"] = self._frame_idx

        lost = []
        for tid, s in self._tracks.items():
            if tid not in active_ids:
                if self._frame_idx - s["last_seen"] >= self.lost_frames:
                    event = self._finalize_track(tid, s)
                    if event:
                        events.append(event)
                    lost.append(tid)

        for tid in lost:
            del self._tracks[tid]

        return events

    def finalize_remaining(self) -> list[dict]:
        """добить все оставшиеся треки (конец видео)"""
        events = []
        for tid, s in self._tracks.items():
            event = self._finalize_track(tid, s)
            if event:
                events.append(event)
        self._tracks.clear()
        return events

    def reset(self) -> None:
        """сброс счётчиков и треков"""
        self.entries = 0
        self.exits = 0
        self._tracks.clear()
        self._frame_idx = 0

    @property
    def stats(self) -> dict:
        return {
            "entries": self.entries,
            "exits": self.exits,
            "total": self.entries + self.exits,
            "current_inside": max(0, self.entries - self.exits),
        }


class ZoneCounter:
    """счёт по смещению внутри зоны двери (хорош для голов): направление по знаку
    суммарного dy, без пересечения линий - устойчивее к коротким трекам. зона
    отсекает стоячих в салоне, направление задаётся явно. совместим с LineCounter"""

    def __init__(
        self,
        zone=(0.0, 0.0, 0.45, 1.0),
        direction: str = "down_in",
        travel_rel: float = 0.20,
        min_frames: int = 5,
        lost_frames: int = 100,
    ) -> None:
        self.poly = self._to_poly(zone)
        self.direction = direction
        self.travel_rel = travel_rel
        self.min_frames = min_frames
        self.lost_frames = lost_frames

        self._frame_w = 0
        self._frame_h = 0
        self.entries = 0
        self.exits = 0
        self._tracks: dict[int, dict] = {}
        self._frame_idx = 0

    @staticmethod
    def _to_poly(zone) -> list:
        """зона - 4 угла в относит координатах. принимает (x1,y1,x2,y2) или [[x,y]*4]"""
        if zone and isinstance(zone[0], (list, tuple)):
            return [(float(p[0]), float(p[1])) for p in zone]
        x1, y1, x2, y2 = zone
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_w = width
        self._frame_h = height

    def update_zone(self, points=None, direction=None, relative=True,
                    x1=None, y1=None, x2=None, y2=None) -> None:
        """обновить зону (4 угла или прямоугольник) + направление"""
        if points is not None:
            poly = [(float(px), float(py)) for px, py in points]
            if not relative and self._frame_w:
                poly = [(px / self._frame_w, py / self._frame_h) for px, py in poly]
            self.poly = poly
        elif x1 is not None:
            if not relative and self._frame_w:
                x1, x2 = x1 / self._frame_w, x2 / self._frame_w
                y1, y2 = y1 / self._frame_h, y2 / self._frame_h
            self.poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if direction:
            self.direction = direction

    def update_line(self, start, end, relative: bool = False) -> None:
        self.update_zone(x1=start[0], y1=start[1], x2=end[0], y2=end[1], relative=relative)

    def _in_zone(self, px, py) -> bool:
        """точка внутри полигона, координаты относит"""
        poly = self.poly
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _finalize(self, s: dict) -> dict | None:
        if s["n"] < self.min_frames:
            return None
        if self._frame_h and (s["max_y"] - s["min_y"]) < self.travel_rel * self._frame_h:
            return None
        if self._frame_w:
            cx_rel = (s["sum_x"] / s["n"]) / self._frame_w
            cy_rel = (s["sum_y"] / s["n"]) / self._frame_h
            if not self._in_zone(cx_rel, cy_rel):
                return None
        dy = s["last_y"] - s["first_y"]
        if dy == 0:
            return None
        down = dy > 0
        is_entry = down if self.direction == "down_in" else (not down)
        if is_entry:
            self.entries += 1
            return {"track_id": -1, "event": "entry"}
        self.exits += 1
        return {"track_id": -1, "event": "exit"}

    def update(self, tracks: list[dict]) -> list[dict]:
        self._frame_idx += 1
        active = set()
        for t in tracks:
            tid = t["track_id"]
            bb = t["bbox"]
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            active.add(tid)
            s = self._tracks.get(tid)
            if s is None:
                self._tracks[tid] = {
                    "first_y": cy, "last_y": cy, "min_y": cy, "max_y": cy,
                    "sum_x": cx, "sum_y": cy, "n": 1, "last_seen": self._frame_idx,
                }
            else:
                s["last_y"] = cy
                s["min_y"] = min(s["min_y"], cy); s["max_y"] = max(s["max_y"], cy)
                s["sum_x"] += cx; s["sum_y"] += cy; s["n"] += 1
                s["last_seen"] = self._frame_idx
        events = []
        lost = [tid for tid, s in self._tracks.items()
                if tid not in active and self._frame_idx - s["last_seen"] >= self.lost_frames]
        for tid in lost:
            ev = self._finalize(self._tracks.pop(tid))
            if ev:
                events.append(ev)
        return events

    def finalize_remaining(self) -> list[dict]:
        events = []
        for tid in list(self._tracks):
            ev = self._finalize(self._tracks.pop(tid))
            if ev:
                events.append(ev)
        return events

    def reset(self) -> None:
        self.entries = 0
        self.exits = 0
        self._tracks.clear()
        self._frame_idx = 0

    @property
    def stats(self) -> dict:
        return {
            "entries": self.entries,
            "exits": self.exits,
            "total": self.entries + self.exits,
            "current_inside": max(0, self.entries - self.exits),
        }


class TripwireCounter:
    """двухлинейный трипвайр - основной счётчик. door1 F1~65%, door2~62% в проде,
    офлайн-потолок 69/66

    две гориз линии y1<y2 в дверном проёме, из конфига, вручную не калибруются.
    направление по порядку пересечения: сначала y1 потом y2 = вниз = вход (conv B).
    калибруется только ROI по X. полный проход через обе линии - устойчив к
    фрагментации id и однозначно даёт направление. финал по концу трека (через
    lost_frames кадров после ухода). совместим с LineCounter"""

    def __init__(
        self,
        y1: float = TRIPWIRE_Y1,
        y2: float = TRIPWIRE_Y2,
        roi_x: tuple[float, float] = TRIPWIRE_ROI_X,
        window_frames: int = TRIPWIRE_WINDOW_FRAMES,
        min_frames: int = TRIPWIRE_MIN_FRAMES,
        lost_frames: int = TRIPWIRE_LOST_FRAMES,
        dedup_frames: int = TRIPWIRE_DEDUP_FRAMES,
        dedup_x_frac: float = TRIPWIRE_DEDUP_X_FRAC,
        convention: str = TRIPWIRE_CONVENTION,
    ) -> None:
        self.y1_rel = min(y1, y2)
        self.y2_rel = max(y1, y2)
        self.roi_x_rel = roi_x
        self.window_frames = window_frames
        self.min_frames = min_frames
        self.lost_frames = lost_frames
        self.dedup_frames = dedup_frames
        self.dedup_x_frac = dedup_x_frac
        self.convention = convention

        self._frame_w = 0
        self._frame_h = 0
        self.y1_px = 0.0
        self.y2_px = 0.0
        self.roi_xlo_px = 0.0
        self.roi_xhi_px = 0.0
        self.dedup_x_px = 0.0

        self.entries = 0
        self.exits = 0
        self._tracks: dict[int, dict] = {}
        self._recent: list[tuple[float, float, int]] = []
        self._frame_idx = 0
        self._max_points = max(window_frames * 3, 360)

    def set_frame_size(self, width: int, height: int) -> None:
        """размер кадра - пересчёт линий и ROI в пиксели"""
        self._frame_w = width
        self._frame_h = height
        self.y1_px = self.y1_rel * height
        self.y2_px = self.y2_rel * height
        self.roi_xlo_px = self.roi_x_rel[0] * width
        self.roi_xhi_px = self.roi_x_rel[1] * width
        self.dedup_x_px = self.dedup_x_frac * width

    def update_zone(self, points=None, direction=None, relative: bool = True,
                    x1=None, y1=None, x2=None, y2=None) -> None:
        """калибровка зоной двери: задаёт только диапазон по X (какие треки учитывать,
        отсекает салон и соседние двери). линии y1/y2 фиксированы. down_in - conv B"""
        xs = None
        if points is not None:
            xs = [float(p[0]) for p in points]
        elif x1 is not None:
            xs = [float(x1), float(x2)]
        if xs is not None:
            if not relative and self._frame_w:
                xs = [x / self._frame_w for x in xs]
            self.roi_x_rel = (max(0.0, min(xs)), min(1.0, max(xs)))
        if direction:
            self.convention = "B" if direction == "down_in" else "A"
        if self._frame_w:
            self.set_frame_size(self._frame_w, self._frame_h)

    def update_line(self, start, end, relative: bool = False) -> None:
        """совместимость с /calibration: одиночная линия = центр зоны, y1/y2
        симметрично вокруг неё"""
        cy = (start[1] + end[1]) / 2
        if not relative and self._frame_h:
            cy = cy / self._frame_h
        half = (self.y2_rel - self.y1_rel) / 2
        self.y1_rel = max(0.0, cy - half)
        self.y2_rel = min(1.0, cy + half)
        if self._frame_h:
            self.set_frame_size(self._frame_w, self._frame_h)

    @staticmethod
    def _center(bbox: list[float]) -> tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    @staticmethod
    def _first_cross(ys, line: float) -> int | None:
        """индекс первого пересечения линии массивом ys (или None)"""
        for i in range(1, len(ys)):
            a, b = ys[i - 1] - line, ys[i] - line
            if (a <= 0 < b) or (a >= 0 > b):
                return i
        return None

    def _classify_track(self, pts: list) -> tuple[float, float, int] | None:
        """кандидат события: (frame, cx, dir) или None. dir +1 вниз (y1-y2), -1 вверх"""
        if len(pts) < self.min_frames:
            return None
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        mean_x = sum(xs) / len(xs)
        if not (self.roi_xlo_px <= mean_x <= self.roi_xhi_px):
            return None
        i1 = self._first_cross(ys, self.y1_px)
        i2 = self._first_cross(ys, self.y2_px)
        if i1 is None or i2 is None:
            return None
        f1, f2 = pts[i1][0], pts[i2][0]
        if abs(f1 - f2) > self.window_frames:
            return None
        direction = 1 if f1 < f2 else (-1 if f2 < f1 else 0)
        if direction == 0:
            return None
        cx_door = (pts[i1][1] + pts[i2][1]) / 2
        return (min(f1, f2), cx_door, direction)

    def _finalize_track(self, pts: list) -> dict | None:
        """классифицировать трек, проверить дубль - событие"""
        cand = self._classify_track(pts)
        if cand is None:
            return None
        f, cx, d = cand
        self._recent = [r for r in self._recent if abs(f - r[0]) < self.dedup_frames]
        for fk, xk, dk in self._recent:
            if dk == d and abs(cx - xk) < self.dedup_x_px:
                return None
        self._recent.append((f, cx, d))
        is_entry = d > 0
        if self.convention != "B":
            is_entry = not is_entry
        if is_entry:
            self.entries += 1
            return {"track_id": -1, "event": "entry"}
        self.exits += 1
        return {"track_id": -1, "event": "exit"}

    def update(self, tracks: list[dict]) -> list[dict]:
        """копим точки траекторий; финализируем треки пропавшие на lost_frames"""
        self._frame_idx += 1
        active = set()
        for track in tracks:
            tid = track["track_id"]
            cx, cy = self._center(track["bbox"])
            active.add(tid)
            s = self._tracks.get(tid)
            if s is None:
                s = {"pts": [], "last_seen": self._frame_idx}
                self._tracks[tid] = s
            s["pts"].append((self._frame_idx, cx, cy))
            if len(s["pts"]) > self._max_points:
                del s["pts"][0]
            s["last_seen"] = self._frame_idx

        lost = [tid for tid, s in self._tracks.items()
                if tid not in active and self._frame_idx - s["last_seen"] >= self.lost_frames]
        lost.sort(key=lambda t: self._tracks[t]["pts"][0][0])
        events = []
        for tid in lost:
            ev = self._finalize_track(self._tracks.pop(tid)["pts"])
            if ev:
                events.append(ev)
        return events

    def finalize_remaining(self) -> list[dict]:
        """добить оставшиеся треки (конец видео)"""
        events = []
        for tid in sorted(self._tracks, key=lambda t: self._tracks[t]["pts"][0][0]):
            ev = self._finalize_track(self._tracks[tid]["pts"])
            if ev:
                events.append(ev)
        self._tracks.clear()
        return events

    def reset(self) -> None:
        self.entries = 0
        self.exits = 0
        self._tracks.clear()
        self._recent.clear()
        self._frame_idx = 0

    @property
    def stats(self) -> dict:
        return {
            "entries": self.entries,
            "exits": self.exits,
            "total": self.entries + self.exits,
            "current_inside": max(0, self.entries - self.exits),
        }
