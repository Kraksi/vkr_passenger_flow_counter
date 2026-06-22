from __future__ import annotations
import numpy as np
import supervision as sv
from app.config import (
    TRACKER_TRACK_THRESH,
    TRACKER_MATCH_THRESH,
    TRACKER_TRACK_BUFFER,
    TRACKER_FRAME_RATE,
)


class Tracker:
    """обёртка над ByteTrack (supervision), сшивает детекции между кадрами"""

    def __init__(
        self,
        track_thresh: float = TRACKER_TRACK_THRESH,
        match_thresh: float = TRACKER_MATCH_THRESH,
        track_buffer: int = TRACKER_TRACK_BUFFER,
        frame_rate: int = TRACKER_FRAME_RATE,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self._tracker: sv.ByteTrack | None = None

    def init(self) -> None:
        """создать трекер"""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            minimum_matching_threshold=self.match_thresh,
            lost_track_buffer=self.track_buffer,
            frame_rate=self.frame_rate,
        )

    def reset(self) -> None:
        """сброс треков (смена видео)"""
        self.init()

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """детекции кадра - активные треки {track_id, bbox, conf}"""
        if self._tracker is None:
            raise RuntimeError("Трекер не инициализирован. Вызовите init().")

        if not detections:
            self._tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        conf = np.array([d["conf"] for d in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_dets)

        tracks = []
        if tracked.tracker_id is not None:
            for i in range(len(tracked.xyxy)):
                tracks.append({
                    "track_id": int(tracked.tracker_id[i]),
                    "bbox": tracked.xyxy[i].tolist(),
                    "conf": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                })
        return tracks
