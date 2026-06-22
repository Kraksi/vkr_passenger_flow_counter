from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import (
    REID_WEIGHTS_PATH,
    OSNET_TRACK_HIGH_THRESH,
    OSNET_NEW_TRACK_THRESH,
    OSNET_MATCH_THRESH,
    OSNET_TRACK_BUFFER,
    OSNET_APPEARANCE_THRESH,
    OSNET_CMC_METHOD,
    TRACKER_FRAME_RATE,
    DETECTOR_DEVICE,
)


class TrackerOSNet:
    """обёртка над boxmot BotSort с OSNet ReID"""

    def __init__(
        self,
        reid_weights: Path = REID_WEIGHTS_PATH,
        track_high_thresh: float = OSNET_TRACK_HIGH_THRESH,
        new_track_thresh: float = OSNET_NEW_TRACK_THRESH,
        match_thresh: float = OSNET_MATCH_THRESH,
        track_buffer: int = OSNET_TRACK_BUFFER,
        appearance_thresh: float = OSNET_APPEARANCE_THRESH,
        cmc_method: str = OSNET_CMC_METHOD,
        frame_rate: int = TRACKER_FRAME_RATE,
        device: str = DETECTOR_DEVICE,
    ) -> None:
        self.reid_weights = Path(reid_weights)
        self.track_high_thresh = track_high_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.appearance_thresh = appearance_thresh
        self.cmc_method = cmc_method
        self.frame_rate = frame_rate
        self.device = device
        self._tracker = None

    def init(self) -> None:
        """создать трекер (грузит OSNet в gpu)"""
        import torch
        from boxmot import BotSort

        self._tracker = BotSort(
            reid_weights=self.reid_weights,
            device=torch.device(self.device if ":" in self.device else f"{self.device}:0"),
            half=False,
            track_high_thresh=self.track_high_thresh,
            new_track_thresh=self.new_track_thresh,
            match_thresh=self.match_thresh,
            track_buffer=self.track_buffer,
            appearance_thresh=self.appearance_thresh,
            cmc_method=self.cmc_method,
            frame_rate=self.frame_rate,
            with_reid=True,
        )

    def reset(self) -> None:
        """сброс треков (смена видео)"""
        self.init()

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """детекции кадра - активные треки {track_id, bbox, conf}"""
        if self._tracker is None:
            raise RuntimeError("Трекер не инициализирован. Вызовите init().")

        if detections:
            dets = np.array(
                [[*d["bbox"], d["conf"], 0] for d in detections], dtype=np.float32
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        res = self._tracker.update(dets, frame)
        tracks = []
        for row in res:
            tracks.append({
                "track_id": int(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "conf": float(row[5]),
            })
        return tracks
