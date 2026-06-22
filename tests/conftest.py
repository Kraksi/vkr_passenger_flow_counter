"""общие фикстуры для тестов"""
from __future__ import annotations
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_frame() -> np.ndarray:
    """кадр 640x480 BGR"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[dict]:
    """детекции двух людей"""
    return [
        {"bbox": [100.0, 100.0, 200.0, 300.0], "conf": 0.85, "class": 0},
        {"bbox": [400.0, 150.0, 500.0, 350.0], "conf": 0.72, "class": 0},
    ]


@pytest.fixture
def sample_tracks() -> list[dict]:
    """треки двух людей"""
    return [
        {"track_id": 1, "bbox": [100.0, 100.0, 200.0, 300.0], "conf": 0.85},
        {"track_id": 2, "bbox": [400.0, 150.0, 500.0, 350.0], "conf": 0.72},
    ]


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """путь к временной SQLite БД"""
    return tmp_path / "test_events.db"
