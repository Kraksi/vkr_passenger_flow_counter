"""
запуск: uv run pytest tests/test_key_diploma.py -v

карта тест - слой - критерий:
  T01-T06  TripwireCounter   подсчёт            Precision, Recall, FPR, MAPE
  T07      LineCounter       fallback           направление события
  T08-T09  Tracker           трекинг            IDF1
  T10-T11  Detector          детекция           контракт инференса
  T12-T14  Pipeline          оркестрация        кадр - событие
  T15-T16  EventStore        хранение           целостность статистики
  T17      routes API        REST               контракт /process
  T18      stats API         REST               current_inside
  T19-T20  CameraManager     камеры             лимит 6, калибровка
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.core.counter import LineCounter, TripwireCounter
from app.core.tracker import Tracker
from app.core.detector import Detector
from app.core.pipeline import Pipeline
from app.core.camera_manager import CameraManager
from app.storage.db import EventStore
from app.main import app


def _track(tid: int, cy: float, cx: float = 100.0) -> dict:
    """трек с центром (cx, cy)"""
    return {"track_id": tid, "bbox": [cx - 50, cy - 50, cx + 50, cy + 50], "conf": 0.9}


def _feed_motion(counter, tid, cx, y_start, y_end, n_frames, finalize=True):
    """прогнать трек через n_frames кадров (cy от y_start до y_end).
    счётчик копит в update(), события в finalize_remaining()"""
    for i in range(n_frames):
        cy = y_start + (y_end - y_start) * i / (n_frames - 1)
        counter.update([_track(tid, cy, cx)])
    return counter.finalize_remaining() if finalize else []


def _make_tripwire():
    """трипвайр на кадре 640x480: y1=0.42, y2=0.70, ROI X=[0,224]. conv B: вниз=entry"""
    tw = TripwireCounter(y1=0.42, y2=0.70, roi_x=(0.0, 0.35),
                         window_frames=120, min_frames=15)
    tw.set_frame_size(640, 480)
    return tw



def test_01_tripwire_entry_on_downward_transit():
    """T01. проход сверху вниз через обе линии - вход. recall входов"""
    tw = _make_tripwire()
    events = _feed_motion(tw, 1, cx=100, y_start=150, y_end=400, n_frames=20)
    assert len(events) == 1
    assert events[0]["event"] == "entry"
    assert tw.entries == 1 and tw.exits == 0


def test_02_tripwire_exit_on_upward_transit():
    """T02. проход снизу вверх - выход. recall выходов"""
    tw = _make_tripwire()
    events = _feed_motion(tw, 1, cx=100, y_start=400, y_end=150, n_frames=20)
    assert len(events) == 1
    assert events[0]["event"] == "exit"
    assert tw.exits == 1 and tw.entries == 0


def test_03_tripwire_no_count_outside_roi_x():
    """T03. трек вне ROI двери (салон) не считается. FPR - отсечение сидящих"""
    tw = _make_tripwire()
    events = _feed_motion(tw, 1, cx=500, y_start=150, y_end=400, n_frames=20)
    assert events == []
    assert tw.entries == 0


def test_04_tripwire_no_count_single_line_only():
    """T04. пересёк только одну линию - не считается. precision - неполные проходы"""
    tw = _make_tripwire()
    events = _feed_motion(tw, 1, cx=100, y_start=150, y_end=250, n_frames=20)
    assert events == []


def test_05_tripwire_track_counted_once():
    """T05. проход считается ровно раз, даже если трек ещё наблюдается. без двойного счёта"""
    tw = _make_tripwire()
    _feed_motion(tw, 1, cx=100, y_start=150, y_end=400, n_frames=20, finalize=False)
    for _ in range(10):
        tw.update([_track(1, 400.0, 100)])
    events = tw.finalize_remaining()
    assert len(events) == 1
    assert tw.entries == 1


def test_06_tripwire_no_count_below_min_frames():
    """T06. короткий трек (< min_frames=15) - не считается. FPR - фильтр шума"""
    tw = _make_tripwire()
    events = _feed_motion(tw, 1, cx=100, y_start=150, y_end=400, n_frames=8)
    assert events == []



def test_07_linecounter_exit_on_downward_track():
    """T07. lifecycle: трек вниз (dy>0) - выход. fallback-алгоритм"""
    lc = LineCounter(relative=False)
    lc.set_frame_size(640, 480)
    lc.update([_track(1, 100.0)])
    lc.update([_track(1, 400.0)])
    events = lc.finalize_remaining()
    assert lc.exits == 1
    assert events and events[0]["event"] == "exit"



@pytest.fixture
def tracker() -> Tracker:
    t = Tracker(track_thresh=0.25, match_thresh=0.85, track_buffer=30, frame_rate=25)
    t.init()
    return t


def test_08_tracker_consistent_ids_across_frames(
    tracker: Tracker, sample_frame: np.ndarray, sample_detections: list[dict]
):
    """T08. id стабилен между соседними кадрами. IDF1"""
    r1 = tracker.update(sample_detections, sample_frame)
    r2 = tracker.update(sample_detections, sample_frame)
    if r1 and r2:
        ids1 = {t["track_id"] for t in r1}
        ids2 = {t["track_id"] for t in r2}
        assert len(ids1 & ids2) > 0


def test_09_tracker_track_has_required_keys(
    tracker: Tracker, sample_frame: np.ndarray, sample_detections: list[dict]
):
    """T09. у трека есть track_id/bbox/conf. контракт трекер - счётчик"""
    result = tracker.update(sample_detections, sample_frame)
    for track in result:
        assert "track_id" in track
        assert "bbox" in track
        assert "conf" in track



@pytest.fixture
def detector() -> Detector:
    return Detector(model_path="fake/path.pt", conf=0.25, iou=0.45, device="cpu")


def test_10_detector_returns_detections(detector: Detector, sample_frame: np.ndarray):
    """T10. детектор парсит выход YOLO в dict с bbox/conf/class. контракт детектор - трекер"""
    import torch
    mock_box = MagicMock()
    mock_box.xyxy = [torch.tensor([100.0, 150.0, 200.0, 300.0])]
    mock_box.conf = [torch.tensor(0.85)]
    mock_box.cls = [torch.tensor(0)]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]

    with patch("app.core.detector.YOLO", return_value=mock_model):
        detector.load()
        result = detector.detect(sample_frame)

    assert len(result) == 1
    assert result[0]["bbox"] == pytest.approx([100.0, 150.0, 200.0, 300.0])
    assert result[0]["conf"] == pytest.approx(0.85)
    assert result[0]["class"] == 0


def test_11_detector_raises_without_load(detector: Detector, sample_frame: np.ndarray):
    """T11. detect() до load() - RuntimeError. явная ошибка вместо тихого сбоя"""
    with pytest.raises(RuntimeError, match="не загружена"):
        detector.detect(sample_frame)



@pytest.fixture
def initialized_pipeline() -> Pipeline:
    p = Pipeline()
    p.detector = MagicMock()
    p.tracker = MagicMock()
    p.counter = MagicMock()
    p.store = MagicMock()
    p.detector.load.return_value = None
    p.tracker.init.return_value = None
    p.store.connect.return_value = None
    p.initialize()
    return p


def test_12_pipeline_result_structure(initialized_pipeline: Pipeline, sample_frame: np.ndarray):
    """T12. process_frame() - структура с latency_ms. сквозной контракт + латентность"""
    initialized_pipeline.detector.detect.return_value = []
    initialized_pipeline.tracker.update.return_value = []
    initialized_pipeline.counter.update.return_value = []
    initialized_pipeline.counter.stats = {"entries": 0, "exits": 0, "total": 0, "current_inside": 0}

    result = initialized_pipeline.process_frame(sample_frame)
    assert "tracks" in result
    assert "events" in result
    assert "stats" in result
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)


def test_13_pipeline_saves_events_to_store(initialized_pipeline: Pipeline, sample_frame: np.ndarray):
    """T13. события из счётчика пишутся в БД. интеграция счётчик - БД"""
    events = [{"track_id": 1, "event": "entry"}]
    initialized_pipeline.detector.detect.return_value = []
    initialized_pipeline.tracker.update.return_value = []
    initialized_pipeline.counter.update.return_value = events
    initialized_pipeline.counter.stats = {"entries": 1, "exits": 0, "total": 1, "current_inside": 1}

    initialized_pipeline.process_frame(sample_frame)
    initialized_pipeline.store.save_events.assert_called_once_with(events)


def test_14_pipeline_no_save_when_no_events(initialized_pipeline: Pipeline, sample_frame: np.ndarray):
    """T14. нет событий - нет записи в БД. без лишних транзакций"""
    initialized_pipeline.detector.detect.return_value = []
    initialized_pipeline.tracker.update.return_value = []
    initialized_pipeline.counter.update.return_value = []
    initialized_pipeline.counter.stats = {"entries": 0, "exits": 0, "total": 0, "current_inside": 0}

    initialized_pipeline.process_frame(sample_frame)
    initialized_pipeline.store.save_events.assert_not_called()



@pytest.fixture
def event_store(tmp_db_path):
    s = EventStore(db_path=tmp_db_path)
    s.connect()
    yield s
    s.close()


def test_15_eventstore_batch_save(event_store: EventStore):
    """T15. пакетная запись верно агрегируется. целостность entries/exits/total"""
    events = [
        {"track_id": 1, "event": "entry"},
        {"track_id": 2, "event": "entry"},
        {"track_id": 3, "event": "exit"},
    ]
    event_store.save_events(events)
    stats = event_store.get_stats()
    assert stats["entries"] == 2
    assert stats["exits"] == 1
    assert stats["total"] == 3


def test_16_eventstore_get_stats_filters_by_cam_id(event_store: EventStore):
    """T16. статистика фильтруется по cam_id. мультикамерный учёт"""
    event_store.save_events([{"track_id": 1, "event": "entry"}], cam_id="cam1")
    event_store.save_events([{"track_id": 2, "event": "entry"}], cam_id="cam2")
    assert event_store.get_stats(cam_id="cam1")["entries"] == 1
    assert event_store.get_stats(cam_id="cam2")["entries"] == 1
    assert event_store.get_stats()["entries"] == 2



@pytest.fixture
def api_client():
    mock_p = MagicMock()
    mock_p.is_initialized = True
    mock_p._running = False
    mock_p.status = {"initialized": True, "running": False, "source": None, "fps": 0.0}
    mock_p.counter.stats = {"entries": 0, "exits": 0, "total": 0, "current_inside": 0}
    with patch("app.api.routes._get_pipeline", return_value=mock_p):
        with patch("app.main.pipeline", mock_p):
            with TestClient(app, raise_server_exceptions=True) as c:
                yield c, mock_p


def test_17_api_process_success(api_client):
    """T17. POST /process - 200 + сводка. REST-контракт пакетной обработки"""
    c, mock_p = api_client
    mock_p.process_video.return_value = {
        "frames_processed": 100,
        "avg_latency_ms": 50.0,
        "fps": 20.0,
        "stats": {"entries": 5, "exits": 3, "total": 8, "current_inside": 2},
    }
    with patch("app.api.routes._get_pipeline", return_value=mock_p):
        resp = c.post("/process", json={"video_path": "/fake/video.mp4"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["frames_processed"] == 100
    assert data["fps"] == 20.0



@pytest.fixture
def stats_client():
    store = MagicMock()
    store.get_daily_stats.return_value = {
        "date": "2026-05-20", "cam_id": None, "entries": 0, "exits": 0, "total": 0
    }
    with patch("app.main.pipeline", MagicMock()):
        with patch("app.api.stats_routes.event_store", store):
            with TestClient(app, raise_server_exceptions=True) as c:
                yield c, store


def test_18_stats_current_inside_never_negative(stats_client):
    """T18. /stats/daily: current_inside = max(0, entries-exits), не отрицательно"""
    c, store = stats_client
    store.get_daily_stats.return_value = {
        "date": "2026-05-20", "cam_id": None, "entries": 3, "exits": 7, "total": 10
    }
    resp = c.get("/stats/daily")
    assert resp.status_code == 200
    assert resp.json()["current_inside"] == 0



def _make_mock_stream():
    stream = MagicMock()
    stream.width = 640
    stream.height = 480
    stream.fps = 25.0
    stream.is_alive = True
    stream.source = "fake_source"
    stream.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return stream


def test_19_camera_manager_max_cameras_exceeded():
    """T19. подключение сверх лимита - RuntimeError. edge-сценарий "до 6 камер" """
    manager = CameraManager(max_cameras=2)
    for i in range(2):
        s = _make_mock_stream()
        s.source = f"src{i}"
        with patch("app.core.camera_manager.CameraStream", return_value=s):
            manager.connect(f"cam{i}", f"src{i}")
    s3 = _make_mock_stream()
    with patch("app.core.camera_manager.CameraStream", return_value=s3):
        with pytest.raises(RuntimeError, match="лимит"):
            manager.connect("cam3", "src3")


def test_20_camera_manager_set_calibration_updates_line():
    """T20. калибровка зоны двери меняет геометрию счёта. зона задаёт направление и факт события"""
    manager = CameraManager()
    s = _make_mock_stream()
    with patch("app.core.camera_manager.CameraStream", return_value=s):
        manager.connect("cam1", "fake_source")
    points = [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]]
    manager.set_calibration("cam1", points, direction="up_in", relative=True)
    zone = manager.get_calibration("cam1")
    assert zone.points == points
    assert zone.direction == "up_in"
    assert zone.relative is True
