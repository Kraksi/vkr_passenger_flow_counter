# Changelog

Все значимые изменения фиксируются в этом файле.  
Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/)  
Версионирование: [Semantic Versioning](https://semver.org/lang/ru/)

---

## [Unreleased]

### Планируется
- Поддержка RTSP-потока в реальном времени
- Дообучение на собственном доменном датасете (автобус/трамвай)
- Сборка ONNX-экспорта для edge-устройств

---

## [0.1.0] — 2025-04-30

### MVP — минимально рабочий прототип (desktop GPU)

#### Добавлено
- **Детектор** (`app/core/detector.py`) — YOLOv11n, файнтюнинг на MOT20, инференс на CUDA
- **Трекер** (`app/core/tracker.py`) — ByteTrack через `supervision`, подбор параметров на MOT20
- **Счётчик** (`app/core/counter.py`) — виртуальная линия, гистерезис 2–10 сек, cross-product для определения стороны
- **Пайплайн** (`app/core/pipeline.py`) — оркестратор: кадр → детекция → трекинг → счёт
- **REST API** (`app/api/`) — FastAPI эндпоинты: `/status`, `/stats`, `/calibration`, `/process`, `/upload`, `/upload-and-process`, `/videos`
- **SQLite хранилище** (`app/storage/db.py`) — сохранение событий entry/exit с timestamp
- **Docker** — образ на базе `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
- **Ноутбуки**:
  - `01_dataset_prep.ipynb` — EDA, фильтрация по Laplacian variance
  - `02_train_detection.ipynb` — обучение YOLOv11n на COCO + MOT20
  - `03_tracker_tuning_eval.ipynb` — grid search параметров ByteTrack, расчёт IDF1 на MOT20
  - `04_metrics_acceptance.ipynb` — сводная таблица приёмочных метрик
- **CLI-демо** (`demo.py`) — запуск на тестовых видео с выводом таблицы метрик

#### Метрики v0.1.0 (RTX 5080)

| Метрика | Цель | Факт |
|---|---|---|
| F1-Score | ≥ 88% | 88.1% ✅ |
| Precision | ≥ 90% | 89.7% ⚠️ |
| Recall | ≥ 90% | 86.9% ⚠️ |
| IDF1 | ≥ 75% | 74.93% ⚠️ |
| Latency | ≤ 100 мс | 3.8–12 мс ✅ |
| FPS | 15–30 | 82–261 ✅ |

#### Модели
- `yolo11n_mot20_v2.pt` — YOLOv11n файнтюнинг на MOT20, mAP@50 = 93.1%
- `bytetrack_best_params.json` — лучшие параметры ByteTrack из grid search

---

## [0.2.0] — планируется

### Edge-развёртывание (RK3588)

#### Планируется
- Экспорт модели в ONNX / RKNN для NPU RK3588
- Поддержка 6 камер × 15 fps при энергопотреблении ≤ 15 Вт
- Оптимизация под latency ≤ 200 мс на CPU/NPU
- Интеграция с ИТС (MQTT / HTTP callback)
- Веб-интерфейс для мониторинга нескольких камер
