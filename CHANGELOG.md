# Changelog

Все значимые изменения фиксируются в этом файле.  
Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/)  
Версионирование: [Semantic Versioning](https://semver.org/lang/ru/)

---

## [Unreleased]

### Планируется
- Распределение 6 камер по 3 NPU-ядрам RK3588 (отдельный `DetectorRKNN` с `core_mask` на камеру)
- Рабочий INT8-квант для RKNN (split-head экспорт по rknn_model_zoo)
- Интеграция с ИТС (MQTT / HTTP callback)

---

## [2.0.0] - 2026-06-08

### Повышение точности трекинга

#### Добавлено
- **Трекер BotSort + OSNet ReID** (`app/core/tracker_osnet.py`, boxmot) - стал
  дефолтным на PC (`VKR_TRACKER_BACKEND=osnet`); вдвое меньше фрагментации ID на
  плотных дверных сценах, чем у ByteTrack
- **Двухлинейный счётчик tripwire** (`app/core/counter.py`, `COUNTER_MODE=tripwire`)
  - направление по порядку пересечения двух линий
- **Многокамерный режим** (`app/api/camera_routes.py`, `stats_routes.py`,
  `app/core/camera_manager.py`) + персист конфига камер с авто-восстановлением
  после рестарта (`app/storage/camera_config.py`) + веб-UI калибровки
  (`app/static/calibrate.html`, `/ui/calibrate`)
- **Сменные бэкенды детектора** через env `VKR_DETECTOR_BACKEND`:
  `pytorch` / `onnx` (`detector_onnx.py`) / `rknn` (`detector_rknn.py`)
- **Экспорт детектора в ONNX** и **конвертация ONNX-RKNN**
  (`notebooks/05_export_onnx.ipynb`, п.10; FP16, проверено на x86-симуляторе)
- **Edge-образ** `Dockerfile.edge` (RK3588/arm64, NPU через rknnlite2, без torch)
  + `requirements-edge.txt`; руководство по деплою - `EDGE_DEPLOY.md`

#### Изменено
- Docker PC-образ переведён с `nvidia/cuda:...` на **`python:3.11-slim`** (multi-stage):
  torch+cu128 несёт свои CUDA-либы, драйвер пробрасывается через
  `nvidia-container-toolkit` - меньше размер, нет дублирования CUDA

#### Метрики подсчёта событий (реальные дверные сцены, замер на кэшах OSNet)
Обе камеры измерены одним классом `TripwireCounter`:
- **door1** (99 клипов / 196 событий): F1 65.2% (P 63.6%, R 66.8%); офлайн-потолок
  подбора геометрии - 69.0%. Агрегат за рейс: смещение +5.1%, r 0.68, +/-1 - 71% клипов.
- **door2** (197 клипов / 855 событий): F1 61.8% (P 58.9%, R 65.0%); офлайн-потолок - 65.6%.
  Агрегат: смещение +10.4%, r 0.56.
- Альт. ветка door1 "детектор голов + смещение" - F1 72.0% (не дефолт).

#### Модели
- `osnet_x0_25_msmt17.pt` - ReID-веса для BotSort+OSNet
- `yolo11n_mot20_v2.onnx`, `yolo11n_mot20_v2.rknn` (FP16, ~7.2 МБ) - для edge

---

## [1.0.0] - 2025-04-30

### MVP - минимально рабочий прототип (desktop GPU)

#### Добавлено
- **Детектор** (`app/core/detector.py`) - YOLOv11n, файнтюнинг на MOT20, инференс на CUDA
- **Трекер** (`app/core/tracker.py`) - ByteTrack через `supervision`, подбор параметров на MOT20
- **Счётчик** (`app/core/counter.py`) - виртуальная линия, гистерезис 2-10 сек, cross-product для определения стороны
- **Пайплайн** (`app/core/pipeline.py`) - оркестратор: кадр - детекция - трекинг - счёт
- **REST API** (`app/api/`) - FastAPI эндпоинты: `/status`, `/stats`, `/calibration`, `/process`, `/upload`, `/upload-and-process`, `/videos`
- **SQLite хранилище** (`app/storage/db.py`) - сохранение событий entry/exit с timestamp
- **Docker** - GPU-образ (в 0.2.0 переведён на `python:3.11-slim`, см. выше)
- **Ноутбуки**:
  - `01_dataset_prep.ipynb` - EDA, фильтрация по Laplacian variance
  - `02_train_detection.ipynb` - обучение YOLOv11n на COCO + MOT20
  - `03_tracker_tuning_eval.ipynb` - grid search параметров ByteTrack, расчёт IDF1 на MOT20
  - `04_metrics_acceptance.ipynb` - сводная таблица приёмочных метрик
- **CLI-демо** для запуска на тестовых видео с выводом таблицы метрик
  (dev-утилита, после реорганизации - `workspace/tools/demo.py`)

#### Метрики v0.1.0 (RTX 5080) - уровень детектора/трекера на MOT20

> Это качество детекции боксов и стабильности ID, **не** подсчёта событий.
> Метрики подсчёта появились в 0.2.0 (см. выше) и измеряются на door1/door2.

| Метрика | Цель | Факт |
|---|---|---|
| F1 (bbox) | >= 88% | 88.3% (соответствует) |
| Precision (bbox) | >= 90% | 89.7% (частично) |
| Recall (bbox) | >= 90% | 86.9% (частично) |
| IDF1 | >= 75% | 74.93% (частично) |
| Latency | <= 100 мс | 3.8-12 мс (соответствует) |
| FPS | 15-30 | 82-261 (соответствует) |

#### Модели
- `yolo11n_mot20_v2.pt` - YOLOv11n файнтюнинг на MOT20, mAP@50 = 93.1%
- `bytetrack_best_params.json` - лучшие параметры ByteTrack из grid search
