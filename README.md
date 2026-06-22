# Счётчик пассажиропотока

> Магистерская диссертация, НИ ТГУ - Краснов Данила Анатольевич

Система автоматического подсчёта пассажиропотока на базе компьютерного зрения.  
Детектирует и отслеживает людей, считает события **входа/выхода** через виртуальную линию, предоставляет REST API.

**Стек:** YOLOv11n | BotSort+OSNet ReID (дефолт) / ByteTrack | FastAPI | OpenCV | PyTorch (CUDA) | SQLite

---

## Демонстрация

### Swagger UI (REST API)

Интерактивная документация доступна после запуска сервера на
[http://localhost:8000/docs](http://localhost:8000/docs).

### Пример вывода обработки видео (RTX 5080)

```
Клип                    GT  Pred  MAPE   FPS   Latency
store_entrance           5     5   0.0%  261   3.8 мс
mall_segment_1          30    24  20.0%  184   5.4 мс
mall_segment_2          28    22  21.4%  184   5.4 мс
mall_segment_3          33    25  24.2%  184   5.4 мс
mall_segment_4          29    21  27.6%  184   5.4 мс
mot20_01_clip            -     -     -    82  12.0 мс
```

---

## Требования

| Зависимость | Версия |
|---|---|
| Python | >= 3.11 |
| CUDA | 12.8+ (NVIDIA GPU) |
| torch | 2.11.0+cu128 |
| ultralytics | 8.4.37 |
| supervision | 0.27.0.post2 |
| fastapi | 0.135.3 |
| opencv-python | 4.13.0.92 |

Полный список - [`requirements-docker.txt`](requirements-docker.txt) (runtime) и [`requirements.txt`](requirements.txt) (dev + Jupyter).

---

## Установка

### Вариант 1 - локально (uv)

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd vkr_passenger_flow_counter

# 2. Создать виртуальное окружение (Python 3.11+)
uv venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 3. Установить зависимости
# Runtime (без Jupyter):
uv pip install -r requirements-docker.txt

# Полный набор (включая Jupyter для ноутбуков):
uv pip install -r requirements.txt

# 4. Положить веса модели в models/
# Скачать yolo11n_mot20_v2.pt
```

### Вариант 2 - Docker (PC, GPU)

Образ на базе `python:3.11-slim` (multi-stage); torch+cu128 несёт свои CUDA-либы,
GPU пробрасывается через `nvidia-container-toolkit`.

```bash
# Через docker compose (рекомендуется - GPU)
docker compose up --build
#   Swagger UI: http://localhost:8000/docs

# Либо вручную
docker build -t passenger-counter .
docker run --gpus all -p 8000:8000 passenger-counter
```

В образ копируются веса обоих продакшен-стеков: детектор тела `yolo11n_mot20_v2.pt`
и детектор голов `yolo11_head.pt` + соответствующие ReID-веса OSNet
(`osnet_x0_25_msmt17.pt`, `osnet_x0_25_mot20head.pt`); БД и конфиг камер пишутся
в том `counter-db` (`/app/state`).

### Вариант 3 - Edge (RK3588, NPU)

Лёгкий образ для одноплатника на Rockchip RK3588: детектор на NPU (`rknnlite2`),
трекинг ByteTrack без torch. Сборка и деплой - см. **[EDGE_DEPLOY.md](EDGE_DEPLOY.md)**.

```bash
docker build -f Dockerfile.edge -t passenger-counter:edge .
```

Режим переключается переменными окружения без правки кода:
`VKR_DETECTOR_BACKEND=rknn`, `VKR_TRACKER_BACKEND=bytetrack`.

---

## Запуск

### REST API (основной режим)

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Ноутбуки (подготовка двух продакшен-путей)

```bash
# Открыть VS Code с Jupyter Extension, выбрать kernel .venv
# Restart & Run All в порядке:
#   data_preparation  - detector_training
#   - tracker_edge_bytetrack  (edge-путь: ByteTrack)
#   - tracker_desktop_botsort (desktop-путь: BoT-SORT + OSNet)
```

Экспорт детектора для инференса:
`python scripts/export.py --format onnx` (desktop) / `--format rknn` (edge RK3588).

---

## API - примеры запросов

### Загрузить и обработать видео

```bash
# Загрузить файл и получить статистику
curl -X POST http://localhost:8000/upload-and-process \
  -F "file=@data/stock_videos/uhd_30fps.mp4"
```

```json
{
  "frames_processed": 450,
  "avg_latency_ms": 3.8,
  "fps": 261.5,
  "stats": {
    "entries": 5,
    "exits": 0,
    "total": 5,
    "current_inside": 5
  }
}
```

### Получить статистику

```bash
curl http://localhost:8000/stats
```

```json
{
  "entries": 12,
  "exits": 8,
  "total": 20,
  "current_inside": 4
}
```

### Получить статус пайплайна

```bash
curl http://localhost:8000/status
```

---

## Метрики производительности (RTX 5080)

| Видео | Разрешение | FPS | Latency |
|---|---|---|---|
| MOT20-01 | 1920x1080 | 82 | 12 мс |
| Mall Dataset | 640x480 | 184 | 5.4 мс |
| Store entrance | 2560x1440 | 261 | 3.8 мс |
| Bus boarding | 2160x3840 | 172 | 5.8 мс |

Целевые значения MVP: FPS >= 15, Latency <= 100 мс - **выполнено с запасом**.

---

## Метрики точности

Метрики разнесены по двум уровням - это разные задачи и разные наборы данных.

### Уровень 1 - детектор и трекер (бенчмарк MOT20)

| Метрика | Цель | Факт | Набор |
|---|---|---|---|
| mAP@50 (детектор) | - | 93.1% | MOT20 val, `yolo11n_mot20_v2.pt` |
| Precision (bbox) | - | 89.7% | MOT20 val |
| Recall (bbox) | - | 86.9% | MOT20 val |
| F1 (bbox) | - | 88.3% | MOT20 val |
| IDF1 (ByteTrack) | >= 75% | 74.93% | MOT20, grid search |
| IDF1 (BoT-SORT+OSNet) | >= 75% | 73.41% | MOT20 |

> Это качество **детекции и стабильности ID**, а не подсчёта событий. Целевой
> IDF1 >= 75% фактически достигнут (ByteTrack 74.93%).

### Уровень 2 - подсчёт событий вход/выход (реальные дверные сцены)

Главная прикладная метрика. Замерено на размеченных клипах двух камер автобуса
(`door1` - 99 клипов / 196 событий; `door2` - 197 клипов / 855 событий).
Дефолтный стек: детектор `yolo11n_mot20_v2` - **BotSort + OSNet ReID** - двухлинейный
счётчик **tripwire**.

Камеры измерены **одним и тем же продакшен-классом** `TripwireCounter`:

| Камера | F1 (продакшен) | F1 (офлайн-потолок) | Precision | Recall |
|---|---|---|---|---|
| **door1** (передняя дверь, крупный план) | **65.2%** | 69.0% | 63.6% | 66.8% |
| **door2** (средняя дверь, виден салон) | **61.8%** | 65.6% | 58.9% | 65.0% |

- *Продакшен* - класс `TripwireCounter` из `app/` с базовыми порогами и
  финализацией по `lost_frames` (как работает развёрнутая система).
- *Офлайн-потолок* - лучший конфиг при грид-подборе геометрии на той же камере.
- Разрыв офлайн-продакшен ~ -4 п.п. на **обеих** камерах.
- Альтернативная ветка door1 "детектор голов + классификатор по смещению" даёт
  **F1 72.0%**, но требует отдельной модели голов и не вынесена в
  дефолт.

---

## Архитектура системы

```
Видео/кадр
    |
    v
[Detector]          YOLOv11n - бэкенд по env: pytorch (CUDA) / onnx / rknn (NPU)
    | bbox + conf
    v
[Tracker]           BotSort+OSNet ReID (дефолт, PC) | ByteTrack (лёгкий, edge)
    | track_id + bbox
    v
[Counter]           tripwire (две линии, дефолт) | lifecycle | гистерезис 2-10 с
    | entry/exit events
    v
[EventStore]        SQLite - персистентность событий
    |
    v
[FastAPI]           REST API (/status, /stats, /calibration, /process, камеры)
```

Бэкенды детектора и трекера переключаются переменными окружения
(`VKR_DETECTOR_BACKEND`, `VKR_TRACKER_BACKEND`) без правки кода - это позволяет
одним кодом покрыть и десктоп с GPU, и edge-плату RK3588 (см. [EDGE_DEPLOY.md](EDGE_DEPLOY.md)).

---

## Модель

Используется **YOLOv11n** - файнтюнинг на MOT20 (детекция людей в плотных сценах).

- Весовой файл детектора: `models/yolo11n_mot20_v2.pt`
- Обучение (2 этапа): [`notebooks/detector_training.ipynb`](notebooks/detector_training.ipynb)
- mAP@50: **93.1%**, F1 (bbox): **88.3%** - это качество детекции боксов на MOT20,
  не путать с F1 подсчёта событий (см. раздел "Метрики точности")

Дефолтный трекер (BotSort+OSNet ReID) дополнительно использует ReID-веса
`models/osnet_x0_25_msmt17.pt`. Для edge есть экспорт детектора:
`models/yolo11n_mot20_v2.onnx` и `models/yolo11n_mot20_v2.rknn` (NPU RK3588).

Базовые предобученные веса (COCO) загружаются автоматически через Ultralytics при первом запуске.

---

## Структура проекта

```
vkr_passenger_flow_counter/
+-- app/
|   +-- main.py               # Точка входа FastAPI
|   +-- config.py             # Все настройки (пороги, пути, бэкенды, линии)
|   +-- api/
|   |   +-- routes.py         # /status, /stats, /calibration, /process, /upload...
|   |   +-- camera_routes.py  # Многокамерный режим (connect/start/stop)
|   |   +-- stats_routes.py   # Агрегированная статистика по камерам
|   |   +-- schemas.py        # Pydantic-схемы запросов/ответов
|   +-- core/
|   |   +-- detector.py       # YOLOv11n inference (PyTorch/CUDA)
|   |   +-- detector_onnx.py  # Бэкенд ONNX Runtime
|   |   +-- detector_rknn.py  # Бэкенд RKNN NPU (edge RK3588)
|   |   +-- tracker.py        # ByteTrack (лёгкий, без ReID)
|   |   +-- tracker_osnet.py  # BotSort + OSNet ReID (дефолт на PC)
|   |   +-- counter.py        # Подсчёт: tripwire (2 линии) / lifecycle / гистерезис
|   |   +-- pipeline.py       # Оркестратор кадр-детекция-трекинг-счёт
|   +-- storage/
|       +-- db.py             # SQLite EventStore
|       +-- camera_config.py  # Персист конфига камер (авто-восстановление)
+-- notebooks/                # Подготовка двух продакшен-путей (ПО)
|   +-- data_preparation.ipynb        # Данные: COCO-person + MOT20 (общая база)
|   +-- detector_training.ipynb       # YOLOv11n: COCO -> MOT20 DA -> yolo11n_mot20_v2
|   +-- tracker_edge_bytetrack.ipynb  # Edge-путь: ByteTrack (grid на MOT20)
|   +-- tracker_desktop_botsort.ipynb # Desktop-путь: BoT-SORT + OSNet ReID
+-- scripts/
|   +-- export.py             # Экспорт детектора: PT->ONNX (desktop) / ONNX->RKNN (edge)
+-- tests/                    # 20 ключевых тестов (test_key_diploma.py) + conftest
+-- models/                   # Веса (не в репозитории, гитигнор) + конфиги трекера
+-- Dockerfile                # PC-образ (GPU)
+-- Dockerfile.edge           # Edge-образ (RK3588, NPU)
+-- docker-compose.yml
+-- requirements-docker.txt   # Runtime PC (GPU)
+-- requirements-edge.txt     # Runtime edge (RK3588, без torch)
+-- requirements.txt          # Dev + Jupyter
+-- EDGE_DEPLOY.md            # Деплой на RK3588
+-- CHANGELOG.md
```

---

## Презентация

- Презентация: *будет добавлена ссылка после защиты*

---

## Лицензия

**GNU AGPL-3.0** - см. [LICENSE](LICENSE).

```
Copyright (c) 2025 Краснов Данила Анатольевич
```

Проект использует **Ultralytics YOLO** и **boxmot**, распространяемые под **AGPL-3.0**
(сильный копилефт + сетевая оговорка §13). Поэтому комбинированное произведение и
сетевой сервис на его основе наследуют AGPL-3.0: при предоставлении доступа по сети
исходный код должен оставаться открытым под той же лицензией. Проект академический
(ВКР, НИ ТГУ), без коммерческого использования.

Лицензии всех зависимостей - в [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

> **Ограничение на данные (некоммерческое).** Обученные веса наследуют лицензии
> датасетов: `yolo11n_mot20_v2` обучен на **MOT20** (академическая, некоммерческая),
> детектор голов - на **CrowdHuman/SCUT-HEAD** (research-only), ReID-веса OSNet - на
> **MSMT17** (research). Использование - только в научных/учебных целях.
