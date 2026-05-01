# Счётчик пассажиропотока

> Магистерская диссертация, НИ ТГУ — Краснов Данила Анатольевич

Система автоматического подсчёта пассажиропотока на базе компьютерного зрения.  
Детектирует и отслеживает людей, считает события **входа/выхода** через виртуальную линию, предоставляет REST API.

**Стек:** YOLOv11n · ByteTrack · FastAPI · OpenCV · PyTorch (CUDA) · SQLite

---

## Демонстрация

### Swagger UI (REST API)

![Swagger UI](assets/swagger_ui.png)

### Пример вывода CLI-демо

```
Клип                    GT  Pred  MAPE   FPS   Latency
store_entrance           5     5   0.0%  261   3.8 мс
mall_segment_1          30    24  20.0%  184   5.4 мс
mall_segment_2          28    22  21.4%  184   5.4 мс
mall_segment_3          33    25  24.2%  184   5.4 мс
mall_segment_4          29    21  27.6%  184   5.4 мс
mot20_01_clip            —     —     —    82  12.0 мс
```

> Подробные метрики приёмочных испытаний — в [`notebooks/04_metrics_acceptance.ipynb`](notebooks/04_metrics_acceptance.ipynb)

---

## Требования

| Зависимость | Версия |
|---|---|
| Python | ≥ 3.11 |
| CUDA | 12.8+ (NVIDIA GPU) |
| torch | 2.11.0+cu128 |
| ultralytics | 8.4.37 |
| supervision | 0.27.0.post2 |
| fastapi | 0.135.3 |
| opencv-python | 4.13.0.92 |

Полный список — [`requirements-docker.txt`](requirements-docker.txt) (runtime) и [`requirements.txt`](requirements.txt) (dev + Jupyter).

---

## Установка

### Вариант 1 — локально (uv)

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd vkr-passenger-flow-counter

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
# Скачать yolo11n_mot20_v2.pt (см. раздел «Модель» ниже)
```

### Вариант 2 — Docker (GPU)

```bash
# Сборка образа
docker build -t passenger-counter .

# Запуск с проброской GPU
docker run --gpus all -p 8000:8000 passenger-counter
```

---

## Запуск

### REST API (основной режим)

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc:      [http://localhost:8000/redoc](http://localhost:8000/redoc)

### CLI-демо (5 контрольных примеров)

```bash
# Запустить демо на тестовых видео (выводит таблицу метрик)
python demo.py

# Указать конкретное видео
python demo.py --video data/stock_videos/uhd_30fps.mp4 --gt-entries 4 --gt-exits 1

# Запустить на всех видео из папки
python demo.py --dir data/stock_videos/
```

### Ноутбуки (обучение и метрики)

```bash
# Открыть VS Code с Jupyter Extension, выбрать kernel .venv
# Запускать Restart & Run All для каждого ноутбука в порядке:
# 01 → 02 → 03 → 04
```

---

## API — примеры запросов

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

### Установить виртуальную линию подсчёта

```bash
# Вертикальная линия на 50% ширины кадра (относительные координаты)
curl -X POST http://localhost:8000/calibration \
  -H "Content-Type: application/json" \
  -d '{"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0, "relative": true}'
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
| MOT20-01 | 1920×1080 | 82 | 12 мс |
| Mall Dataset | 640×480 | 184 | 5.4 мс |
| Store entrance | 2560×1440 | 261 | 3.8 мс |
| Bus boarding | 2160×3840 | 172 | 5.8 мс |

Целевые значения MVP: FPS ≥ 15, Latency ≤ 100 мс — **выполнено с запасом**.

---

## Метрики точности

| Метрика | Цель | Факт | Статус |
|---|---|---|---|
| F1-Score | ≥ 88% | 88.1% | ✅ |
| Precision | ≥ 90% | 89.7% | ⚠️ −0.3% |
| Recall | ≥ 90% | 86.9% | ⚠️ −3.1% |
| IDF1 | ≥ 75% | 74.93% | ⚠️ −0.07% |
| MAPE | ≤ 5% | 0% (1 клип) | ⏳ |

Детальный расчёт всех метрик — [`notebooks/04_metrics_acceptance.ipynb`](notebooks/04_metrics_acceptance.ipynb).

---

## Архитектура системы

```
Видео/кадр
    │
    ▼
[Detector]          YOLOv11n (CUDA) — детекция людей
    │ bbox + conf
    ▼
[Tracker]           ByteTrack (supervision) — сопоставление треков
    │ track_id + bbox
    ▼
[LineCounter]       виртуальная линия + гистерезис 2–10 с
    │ entry/exit events
    ▼
[EventStore]        SQLite — персистентность событий
    │
    ▼
[FastAPI]           REST API (/status, /stats, /calibration, /process)
```

---

## Модель

Используется **YOLOv11n** — файнтюнинг на MOT20 (детекция людей в плотных сценах).

- Весовой файл: `models/yolo11n_mot20_v2.pt`
- Обучение: [`notebooks/02_train_detection.ipynb`](notebooks/02_train_detection.ipynb)
- mAP@50: **93.1%**, F1: **88.3%**

Базовые предобученные веса (COCO) загружаются автоматически через Ultralytics при первом запуске.

---

## Структура проекта

```
vkr/
├── app/
│   ├── main.py               # Точка входа FastAPI
│   ├── config.py             # Все настройки (пороги, пути, параметры линии)
│   ├── api/
│   │   ├── routes.py         # Эндпоинты: /status, /stats, /calibration, /process
│   │   └── schemas.py        # Pydantic-схемы запросов/ответов
│   ├── core/
│   │   ├── detector.py       # YOLOv11n inference
│   │   ├── tracker.py        # ByteTrack
│   │   ├── counter.py        # Подсчёт по линии с гистерезисом
│   │   └── pipeline.py       # Оркестратор кадр→детекция→трекинг→счёт
│   └── storage/
│       └── db.py             # SQLite EventStore
├── notebooks/
│   ├── 01_dataset_prep.ipynb       # Подготовка данных
│   ├── 02_train_detection.ipynb    # Обучение YOLOv11n
│   ├── 03_tracker_tuning_eval.ipynb# Настройка ByteTrack, IDF1 на MOT20
│   └── 04_metrics_acceptance.ipynb # Приёмочные метрики
├── models/                   # Веса (не в репозитории, см. Releases)
├── demo.py                   # CLI-демо на тестовых видео
├── Dockerfile
├── requirements-docker.txt   # Runtime-зависимости
├── requirements.txt          # Dev + Jupyter
└── CHANGELOG.md
```

---

## Отчёт и презентация

- Отчёт (ВКР): *будет добавлена ссылка после защиты*
- Презентация: *будет добавлена ссылка после защиты*

---

## Лицензия

MIT License — см. [LICENSE](LICENSE)

```
Copyright (c) 2025 Краснов Данила Анатольевич
```

Используемые компоненты: Ultralytics YOLO (AGPL-3.0), supervision (MIT), FastAPI (MIT), PyTorch (BSD-3).

> **Важно:** Ultralytics YOLO распространяется под лицензией AGPL-3.0. Для коммерческого использования требуется отдельная коммерческая лицензия от Ultralytics.
