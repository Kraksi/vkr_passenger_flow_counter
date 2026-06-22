# Деплой на edge-устройство RK3588 (NPU)

Руководство по переносу счётчика пассажиропотока с десктопа (NVIDIA GPU) на
одноплатный компьютер на базе **Rockchip RK3588**. Детекция выполняется на **NPU 6 TOPS** через
`rknnlite2`, трекинг - на CPU (ByteTrack, без torch).

---

## 1. Чем edge-сборка отличается от десктопной

| | Десктоп (PC) | Edge (RK3588) |
|---|---|---|
| Детектор | YOLOv11n `.pt`, CUDA (`DETECTOR_BACKEND=pytorch`) | YOLOv11n `.rknn`, NPU (`DETECTOR_BACKEND=rknn`) |
| Трекер | BotSort + **OSNet ReID** (нужен torch) | **ByteTrack** (numpy, без torch) |
| Зависимости | torch, ultralytics, boxmot (~5 ГБ) | rknnlite2, supervision, opencv (~0.5 ГБ) |
| Dockerfile | `Dockerfile` | `Dockerfile.edge` |
| requirements | `requirements-docker.txt` | `requirements-edge.txt` |

> **Важно про точность.** OSNet ReID на NPU не запускается (нет аппаратной
> поддержки модели внешности в текущей сборке), поэтому на edge используется
> ByteTrack. Это даёт более высокую фрагментацию ID на плотных сценах. По
> результатам ВКР вклад ReID - порядка +10-18 п.п. F1 на пиковых сценах; на edge
> агрегатный счёт остаётся рабочим, но per-event метрики ниже десктопных.
> Целевые edge-требования из ТЗ: latency <= 200 мс/кадр, 6 камер x 15 fps,
> энергопотребление <= 15 Вт.

Переключение режима задаётся переменными окружения (уже прописаны в
`Dockerfile.edge`):

```bash
VKR_DETECTOR_BACKEND=rknn
VKR_TRACKER_BACKEND=bytetrack
VKR_DETECTOR_DEVICE=cpu
```

---

## 2. Шаг 1 - Экспорт модели в ONNX (на x86)

ONNX-модель уже есть в репозитории: `models/yolo11n_mot20_v2.onnx`.
Если нужно пересобрать из `.pt`:

```bash
# через скрипт экспорта (PT -> ONNX)
uv run python scripts/export.py --format onnx
# либо напрямую через ultralytics
uv run python -c "from ultralytics import YOLO; \
    YOLO('models/yolo11n_mot20_v2.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
```

---

## 3. Шаг 2 - Конвертация ONNX - RKNN (на x86)

Выполняется на **x86-машине разработчика** в ИЗОЛИРОВАННОМ venv (rknn-toolkit2
пинит свои torch/onnx и сломал бы основной torch+cu128). Lite-версия только
запускает модель на плате.

```bash
# 1. изолированный venv + полный toolkit (только x86, Python 3.11)
uv venv .venv-rknn --python 3.11
VIRTUAL_ENV=.venv-rknn uv pip install rknn-toolkit2==2.3.2
VIRTUAL_ENV=.venv-rknn uv pip install "setuptools<81" "onnx==1.16.2"  # см. примечание

# 2. FP16-конвертация (рекомендуется - проверена, точность как у ONNX)
#    Конвертация ONNX -> RKNN в scripts/export.py (--format rknn)
.venv-rknn/bin/python scripts/export.py --format rknn \
    --onnx models/yolo11n_mot20_v2.onnx \
    --rknn models/yolo11n_mot20_v2.rknn
```

На выходе - `models/yolo11n_mot20_v2.rknn` (FP16, ~7.2 МБ).

> **Примечание по версиям (rknn-toolkit2 2.3.2):** требует `setuptools<81`
> (иначе нет `pkg_resources`) и `onnx==1.16.x` (новый onnx 1.18+ убрал
> `onnx.mapping`, который использует toolkit).

---

## 4. Шаг 3 - Получить rknnlite2 и системную либу (для платы)

`rknn-toolkit-lite2` и `librknnrt.so` берутся из официального репозитория
Rockchip под версию платы. Версии lite-toolkit и `librknnrt.so` **должны
совпадать** с версией, которой конвертировали модель.

```bash
# на плате (Debian/Ubuntu arm64). ВАЖНО: версия lite2 и librknnrt.so должны
# совпадать с версией toolkit, которой конвертировали модель (см. раздел 3 - 2.3.2):
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2

# 1) системная NPU-либа
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

# 2) python-обёртка lite2 (wheel под cp311/aarch64)
pip install rknn_toolkit_lite2/packages/rknn_toolkit_lite2-2.3.2-cp311-*aarch64.whl

# 3) проверка
python -c "from rknnlite.api import RKNNLite; print('rknnlite ok')"
```

Для контейнерной сборки положи wheel в `wheels/` рядом с `Dockerfile.edge`:

```bash
mkdir -p wheels
cp .../rknn_toolkit_lite2-*-aarch64.whl wheels/
```

---

## 5. Шаг 4 - Запуск

### Вариант A - нативно на плате (рекомендуется для отладки)

```bash
# скопировать проект и .rknn на плату
scp -r app/ models/yolo11n_mot20_v2.rknn requirements-edge.txt user@<board-ip>:~/vkr_passenger_flow_counter/

# на плате
cd ~/vkr_passenger_flow_counter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-edge.txt
# + установить rknnlite2 (см. шаг 3)

export VKR_DETECTOR_BACKEND=rknn
export VKR_TRACKER_BACKEND=bytetrack
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Вариант B - Docker на плате

```bash
# собрать на плате (нативно arm64) либо на x86 через buildx+qemu
docker build -f Dockerfile.edge -t passenger-counter:edge .

# запуск с доступом к NPU
docker run --privileged \
    --device /dev/dri:/dev/dri \
    -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \
    -p 8000:8000 \
    passenger-counter:edge
```

Проверка: `curl http://<board-ip>:8000/health` - `{"status":"ok"}`,
Swagger - `http://<board-ip>:8000/docs`.