from pathlib import Path

# Корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути к моделям и данным
MODEL_PATH = BASE_DIR / "models" / "yolo11n_mot20_v2.pt"
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "storage.db"

# Параметры детектора
DETECTOR_CONF_THRESHOLD = 0.20  # минимальная уверенность детекции
DETECTOR_IOU_THRESHOLD = 0.45   # порог NMS
DETECTOR_DEVICE = "cuda"        # 'cuda' или 'cpu'
DETECTOR_CLASS_PERSON = 0       # индекс класса "person" в COCO

# Параметры трекера ByteTrack
TRACKER_TRACK_THRESH = 0.25     # порог активации нового трека
TRACKER_MATCH_THRESH = 0.85     # порог сопоставления детекций с треками
TRACKER_TRACK_BUFFER = 75       # кадров до удаления потерянного трека
TRACKER_FRAME_RATE = 25         # FPS видео (используется ByteTrack)

# Параметры счётчика по линии
COUNTER_HYSTERESIS_MIN_SEC = 2.0   # минимальный интервал между событиями одного трека
COUNTER_HYSTERESIS_MAX_SEC = 10.0  # максимальный интервал (сброс состояния)

# Виртуальная линия — относительные координаты (0.0–1.0 от размера кадра)
# Переопределяется через API /calibration (принимает как относительные, так и абсолютные)
# По умолчанию: горизонтальная линия на 75% высоты кадра
LINE_START = (0.0, 0.75)
LINE_END = (1.0, 0.75)
LINE_COORDS_RELATIVE = True  # True = относительные (0–1), False = абсолютные пиксели

# Папка для загруженных видео
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

# FastAPI
API_HOST = "0.0.0.0"
API_PORT = 8000
