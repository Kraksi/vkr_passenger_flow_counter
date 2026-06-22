"""конфиг камер на диске (camera_config.json) - чтобы после рестарта
восстановить подключения и подсчёт без повторной калибровки"""
from __future__ import annotations
import json
import threading
from pathlib import Path
from app.config import CAMERA_CONFIG_PATH

CONFIG_PATH = CAMERA_CONFIG_PATH


class CameraConfigStore:
    """потокобезопасное хранилище конфига камер"""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if not self._path.exists():
            return {"cameras": {}}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"cameras": {}}

    def _write(self, data: dict) -> None:
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_camera(
        self,
        cam_id: str,
        source: str | int,
        line: dict | None = None,
        autostart: bool = False,
    ) -> None:
        """сохранить/обновить камеру"""
        with self._lock:
            data = self._read()
            existing = data["cameras"].get(cam_id, {})
            data["cameras"][cam_id] = {
                "source": str(source),
                "autostart": autostart if autostart else existing.get("autostart", False),
                "line": line if line is not None else existing.get("line", {
                    "x1": 0.1, "y1": 0.5, "x2": 0.9, "y2": 0.5, "relative": True
                }),
            }
            self._write(data)

    def update_line(self, cam_id: str, line: dict) -> None:
        """обновить только линию калибровки"""
        with self._lock:
            data = self._read()
            if cam_id in data["cameras"]:
                data["cameras"][cam_id]["line"] = line
                self._write(data)

    def set_autostart(self, cam_id: str, autostart: bool) -> None:
        """автозапуск подсчёта при старте сервера вкл/выкл"""
        with self._lock:
            data = self._read()
            if cam_id in data["cameras"]:
                data["cameras"][cam_id]["autostart"] = autostart
                self._write(data)

    def delete_camera(self, cam_id: str) -> None:
        """удалить камеру из конфига"""
        with self._lock:
            data = self._read()
            data["cameras"].pop(cam_id, None)
            self._write(data)

    def load_all(self) -> dict[str, dict]:
        """все камеры: {cam_id: {source, autostart, line}}"""
        with self._lock:
            return dict(self._read().get("cameras", {}))


config_store = CameraConfigStore()
