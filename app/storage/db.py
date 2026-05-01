# SQLite: запись событий вход/выход со статистикой
from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from app.config import DB_PATH


class EventStore:
    """Хранит события пересечения линии в SQLite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Открыть соединение и создать таблицы если их нет."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit'))
            )
        """)
        self._conn.commit()

    def save_event(self, track_id: int, event_type: str) -> None:
        """Сохранить событие 'entry' или 'exit'."""
        if self._conn is None:
            raise RuntimeError("БД не подключена. Вызовите connect().")
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO events (timestamp, track_id, event_type) VALUES (?, ?, ?)",
            (ts, track_id, event_type),
        )
        self._conn.commit()

    def save_events(self, events: list[dict]) -> None:
        """Сохранить пакет событий за один кадр."""
        if self._conn is None or not events:
            return
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "INSERT INTO events (timestamp, track_id, event_type) VALUES (?, ?, ?)",
            [(ts, e["track_id"], e["event"]) for e in events],
        )
        self._conn.commit()

    def get_stats(self) -> dict:
        """Вернуть агрегированную статистику из БД."""
        if self._conn is None:
            return {"entries": 0, "exits": 0, "total": 0}
        cur = self._conn.execute(
            "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
        )
        counts = dict(cur.fetchall())
        entries = counts.get("entry", 0)
        exits = counts.get("exit", 0)
        return {"entries": entries, "exits": exits, "total": entries + exits}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
