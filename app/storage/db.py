from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from app.config import DB_PATH


class EventStore:
    """события пересечения линии в SQLite"""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """коннект + создать таблицы если нет"""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._migrate()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
                cam_id TEXT NOT NULL DEFAULT 'default'
            )
        """)
        self._conn.commit()

    def _migrate(self) -> None:
        """добавить cam_id в старую таблицу (если нет)"""
        try:
            self._conn.execute(
                "ALTER TABLE events ADD COLUMN cam_id TEXT NOT NULL DEFAULT 'default'"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_cam_ts
            ON events (cam_id, timestamp)
        """)
        self._conn.commit()


    def save_event(self, track_id: int, event_type: str, cam_id: str = "default") -> None:
        """сохранить одно событие entry/exit"""
        if self._conn is None:
            raise RuntimeError("БД не подключена. Вызовите connect().")
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO events (timestamp, track_id, event_type, cam_id) VALUES (?, ?, ?, ?)",
            (ts, track_id, event_type, cam_id),
        )
        self._conn.commit()

    def save_events(self, events: list[dict], cam_id: str = "default") -> None:
        """сохранить пачку событий за кадр"""
        if self._conn is None or not events:
            return
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "INSERT INTO events (timestamp, track_id, event_type, cam_id) VALUES (?, ?, ?, ?)",
            [(ts, e["track_id"], e["event"], cam_id) for e in events],
        )
        self._conn.commit()


    def get_stats(self, cam_id: str | None = None) -> dict:
        """суммарная статистика за всё время. без cam_id - по всем камерам"""
        if self._conn is None:
            return {"entries": 0, "exits": 0, "total": 0}
        if cam_id:
            cur = self._conn.execute(
                "SELECT event_type, COUNT(*) FROM events WHERE cam_id=? GROUP BY event_type",
                (cam_id,),
            )
        else:
            cur = self._conn.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
            )
        counts = dict(cur.fetchall())
        entries = counts.get("entry", 0)
        exits = counts.get("exit", 0)
        return {"entries": entries, "exits": exits, "total": entries + exits}


    def get_daily_stats(self, date_str: str, cam_id: str | None = None) -> dict:
        """статистика за день (UTC), date_str вида '2026-05-20'"""
        if self._conn is None:
            return {"date": date_str, "cam_id": cam_id, "entries": 0, "exits": 0, "total": 0}

        if cam_id:
            cur = self._conn.execute(
                """
                SELECT event_type, COUNT(*)
                FROM events
                WHERE date(timestamp) = ? AND cam_id = ?
                GROUP BY event_type
                """,
                (date_str, cam_id),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT event_type, COUNT(*)
                FROM events
                WHERE date(timestamp) = ?
                GROUP BY event_type
                """,
                (date_str,),
            )
        counts = dict(cur.fetchall())
        entries = counts.get("entry", 0)
        exits = counts.get("exit", 0)
        return {
            "date": date_str,
            "cam_id": cam_id,
            "entries": entries,
            "exits": exits,
            "total": entries + exits,
        }

    def get_hourly_breakdown(self, date_str: str, cam_id: str | None = None) -> list[dict]:
        """почасовая разбивка за день (UTC) - 24 записи [{hour, entries, exits}]"""
        if self._conn is None:
            return [{"hour": h, "entries": 0, "exits": 0} for h in range(24)]

        if cam_id:
            cur = self._conn.execute(
                """
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                       event_type,
                       COUNT(*) AS cnt
                FROM events
                WHERE date(timestamp) = ? AND cam_id = ?
                GROUP BY hour, event_type
                """,
                (date_str, cam_id),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                       event_type,
                       COUNT(*) AS cnt
                FROM events
                WHERE date(timestamp) = ?
                GROUP BY hour, event_type
                """,
                (date_str,),
            )

        by_hour: dict[int, dict] = {}
        for hour, event_type, cnt in cur.fetchall():
            if hour not in by_hour:
                by_hour[hour] = {"entries": 0, "exits": 0}
            if event_type == "entry":
                by_hour[hour]["entries"] = cnt
            else:
                by_hour[hour]["exits"] = cnt

        return [
            {
                "hour": h,
                "entries": by_hour.get(h, {}).get("entries", 0),
                "exits": by_hour.get(h, {}).get("exits", 0),
            }
            for h in range(24)
        ]

    def get_date_range_stats(
        self,
        start_date: str,
        end_date: str,
        cam_id: str | None = None,
    ) -> list[dict]:
        """статистика по дням за [start_date, end_date] включительно"""
        if self._conn is None:
            return []

        if cam_id:
            cur = self._conn.execute(
                """
                SELECT date(timestamp) AS day,
                       event_type,
                       COUNT(*) AS cnt
                FROM events
                WHERE day BETWEEN ? AND ? AND cam_id = ?
                GROUP BY day, event_type
                ORDER BY day
                """,
                (start_date, end_date, cam_id),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT date(timestamp) AS day,
                       event_type,
                       COUNT(*) AS cnt
                FROM events
                WHERE day BETWEEN ? AND ?
                GROUP BY day, event_type
                ORDER BY day
                """,
                (start_date, end_date),
            )

        by_date: dict[str, dict] = {}
        for day, event_type, cnt in cur.fetchall():
            if day not in by_date:
                by_date[day] = {"entries": 0, "exits": 0}
            if event_type == "entry":
                by_date[day]["entries"] = cnt
            else:
                by_date[day]["exits"] = cnt

        return [
            {
                "date": day,
                "cam_id": cam_id,
                "entries": v["entries"],
                "exits": v["exits"],
                "total": v["entries"] + v["exits"],
            }
            for day, v in sorted(by_date.items())
        ]

    def get_active_dates(self, cam_id: str | None = None) -> list[str]:
        """даты (UTC) где есть хоть одно событие"""
        if self._conn is None:
            return []
        if cam_id:
            cur = self._conn.execute(
                "SELECT DISTINCT date(timestamp) FROM events WHERE cam_id=? ORDER BY 1",
                (cam_id,),
            )
        else:
            cur = self._conn.execute(
                "SELECT DISTINCT date(timestamp) FROM events ORDER BY 1"
            )
        return [row[0] for row in cur.fetchall()]

    def get_active_cameras(self) -> list[str]:
        """cam_id где есть хоть одно событие"""
        if self._conn is None:
            return []
        cur = self._conn.execute(
            "SELECT DISTINCT cam_id FROM events ORDER BY cam_id"
        )
        return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


event_store = EventStore()
