import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.core.pipeline import Pipeline
from app.api.routes import router
from app.api.camera_routes import router as camera_router, camera_manager
from app.api.stats_routes import router as stats_router
from app.storage.camera_config import config_store
from app.storage.db import event_store
from app.core.camera_manager import CameraLine
from app.config import DOOR_ZONE_POINTS

log = logging.getLogger("vkr.startup")

pipeline = Pipeline()


async def _restore_cameras() -> None:
    """поднять камеры из camera_config.json после рестарта:
    connect - применить линию - если autostart запустить подсчёт"""
    saved = config_store.load_all()
    if not saved:
        return

    log.info("Восстановление %d камер из конфига...", len(saved))
    loop = asyncio.get_event_loop()

    for cam_id, cfg in saved.items():
        source_str = cfg.get("source", "")
        source = int(source_str) if source_str.isdigit() else source_str
        line_d = cfg.get("line", {}) or cfg.get("zone", {})
        autostart = cfg.get("autostart", False)

        try:
            await loop.run_in_executor(None, camera_manager.connect, cam_id, source)
            log.info("  [%s] подключена (%s)", cam_id, source_str)
        except RuntimeError as exc:
            log.warning("  [%s] не удалось подключить: %s", cam_id, exc)
            continue

        if line_d:
            pts = line_d.get("points")
            if not pts and any(k in line_d for k in ("x1", "y1", "x2", "y2")):
                x1 = line_d.get("x1", 0.0); y1 = line_d.get("y1", 0.0)
                x2 = line_d.get("x2", 0.45); y2 = line_d.get("y2", 1.0)
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            if not pts:
                pts = [list(p) for p in DOOR_ZONE_POINTS]
            camera_manager.set_calibration(
                cam_id, pts,
                line_d.get("direction", "down_in"),
                line_d.get("relative", True),
            )

        if autostart:
            try:
                await loop.run_in_executor(None, camera_manager.start_counting, cam_id)
                log.info("  [%s] подсчёт запущен (autostart)", cam_id)
            except RuntimeError as exc:
                log.warning("  [%s] не удалось запустить подсчёт: %s", cam_id, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """старт: инициализация, стоп: очистка"""
    event_store.connect()
    pipeline.initialize()
    await _restore_cameras()
    yield
    pipeline.shutdown()
    camera_manager.disconnect_all()
    event_store.close()


app = FastAPI(
    title="Счётчик пассажиропотока",
    description="REST API для подсчёта входов/выходов через виртуальную линию",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(camera_router)
app.include_router(stats_router)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@app.get("/ui/calibrate", include_in_schema=False)
async def calibrate_ui():
    """веб-интерфейс калибровки линий"""
    html_path = Path(__file__).parent / "static" / "calibrate.html"
    return FileResponse(html_path)
