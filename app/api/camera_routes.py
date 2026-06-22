"""эндпоинты камер: управление, калибровка, подсчёт в реалтайме"""
from __future__ import annotations
import asyncio
import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.camera_manager import CameraManager
from app.storage.camera_config import config_store
from app.config import TRIPWIRE_Y1, TRIPWIRE_Y2

router = APIRouter(prefix="/cameras", tags=["cameras"])

camera_manager = CameraManager()



class ConnectRequest(BaseModel):
    source: str
    autostart: bool = False


class CameraCalibrationRequest(BaseModel):
    points: list[list[float]] | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    direction: str = "down_in"
    relative: bool = True


class AutostartRequest(BaseModel):
    autostart: bool



@router.get("")
async def list_cameras():
    """все подключённые камеры (статус, размер, fps, counting, линия)"""
    saved = config_store.load_all()
    cameras = camera_manager.list_cameras()
    for cam in cameras:
        cam["autostart"] = saved.get(cam["cam_id"], {}).get("autostart", False)
    return {
        "cameras": cameras,
        "max_cameras": camera_manager.max_cameras,
    }


@router.get("/summary", tags=["stats"])
async def summary():
    """сводка по всем камерам одним запросом (опрос без UI). только counting камеры"""
    result = {}
    for cam in camera_manager.list_cameras():
        cam_id = cam["cam_id"]
        if cam["counting"]:
            stats = camera_manager.get_stats(cam_id) or {}
            result[cam_id] = {
                "entries":  stats.get("entries", 0),
                "exits":    stats.get("exits", 0),
                "current_inside": stats.get("current_inside", 0),
                "fps":      stats.get("fps", 0.0),
                "latency_ms": stats.get("latency_ms", 0.0),
            }
    return {"counting_cameras": len(result), "stats": result}



@router.post("/{cam_id}/connect")
async def connect_camera(cam_id: str, body: ConnectRequest):
    """подключить камеру + сохранить в конфиг.
    source: "0"/"1" USB, "rtsp://..." IP/телефон, путь к .mp4 для тестов.
    autostart=true - сразу считать и помнить при рестарте"""
    source: str | int = int(body.source.strip()) if body.source.strip().isdigit() else body.source
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, camera_manager.connect, cam_id, source)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    info = camera_manager.camera_info(cam_id)
    line_dict = None
    if info:
        line = camera_manager.get_calibration(cam_id)
        line_dict = line.to_dict() if line else None
    config_store.save_camera(cam_id, source, line_dict, autostart=body.autostart)

    if body.autostart:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, camera_manager.start_counting, cam_id
            )
        except RuntimeError:
            pass

    return {"success": True, "camera": info}


@router.delete("/{cam_id}")
async def disconnect_camera(cam_id: str):
    """стоп подсчёта, отключить камеру, удалить из конфига"""
    camera_manager.disconnect(cam_id)
    config_store.delete_camera(cam_id)
    return {"success": True, "cam_id": cam_id}



@router.get("/{cam_id}/stream")
async def stream_camera(cam_id: str):
    """Нужен только для калибровки в UI, для мониторинга хватит /stats"""
    if camera_manager.camera_info(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"Камера {cam_id!r} не подключена")

    async def generate():
        while True:
            frame = camera_manager.get_frame(cam_id)
            if frame is None:
                await asyncio.sleep(0.04)
                continue
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                await asyncio.sleep(0.04)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes() + b"\r\n"
            )
            await asyncio.sleep(1 / 25)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{cam_id}/snapshot")
async def snapshot(cam_id: str):
    """одна картинка для быстрой проверки"""
    if camera_manager.camera_info(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"Камера {cam_id!r} не подключена")
    frame = camera_manager.get_frame(cam_id)
    if frame is None:
        raise HTTPException(status_code=503, detail="Кадр ещё не получен")
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Ошибка кодирования")
    return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")



@router.get("/{cam_id}/calibration")
async def get_calibration(cam_id: str):
    """текущая зона двери (прямоугольник + направление)"""
    zone = camera_manager.get_calibration(cam_id)
    if zone is None:
        raise HTTPException(status_code=404, detail=f"Камера {cam_id!r} не найдена")
    return {
        "cam_id": cam_id,
        "zone": zone.to_dict(),
        "tripwire": {"y1": TRIPWIRE_Y1, "y2": TRIPWIRE_Y2},
    }


@router.post("/{cam_id}/calibration")
async def set_calibration(cam_id: str, body: CameraCalibrationRequest):
    """обновить зону двери + сохранить в конфиг. применяется сразу, даже на ходу"""
    if camera_manager.camera_info(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"Камера {cam_id!r} не подключена")
    pts = body.points
    if pts is None:
        if None in (body.x1, body.y1, body.x2, body.y2):
            raise HTTPException(status_code=422, detail="Нужны points или x1,y1,x2,y2")
        pts = [[body.x1, body.y1], [body.x2, body.y1],
               [body.x2, body.y2], [body.x1, body.y2]]
    camera_manager.set_calibration(cam_id, pts, body.direction, body.relative)
    zone = camera_manager.get_calibration(cam_id)
    zone_dict = zone.to_dict()
    config_store.update_line(cam_id, zone_dict)
    return {"success": True, "cam_id": cam_id, "zone": zone_dict}



@router.post("/{cam_id}/start")
async def start_counting(cam_id: str):
    """запустить подсчёт + пометить autostart. первый запуск грузит модель, дальше стартует сам даже после рестарта"""
    if camera_manager.camera_info(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"Камера {cam_id!r} не подключена")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, camera_manager.start_counting, cam_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    config_store.set_autostart(cam_id, True)
    return {"success": True, "cam_id": cam_id, "counting": True}


@router.post("/{cam_id}/stop")
async def stop_counting(cam_id: str):
    """стоп подсчёта. камера остаётся, autostart снимается"""
    camera_manager.stop_counting(cam_id)
    config_store.set_autostart(cam_id, False)
    return {"success": True, "cam_id": cam_id, "counting": False}


@router.get("/{cam_id}/stats")
async def get_stats(cam_id: str):
    """текущая статистика. основной эндпоинт для опроса без UI и стрима"""
    stats = camera_manager.get_stats(cam_id)
    if stats is None:
        return {
            "cam_id": cam_id, "counting": False,
            "entries": 0, "exits": 0, "total": 0,
            "current_inside": 0, "fps": 0.0, "latency_ms": 0.0,
        }
    return {"cam_id": cam_id, **stats}


@router.post("/{cam_id}/reset")
async def reset_stats(cam_id: str):
    """обнулить счётчики без остановки трекера"""
    camera_manager.reset_stats(cam_id)
    return {"success": True, "cam_id": cam_id}


@router.post("/{cam_id}/autostart")
async def set_autostart(cam_id: str, body: AutostartRequest):
    """автозапуск подсчёта при старте сервера вкл/выкл"""
    if camera_manager.camera_info(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"Камера {cam_id!r} не найдена в конфиге")
    config_store.set_autostart(cam_id, body.autostart)
    return {"success": True, "cam_id": cam_id, "autostart": body.autostart}
