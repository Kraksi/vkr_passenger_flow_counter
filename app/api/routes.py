# Эндпоинты: /status, /stats, /calibration, /process, /upload, /videos
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from app.config import DATA_DIR, UPLOAD_DIR
from app.api.schemas import (
    StatusResponse,
    StatsResponse,
    CalibrationRequest,
    CalibrationResponse,
    ProcessVideoRequest,
    ProcessVideoResponse,
    UploadResponse,
    VideoInfo,
    VideosListResponse,
)

router = APIRouter()


def _get_pipeline():
    from app.main import pipeline
    return pipeline


@router.get("/status", response_model=StatusResponse, tags=["pipeline"])
async def get_status():
    p = _get_pipeline()
    s = p.status
    return StatusResponse(
        initialized=s["initialized"],
        running=s["running"],
        source=s["source"],
        fps=s["fps"],
    )


@router.get("/stats", response_model=StatsResponse, tags=["pipeline"])
async def get_stats():
    p = _get_pipeline()
    stats = p.counter.stats
    return StatsResponse(**stats)


@router.post("/calibration", response_model=CalibrationResponse, tags=["config"])
async def set_calibration(body: CalibrationRequest):
    p = _get_pipeline()
    p.counter.update_line((body.x1, body.y1), (body.x2, body.y2), relative=body.relative)
    line = {"x1": body.x1, "y1": body.y1, "x2": body.x2, "y2": body.y2, "relative": body.relative}
    return CalibrationResponse(success=True, line=line)


@router.post("/process", response_model=ProcessVideoResponse, tags=["pipeline"])
async def process_video(body: ProcessVideoRequest):
    """прогнать видео - статистика подсчёта"""
    p = _get_pipeline()
    if not p.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline не инициализирован")
    if p._running:
        raise HTTPException(status_code=409, detail="Обработка уже запущена")
    try:
        result = p.process_video(body.video_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ProcessVideoResponse(
        frames_processed=result["frames_processed"],
        avg_latency_ms=result["avg_latency_ms"],
        fps=result["fps"],
        stats=StatsResponse(**result["stats"]),
    )


@router.post("/stop", tags=["pipeline"])
async def stop_processing():
    """стоп текущей обработки видео"""
    p = _get_pipeline()
    p.stop()
    return {"status": "stopped"}


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    """сохранить загруженный файл в папку"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    with open(dest, "wb") as f:
        while chunk := file.file.read(8 * 1024 * 1024):
            f.write(chunk)
    return dest


@router.post("/upload", response_model=UploadResponse, tags=["videos"])
async def upload_video(file: UploadFile = File(...)):
    """залить видео в data/uploads/"""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Поддерживаются форматы: mp4, avi, mov, mkv")
    dest = _save_upload(file, UPLOAD_DIR)
    size_mb = dest.stat().st_size / (1024 * 1024)
    return UploadResponse(filename=dest.name, path=str(dest), size_mb=round(size_mb, 2))


@router.post("/upload-and-process", response_model=ProcessVideoResponse, tags=["videos"])
async def upload_and_process(
    file: UploadFile = File(...),
    save: bool = Query(False, description="Сохранить видео в data/uploads/ после обработки"),
):
    """залить видео и сразу обработать, опц сохранить"""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Поддерживаются форматы: mp4, avi, mov, mkv")
    p = _get_pipeline()
    if not p.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline не инициализирован")
    if p._running:
        raise HTTPException(status_code=409, detail="Обработка уже запущена")

    dest = _save_upload(file, UPLOAD_DIR if save else UPLOAD_DIR / "_tmp")
    try:
        result = p.process_video(dest)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        if not save and dest.exists():
            dest.unlink()

    return ProcessVideoResponse(
        frames_processed=result["frames_processed"],
        avg_latency_ms=result["avg_latency_ms"],
        fps=result["fps"],
        stats=StatsResponse(**result["stats"]),
    )


def _list_videos(directory: Path) -> list[VideoInfo]:
    """собрать видеофайлы из папки"""
    videos = []
    if not directory.exists():
        return videos
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            videos.append(VideoInfo(
                filename=f.name,
                path=str(f),
                size_mb=round(f.stat().st_size / (1024 * 1024), 2),
            ))
    return videos


@router.get("/videos", response_model=VideosListResponse, tags=["videos"])
async def list_videos():
    """все доступные видео (data/ + data/uploads/)"""
    videos = _list_videos(DATA_DIR)
    return VideosListResponse(videos=videos, total=len(videos))
