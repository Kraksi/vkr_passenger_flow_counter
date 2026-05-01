# Pydantic-схемы запросов и ответов API
from pydantic import BaseModel


class StatusResponse(BaseModel):
    initialized: bool
    running: bool
    source: str | None = None
    fps: float = 0.0


class StatsResponse(BaseModel):
    entries: int
    exits: int
    total: int
    current_inside: int


class CalibrationRequest(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    relative: bool = False


class CalibrationResponse(BaseModel):
    success: bool
    line: dict


class ProcessVideoRequest(BaseModel):
    video_path: str


class ProcessVideoResponse(BaseModel):
    frames_processed: int
    avg_latency_ms: float
    fps: float
    stats: StatsResponse


class UploadResponse(BaseModel):
    filename: str
    path: str
    size_mb: float


class VideoInfo(BaseModel):
    filename: str
    path: str
    size_mb: float


class VideosListResponse(BaseModel):
    videos: list[VideoInfo]
    total: int
