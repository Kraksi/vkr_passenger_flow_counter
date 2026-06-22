"""эндпоинты суточной и почасовой статистики потока"""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.storage.db import event_store

router = APIRouter(prefix="/stats", tags=["stats"])



class DailyStatsResponse(BaseModel):
    date: str
    cam_id: Optional[str]
    entries: int
    exits: int
    total: int
    current_inside: int


class HourlySlot(BaseModel):
    hour: int
    entries: int
    exits: int


class HourlyStatsResponse(BaseModel):
    date: str
    cam_id: Optional[str]
    hourly: list[HourlySlot]
    totals: DailyStatsResponse


class RangeStatsItem(BaseModel):
    date: str
    cam_id: Optional[str]
    entries: int
    exits: int
    total: int


class RangeStatsResponse(BaseModel):
    start_date: str
    end_date: str
    cam_id: Optional[str]
    days: list[RangeStatsItem]
    totals: dict


class CamerasListResponse(BaseModel):
    cameras: list[str]


class DatesListResponse(BaseModel):
    cam_id: Optional[str]
    dates: list[str]



def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _validate_date(d: str) -> None:
    try:
        date.fromisoformat(d)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Неверный формат даты: {d!r}. Ожидается YYYY-MM-DD")



@router.get(
    "/daily",
    response_model=DailyStatsResponse,
    summary="Статистика за день",
    description=(
        "Суммарные входы/выходы за указанный день"
        "По умолчанию - сегодня"
        "Если **cam_id** не указан - сумма по всем камерам"
    ),
)
async def get_daily_stats(
    date: str = Query(None, description="Дата UTC в формате YYYY-MM-DD (по умолчанию - сегодня)"),
    cam_id: Optional[str] = Query(None, description="ID камеры (пусто = все камеры)"),
):
    date_str = date or _today_utc()
    _validate_date(date_str)
    raw = event_store.get_daily_stats(date_str, cam_id=cam_id)
    return DailyStatsResponse(
        **raw,
        current_inside=max(0, raw["entries"] - raw["exits"]),
    )


@router.get(
    "/hourly",
    response_model=HourlyStatsResponse,
    summary="Почасовая разбивка за день",
    description=(
        "Почасовая разбивка событий за указанный день"
        "Возвращает 24 записи (часы 0-23)"
        "Если **cam_id** не указан - сумма по всем камерам"
    ),
)
async def get_hourly_stats(
    date: str = Query(None, description="Дата UTC в формате YYYY-MM-DD"),
    cam_id: Optional[str] = Query(None, description="ID камеры"),
):
    date_str = date or _today_utc()
    _validate_date(date_str)

    hourly_raw = event_store.get_hourly_breakdown(date_str, cam_id=cam_id)
    daily_raw = event_store.get_daily_stats(date_str, cam_id=cam_id)

    hourly = [HourlySlot(**slot) for slot in hourly_raw]
    totals = DailyStatsResponse(
        **daily_raw,
        current_inside=max(0, daily_raw["entries"] - daily_raw["exits"]),
    )
    return HourlyStatsResponse(date=date_str, cam_id=cam_id, hourly=hourly, totals=totals)


@router.get(
    "/range",
    response_model=RangeStatsResponse,
    summary="Суточная статистика за диапазон дат",
    description=(
        "Статистика по дням за период **[start_date, end_date]** включительно"
        "Максимальный диапазон - 366 дней"
        "Если cam_id не указан - сумма по всем камерам"
    ),
)
async def get_range_stats(
    start_date: str = Query(..., description="Начало диапазона YYYY-MM-DD"),
    end_date: str = Query(..., description="Конец диапазона YYYY-MM-DD"),
    cam_id: Optional[str] = Query(None, description="ID камеры"),
):
    _validate_date(start_date)
    _validate_date(end_date)

    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date должен быть <= end_date")

    days_diff = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
    if days_diff > 366:
        raise HTTPException(status_code=422, detail="Максимальный диапазон - 366 дней")

    raw = event_store.get_date_range_stats(start_date, end_date, cam_id=cam_id)
    days = [RangeStatsItem(**item) for item in raw]

    total_entries = sum(d.entries for d in days)
    total_exits = sum(d.exits for d in days)
    totals = {"entries": total_entries, "exits": total_exits, "total": total_entries + total_exits}

    return RangeStatsResponse(
        start_date=start_date,
        end_date=end_date,
        cam_id=cam_id,
        days=days,
        totals=totals,
    )


@router.get(
    "/dates",
    response_model=DatesListResponse,
    summary="Даты с данными",
    description="Список дат (UTC), за которые есть хотя бы одно событие.",
)
async def get_active_dates(
    cam_id: Optional[str] = Query(None, description="ID камеры"),
):
    dates = event_store.get_active_dates(cam_id=cam_id)
    return DatesListResponse(cam_id=cam_id, dates=dates)


@router.get(
    "/cameras",
    response_model=CamerasListResponse,
    summary="Камеры с историческими данными",
    description="Список cam_id, у которых есть хотя бы одно сохранённое событие в БД.",
)
async def get_active_cameras():
    cameras = event_store.get_active_cameras()
    return CamerasListResponse(cameras=cameras)
