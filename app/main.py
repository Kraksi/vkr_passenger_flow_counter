# Точка входа FastAPI — запуск через: uv run uvicorn app.main:app --reload
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.pipeline import Pipeline
from app.api.routes import router

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте, очистка при остановке."""
    pipeline.initialize()
    yield
    pipeline.shutdown()


app = FastAPI(
    title="Счётчик пассажиропотока",
    description="REST API для подсчёта входов/выходов через виртуальную линию",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}
