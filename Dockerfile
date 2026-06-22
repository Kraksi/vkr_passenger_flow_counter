
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt \
    && pip install --no-deps boxmot==13.0.17

RUN pip uninstall -y opencv-python opencv-python-headless \
        opencv-contrib-python opencv-contrib-python-headless 2>/dev/null; \
    pip install opencv-python-headless==4.13.0.92

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY app/ ./app/

COPY models/yolo11n_mot20_v2.pt   ./models/
COPY models/osnet_x0_25_msmt17.pt ./models/
COPY models/yolo11_head.pt           ./models/
COPY models/osnet_x0_25_mot20head.pt ./models/

ENV VKR_STATE_DIR=/app/state \
    YOLO_CONFIG_DIR=/app/state/Ultralytics
RUN mkdir -p /app/data/uploads /app/state \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
