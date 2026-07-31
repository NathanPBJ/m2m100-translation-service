FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/appuser \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache \
    TORCH_HOME=/models/torch

WORKDIR /service

RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --create-home --home-dir /home/appuser appuser \
    && mkdir -p /service /models \
    && chown appuser:appgroup /service /models

COPY --chown=appuser:appgroup requirements.txt /service/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0+cpu" \
    && python -m pip install -r /service/requirements.txt

COPY --chown=appuser:appgroup app /service/app
COPY --chown=appuser:appgroup LICENSE /service/LICENSE

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8); data = json.load(response); assert 200 <= response.status < 300 and data.get('status') == 'healthy' and data.get('model_loaded') is True and data.get('language_detector_loaded') is True"]

USER appuser

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
