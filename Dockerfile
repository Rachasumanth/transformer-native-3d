FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    T3D_ARTIFACTS=/models \
    T3D_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY t3d ./t3d
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[service]"

RUN useradd --create-home --uid 1000 t3d \
    && mkdir -p /cache/huggingface /models \
    && chown -R t3d:t3d /cache /models /app
USER t3d

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "t3d.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
