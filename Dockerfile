# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

LABEL maintainer="galafis"
LABEL description="COVID-19 ML Prediction Pipeline"

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY tests/ ./tests/

RUN mkdir -p data/raw data/processed models/artifacts reports

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV LOG_LEVEL=INFO \
    MODEL_ALGORITHM=random_forest \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import src.config" || exit 1

CMD ["python", "-m", "src.pipeline"]
