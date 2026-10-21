# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
WORKDIR /app
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY data/raw/breast_cancer.csv ./data/raw/breast_cancer.csv

FROM python:3.12-slim AS production
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-serving.txt

COPY api/ ./api/
COPY --from=builder /app/artifacts/ ./artifacts/
COPY --from=builder /app/data/raw/breast_cancer.csv ./data/raw/breast_cancer.csv

ARG APP_VERSION=1.0.0
LABEL org.opencontainers.image.title="Breast Cancer Triage API" \
      org.opencontainers.image.version="${APP_VERSION}"

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
