# SentinelRay — production container image
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 sentinel \
    && useradd --uid 10001 --gid sentinel --create-home --shell /bin/bash sentinel

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY config.py ingestion_engine.py qa_validator.py drift_detector.py orchestrator.py main.py pytest.ini ./
COPY tests/ ./tests/

RUN mkdir -p /app/data/quarantine /app/alerts /app/logs \
    && chown -R sentinel:sentinel /app

USER sentinel

EXPOSE 8265

CMD ["python", "main.py"]
