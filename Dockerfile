# Default build (CI, docker compose):   docker build -t pdm-api .                      -> mount ./models at runtime
# Kubernetes / self-contained image:    docker build --target baked -t pdm-api:1.0 .   -> model copied into image
FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py features.py validate.py evaluate.py policy.py explain.py report.py train.py monitor.py api.py app.py params.yaml ./
COPY scripts ./scripts
RUN useradd -m -u 1000 appuser && mkdir -p logs models reports && chown -R appuser /app
ENV PDM_PRED_LOG=/app/logs/predictions.jsonl
EXPOSE 8000 8501

FROM base AS baked
COPY --chown=appuser models ./models
COPY --chown=appuser reports ./reports
USER appuser
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# last stage = default target (no model inside)
FROM base AS runtime
USER appuser
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
