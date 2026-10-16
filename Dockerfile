
# Production inference image for Heart Disease MLOps.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_SOURCE=local \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Install only inference dependencies.
COPY requirements-serving.txt .

RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy the application and trained model.
COPY api/ ./api/
COPY src/ ./src/
COPY models/production/ ./models/production/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
