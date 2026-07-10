# syntax=docker/dockerfile:1
#
# Multi-stage build for the telemetry anomaly detector.
#   * target "api"  (default) -> FastAPI service on :8000, with a trained model baked in.
#   * target "demo"           -> Streamlit dashboard on :8501 (trains on first load).
#
#   docker build -t tad-api .                       # the API (default target)
#   docker build -t tad-demo --target demo .        # the Streamlit demo
#   docker compose up                               # both, via docker-compose.yml

# ---- Base: shared dependencies + application code ---------------------------
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# scikit-learn / scipy manylinux wheels need OpenMP (libgomp) at runtime, which
# the slim image doesn't ship by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the exact, reproducible dependency set first for better layer caching.
COPY requirements-lock.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-lock.txt

# Application code + committed showcase artifacts (used by the demo's About panel).
COPY src/ ./src/
COPY api/ ./api/
COPY streamlit_app.py ./
COPY models/comparison.json models/metrics.json ./models/
COPY reports/ ./reports/

# Unprivileged runtime user.
RUN useradd --create-home --uid 1000 appuser

# ---- Demo: Streamlit dashboard ----------------------------------------------
FROM base AS demo

RUN python -m pip install "streamlit>=1.40,<2.0"

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# ---- API: FastAPI service (default target) ----------------------------------
FROM base AS api

# Bake a trained model into the image so `docker run` serves real scores with no
# volume mount or startup training.
RUN python -m src.train --seed 1337

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
