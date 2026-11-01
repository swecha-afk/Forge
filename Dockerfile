# Agritech Swarm Engine — Docker Production Image
# Theme 2, Problem 1: Multilingual Agritech & Resource Advisory Assistant

FROM python:3.11-slim

LABEL maintainer="Forge Buildathon Team"
LABEL description="Agritech Swarm Execution Engine — Multilingual AI Advisory for Indian Smallholder Farmers"

WORKDIR /app

# Install system dependencies for ML libraries (LightGBM, CatBoost, OpenMP, FAISS)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgcc-s1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (faiss-cpu and sentence-transformers separately for Linux wheels)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir faiss-cpu sentence-transformers

# Copy application source
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY data/ ./data/
COPY benchmarks/ ./benchmarks/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/.mplcache
ENV PYTHONDONTWRITEBYTECODE=1

# Render sets PORT for web services; use 10000 for local Docker runs.
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://localhost:{os.getenv('PORT', '10000')}/health\")" || exit 1

# Keep one worker to limit memory use on small deployment instances.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --log-level info"]
