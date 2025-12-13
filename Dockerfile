# syntax=docker/dockerfile:1.4
# ================================
# AURA Chatbot Service
# AI Conversational Support with Gemini + NLP
# OPTIMIZED: BuildKit cache mounts + shared NLP models
# ================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment variables
# NOTE: Removed PIP_NO_CACHE_DIR to enable caching
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/models \
    HF_HOME=/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount
# This caches pip downloads between builds, saving ~1GB on rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# NOTE: NLP model is NOT downloaded during build anymore
# It will be loaded from the shared /models volume at runtime
# This saves ~2GB per build and shares the model with clustering-service

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/api/v1/chat/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
