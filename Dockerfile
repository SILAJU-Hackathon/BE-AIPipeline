# ===== Build Stage =====
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300

WORKDIR /app

# Install PyTorch CPU first (largest download)
RUN pip install --target=/app/deps \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --timeout 300 \
    torch torchvision

# Install remaining dependencies
RUN pip install --target=/app/deps \
    --timeout 300 \
    torch-geometric \
    numpy pandas scipy joblib pyarrow pillow \
    fastapi uvicorn pydantic python-multipart \
    httpx cloudinary python-dotenv psycopg2-binary

# ===== Runtime Stage =====
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/deps:/app" \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy dependencies from builder
COPY --from=builder /app/deps /app/deps

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/

# Create output and temp directories
RUN mkdir -p /app/outputnye /app/temp && \
    chown -R appuser:appgroup /app/outputnye /app/temp

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Production command
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
