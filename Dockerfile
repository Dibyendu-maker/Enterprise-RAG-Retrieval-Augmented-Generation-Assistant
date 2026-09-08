# Multi-stage Dockerfile using uv for ultra-fast, reproducible builds
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.12-slim-bookworm

# Copy uv binary
COPY --from=uv_bin /uv /uvx /bin/

# Set working directory and environment variables
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv into virtual environment
RUN uv sync --no-install-project --no-dev

# Copy application source code
COPY app/ ./app/
COPY main.py README.md ./

# Create data directories
RUN mkdir -p data/sqlite data/chromadb data/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
