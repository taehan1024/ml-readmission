# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Installs packages into a clean prefix so the final image copies only
# the compiled wheels, not pip's cache or build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs needed to compile some wheels (e.g. cryptography, pyarrow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 is required by XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# models/ is mounted as a volume at runtime (see docker-compose.yml)
# so we only create the directory as a mount point here
RUN mkdir -p models data && chown -R appuser:appuser models data

# Switch to non-root
USER appuser

EXPOSE 8000

# Uvicorn settings via env vars; overridable in docker-compose or Railway
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
