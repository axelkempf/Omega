# Multi-Stage Dockerfile für Omega Trading Stack
# Optimiert für Production mit minimaler Image-Größe

# ============================================
# Stage 1: Build Stage
# ============================================
FROM python:3.14-slim AS builder

# Build-Dependencies installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /build

# Python Dependencies installieren
COPY pyproject.toml .
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ============================================
# Stage 2: Runtime Stage
# ============================================
FROM python:3.14-slim

# Metadata
LABEL maintainer="Axel Kempf"
LABEL description="Omega Trading Engine - Multi-Tech Trading Stack"
LABEL version="1.2.0"

# Non-root User erstellen
RUN groupadd -r omega && useradd -r -g omega omega

# Runtime Dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Virtual Environment von Builder kopieren
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Working Directory
WORKDIR /app

# Application Code kopieren
COPY --chown=omega:omega src/ ./src/
COPY --chown=omega:omega configs/ ./configs/
COPY --chown=omega:omega README.md .
COPY --chown=omega:omega pyproject.toml .

# Runtime Verzeichnisse erstellen
RUN mkdir -p var/logs var/results var/tmp data && \
    chown -R omega:omega var/ data/

# Wechsel zu Non-Root User
USER omega

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Expose Ports (FastAPI UI)
EXPOSE 8000

# Default Command: UI Engine
CMD ["uvicorn", "src.ui_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
