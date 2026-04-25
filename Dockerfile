# Multi-stage build using openenv-base
# Adapted from support_env example for the Chess Arena environment.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git and stockfish are available
RUN apt-get update && \
    apt-get install -y --no-install-recommends git stockfish && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=chess_arena

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Install dependencies using uv sync
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Drop bytecode caches to shrink layers
RUN /app/env/.venv/bin/python -c "\
import pathlib, shutil; \
root = pathlib.Path('/app/env/.venv'); \
[p.unlink(missing_ok=True) for p in root.rglob('*.py[co]')]; \
[shutil.rmtree(p) for p in root.rglob('__pycache__') if p.is_dir()] \
"

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app/env

# Install stockfish in the final image as well
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire environment directory (including .venv) from builder
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/env/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
