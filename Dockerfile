# syntax=docker/dockerfile:1

# GPU-enabled image for the `metals` research repo.
#
# Base: CUDA 12.4 runtime + cuDNN so torch / sentence-transformers can use the
# GPU for the Phase 3 embedding pass. (torch ships its own CUDA libs in the wheel;
# the base mainly guarantees a compatible runtime + cuDNN. Any reasonably recent
# host NVIDIA driver works — the driver is forward-compatible.)
#
# Python and every dependency are managed by uv. The project venv lives at
# /opt/venv, deliberately OUTSIDE /workspace, so the runtime bind mount
# (`.:/workspace`) can't shadow it. The `metals` package is editable-installed,
# and its editable pointer targets /workspace/src — which the bind mount resolves
# to the live host source, so code edits need no rebuild (only dependency changes
# do).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# uv binary. Pin this tag deliberately when you want a reproducible toolchain.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates build-essential git \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON=3.11 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_FROZEN=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    METALS_EMBEDDING_CACHE_DIR=/cache/metals/embeddings

# Standalone CPython 3.11, managed by uv (Ubuntu 22.04 ships 3.10).
RUN uv python install 3.11

WORKDIR /workspace

# 1) Dependency layer — cached unless pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --no-install-project

# 2) Project layer — editable install of `metals`. Fast: deps already cached.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev

CMD ["bash"]
