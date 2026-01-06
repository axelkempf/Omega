# Dev Container Dockerfile for Omega Trading System
# Task-ID: P4-10 | Phase: 4 – Build-System
#
# Multi-stage build for a complete Python + Rust + Julia development environment.
# Optimized for VS Code Dev Containers with all necessary tooling pre-installed.
#
# Build arguments:
#   PYTHON_VERSION - Python version (default: 3.12)
#   RUST_VERSION   - Rust version (default: 1.83.0)
#   JULIA_VERSION  - Julia version (default: 1.10)

# ==============================================================================
# Base Stage: Python + System Dependencies
# ==============================================================================

FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm AS base

ARG PYTHON_VERSION=3.12
ARG RUST_VERSION=1.89
ARG JULIA_VERSION=1.10

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    # SSL/TLS
    libssl-dev \
    # For Rust builds
    llvm \
    clang \
    # For Julia
    wget \
    curl \
    ca-certificates \
    # Useful development tools
    git \
    vim \
    less \
    jq \
    htop \
    # For matplotlib/plotting
    libfreetype6-dev \
    libpng-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Rust Installation
# ==============================================================================

# Install Rust as vscode user
USER vscode
ENV CARGO_HOME=/home/vscode/.cargo
ENV RUSTUP_HOME=/home/vscode/.rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
    -y \
    --default-toolchain ${RUST_VERSION} \
    --profile default \
    --component rustfmt clippy

# Install additional Rust tools
RUN cargo install maturin --version ">=1.4" \
    && cargo install cargo-audit \
    && cargo install cargo-watch

# ==============================================================================
# Julia Installation
# ==============================================================================

USER root

# Download and install Julia
RUN JULIA_MINOR=$(echo ${JULIA_VERSION} | cut -d. -f1,2) \
    && JULIA_PATCH=$(curl -sL "https://julialang-s3.julialang.org/bin/versions.json" | jq -r "keys[] | select(startswith(\"${JULIA_MINOR}\"))" | sort -V | tail -1) \
    && if [ -z "$JULIA_PATCH" ]; then JULIA_PATCH="${JULIA_VERSION}.0"; fi \
    && echo "Installing Julia ${JULIA_PATCH}" \
    && ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then JULIA_ARCH="x64"; else JULIA_ARCH="aarch64"; fi \
    && wget -q "https://julialang-s3.julialang.org/bin/linux/${JULIA_ARCH}/${JULIA_MINOR}/julia-${JULIA_PATCH}-linux-${JULIA_ARCH}.tar.gz" -O /tmp/julia.tar.gz \
    && mkdir -p /usr/local/julia \
    && tar -xzf /tmp/julia.tar.gz -C /usr/local/julia --strip-components=1 \
    && rm /tmp/julia.tar.gz \
    && ln -s /usr/local/julia/bin/julia /usr/local/bin/julia

# Set Julia depot path
ENV JULIA_DEPOT_PATH=/home/vscode/.julia
RUN mkdir -p ${JULIA_DEPOT_PATH} && chown -R vscode:vscode ${JULIA_DEPOT_PATH}

# ==============================================================================
# Python Environment Setup
# ==============================================================================

USER vscode
WORKDIR /workspaces/omega

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install common Python development tools
RUN pip install --no-cache-dir \
    ipython \
    jupyterlab \
    notebook \
    pre-commit

# ==============================================================================
# Final Setup
# ==============================================================================

# Create workspace directory
USER root
RUN mkdir -p /workspaces/omega && chown -R vscode:vscode /workspaces/omega

USER vscode
WORKDIR /workspaces/omega

# Default command
CMD ["sleep", "infinity"]
