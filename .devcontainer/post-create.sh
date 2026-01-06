#!/bin/bash
# Post-Create Script for Omega Dev Container
# Task-ID: P4-10 | Phase: 4 – Build-System
#
# This script runs after the dev container is created.
# It sets up the complete development environment.

set -e

echo "🔧 Setting up Omega development environment..."

# ==============================================================================
# Python Setup
# ==============================================================================

echo "📦 Installing Python dependencies..."

# Install the package in editable mode with all dev dependencies
pip install -e ".[dev,analysis]" --quiet

# Verify installation
echo "✅ Python packages installed"
python --version
pip list | head -20

# ==============================================================================
# Rust Setup
# ==============================================================================

echo "🦀 Setting up Rust environment..."

# Verify Rust installation
if command -v rustc &> /dev/null; then
    rustc --version
    cargo --version
    maturin --version
    
    # Build Rust module if it exists
    if [ -d "src/rust_modules/omega_rust" ] && [ -f "src/rust_modules/omega_rust/Cargo.toml" ]; then
        echo "🔨 Building Rust module..."
        cd src/rust_modules/omega_rust
        # Use maturin build + pip install instead of maturin develop
        # to avoid virtual environment requirement issues
        maturin build --release --out dist
        pip install dist/*.whl --force-reinstall
        cd /workspaces/omega
        echo "✅ Rust module built"
    else
        echo "⚠️  Rust module not found, skipping build"
    fi
else
    echo "⚠️  Rust not installed, skipping Rust setup"
fi

# ==============================================================================
# Julia Setup
# ==============================================================================

echo "🔮 Setting up Julia environment..."

# Verify Julia installation
if command -v julia &> /dev/null; then
    julia --version
    
    # Set up Julia environment if it exists
    if [ -d "src/julia_modules/omega_julia" ] && [ -f "src/julia_modules/omega_julia/Project.toml" ]; then
        echo "📥 Installing Julia packages..."
        julia --project=src/julia_modules/omega_julia -e '
            using Pkg
            Pkg.instantiate()
            Pkg.precompile()
        '
        echo "✅ Julia packages installed"
    else
        echo "⚠️  Julia module not found, skipping setup"
    fi
else
    echo "⚠️  Julia not installed, skipping Julia setup"
fi

# ==============================================================================
# Git Setup
# ==============================================================================

echo "🔧 Configuring Git..."

# Configure Git to use VS Code as editor
git config --global core.editor "code --wait"
git config --global init.defaultBranch main

# Set up pre-commit hooks if available
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

# ==============================================================================
# Directory Setup
# ==============================================================================

echo "📁 Setting up directories..."

# Create necessary directories
mkdir -p var/logs var/results var/tmp
mkdir -p data/csv data/parquet data/raw

# ==============================================================================
# Verification
# ==============================================================================

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  make help         - Show all Make targets"
echo "  make test         - Run all tests"
echo "  make rust-build   - Build Rust module"
echo "  make julia-test   - Run Julia tests"
echo "  make lint         - Run linters"
echo ""
echo "Or use 'just' if you prefer:"
echo "  just --list       - Show all recipes"
echo ""

# Run a quick sanity check
echo "🧪 Running quick sanity check..."
python -c "import omega_rust; print('✅ omega_rust module available')" 2>/dev/null || echo "⚠️  omega_rust not yet available (build with 'make rust-build')"
python -c "from juliacall import Main as jl; print('✅ juliacall available')" 2>/dev/null || echo "⚠️  juliacall not yet available (install with 'pip install juliacall')"

echo ""
echo "🚀 Ready to develop!"
