# justfile for Omega Trading System
# Task-ID: P4-09 | Phase: 4 – Build-System
#
# This justfile provides identical targets to the Makefile for users
# who prefer the `just` command runner (https://github.com/casey/just).
#
# Usage:
#   just --list       - Show all available recipes
#   just all          - Build and test everything
#   just rust-build   - Build Rust module with Maturin
#   just julia-test   - Run Julia tests
#   just python-test  - Run Python tests
#
# Installation:
#   brew install just       # macOS
#   cargo install just      # via Cargo
#   apt install just        # Debian/Ubuntu (newer versions)

# ==============================================================================
# Configuration
# ==============================================================================

# Load environment variables from .env file if it exists
set dotenv-load

# Shell configuration
set shell := ["bash", "-c"]

# Directories
project_root := justfile_directory()
rust_dir := project_root / "src/rust_modules/omega_rust"
julia_dir := project_root / "src/julia_modules/omega_julia"
python_src := project_root / "src"
tests_dir := project_root / "tests"
benchmarks_dir := tests_dir / "benchmarks"
property_tests_dir := tests_dir / "property"
golden_tests_dir := tests_dir / "golden"

# Tool commands
python := "python"
pip := python + " -m pip"
pytest := python + " -m pytest"
maturin := "maturin"
cargo := "cargo"
julia := "julia"

# Options
pytest_args := "-v"
cargo_test_args := "--all-features"
julia_args := "--project=" + julia_dir

# Default recipe
default: help

# ==============================================================================
# Help
# ==============================================================================

# Show all available recipes
@help:
    echo "Omega Trading System - Development Commands"
    echo ""
    echo "Usage: just [recipe]"
    echo ""
    just --list --unsorted

# ==============================================================================
# Installation Recipes
# ==============================================================================

# Install Python package in editable mode
install:
    @echo "Installing Omega package..."
    {{pip}} install -e .

# Install Python package with dev dependencies
install-dev:
    @echo "Installing Omega with dev dependencies..."
    {{pip}} install -e ".[dev,analysis]"

# Install Python package with all dependencies
install-all:
    @echo "Installing Omega with all dependencies..."
    {{pip}} install -e ".[all]"

# Install pre-commit hooks
install-pre-commit:
    @echo "Installing pre-commit hooks..."
    {{pip}} install pre-commit
    pre-commit install

# ==============================================================================
# Rust Recipes
# ==============================================================================

# Build Rust module with Maturin (release mode)
rust-build:
    @echo "Building Rust module..."
    @if [ -d "{{rust_dir}}" ] && [ -f "{{rust_dir}}/Cargo.toml" ]; then \
        cd {{rust_dir}} && {{maturin}} develop --release; \
    else \
        echo "Rust module not found at {{rust_dir}}"; \
    fi

# Build Rust module in debug mode (faster compilation)
rust-build-debug:
    @echo "Building Rust module (debug)..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{maturin}} develop; \
    fi

# Run Rust unit tests
rust-test:
    @echo "Running Rust tests..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} test {{cargo_test_args}}; \
    else \
        echo "Rust module not found"; \
    fi

# Run Rust benchmarks
rust-bench:
    @echo "Running Rust benchmarks..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} bench; \
    fi

# Run Clippy linter on Rust code
rust-clippy:
    @echo "Running Clippy..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} clippy --all-targets --all-features -- -D warnings; \
    fi

# Format Rust code
rust-fmt:
    @echo "Formatting Rust code..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} fmt; \
    fi

# Check Rust code formatting
rust-fmt-check:
    @echo "Checking Rust formatting..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} fmt -- --check; \
    fi

# Generate Rust documentation
rust-doc:
    @echo "Generating Rust documentation..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} doc --no-deps --open; \
    fi

# Clean Rust build artifacts
rust-clean:
    @echo "Cleaning Rust build artifacts..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} clean; \
    fi

# Run security audit on Rust dependencies
rust-audit:
    @echo "Auditing Rust dependencies..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{cargo}} audit || echo "Install cargo-audit: cargo install cargo-audit"; \
    fi

# Build Rust wheel for distribution
rust-wheel:
    @echo "Building Rust wheel..."
    @if [ -d "{{rust_dir}}" ]; then \
        cd {{rust_dir}} && {{maturin}} build --release; \
    fi

# ==============================================================================
# Julia Recipes
# ==============================================================================

# Instantiate Julia project dependencies
julia-instantiate:
    @echo "Instantiating Julia dependencies..."
    @if [ -d "{{julia_dir}}" ]; then \
        {{julia}} {{julia_args}} -e 'using Pkg; Pkg.instantiate()'; \
    else \
        echo "Julia module not found at {{julia_dir}}"; \
    fi

# Run Julia tests
julia-test:
    @echo "Running Julia tests..."
    @if [ -d "{{julia_dir}}" ]; then \
        {{julia}} {{julia_args}} -e 'using Pkg; Pkg.test()'; \
    else \
        echo "Julia module not found"; \
    fi

# Run Julia benchmarks
julia-bench:
    @echo "Running Julia benchmarks..."
    @if [ -d "{{julia_dir}}" ] && [ -f "{{julia_dir}}/benchmark/benchmarks.jl" ]; then \
        {{julia}} {{julia_args}} {{julia_dir}}/benchmark/benchmarks.jl; \
    else \
        echo "Julia benchmarks not found"; \
    fi

# Precompile Julia packages
julia-precompile:
    @echo "Precompiling Julia packages..."
    @if [ -d "{{julia_dir}}" ]; then \
        {{julia}} {{julia_args}} -e 'using Pkg; Pkg.precompile()'; \
    fi

# Update Julia dependencies
julia-update:
    @echo "Updating Julia dependencies..."
    @if [ -d "{{julia_dir}}" ]; then \
        {{julia}} {{julia_args}} -e 'using Pkg; Pkg.update()'; \
    fi

# Start Julia REPL with project environment
julia-repl:
    @echo "Starting Julia REPL..."
    {{julia}} {{julia_args}}

# ==============================================================================
# Python Recipes
# ==============================================================================

# Run Python tests
python-test:
    @echo "Running Python tests..."
    {{pytest}} {{tests_dir}} {{pytest_args}}

# Run Python tests with coverage
python-test-cov:
    @echo "Running Python tests with coverage..."
    {{pytest}} {{tests_dir}} --cov={{python_src}} --cov-report=term-missing {{pytest_args}}

# Run Python tests (fast mode, skip slow tests)
python-test-fast:
    @echo "Running fast Python tests..."
    {{pytest}} {{tests_dir}} -m "not slow" {{pytest_args}}

# Run benchmark tests
benchmark:
    @echo "Running benchmark tests..."
    {{pytest}} {{benchmarks_dir}} {{pytest_args}} --benchmark-only

# Run benchmarks and save results
benchmark-save:
    @echo "Running benchmarks with JSON output..."
    {{pytest}} {{benchmarks_dir}} {{pytest_args}} --benchmark-json=benchmark_results.json

# Run property-based tests
property-test:
    @echo "Running property-based tests..."
    {{pytest}} {{property_tests_dir}} {{pytest_args}}

# Run golden file tests
golden-test:
    @echo "Running golden file tests..."
    {{pytest}} {{golden_tests_dir}} {{pytest_args}}

# Regenerate golden files
golden-regenerate:
    @echo "Regenerating golden files..."
    {{pytest}} {{golden_tests_dir}} --regenerate-golden-files {{pytest_args}}

# ==============================================================================
# Code Quality Recipes
# ==============================================================================

# Run all linters (black, isort, flake8, mypy)
lint:
    @echo "Running linters..."
    {{python}} -m black --check {{python_src}} {{tests_dir}}
    {{python}} -m isort --check-only {{python_src}} {{tests_dir}}
    {{python}} -m flake8 {{python_src}} {{tests_dir}}
    {{python}} -m mypy {{python_src}}

# Format Python code (black + isort)
format:
    @echo "Formatting Python code..."
    {{python}} -m black {{python_src}} {{tests_dir}}
    {{python}} -m isort {{python_src}} {{tests_dir}}

# Run mypy type checker
mypy:
    @echo "Running mypy..."
    {{python}} -m mypy {{python_src}}

# Run mypy in strict mode on migrations candidates
mypy-strict:
    @echo "Running mypy (strict mode)..."
    {{python}} -m mypy {{python_src}}/backtest_engine --strict

# Run security checks (bandit, pip-audit)
security:
    @echo "Running security checks..."
    {{python}} -m bandit -r {{python_src}} -ll
    {{python}} -m pip_audit

# Run pre-commit on all files
pre-commit:
    @echo "Running pre-commit hooks..."
    pre-commit run -a

# ==============================================================================
# Combined Recipes
# ==============================================================================

# Build and test everything
all: install-dev rust-build julia-instantiate test
    @echo "All builds and tests completed!"

# Run all tests (Python, Rust, Julia)
test: python-test rust-test julia-test
    @echo "All tests completed!"

# Build all native extensions
build: rust-build
    @echo "Build completed!"

# Run CI checks
ci: lint rust-clippy rust-fmt-check python-test rust-test
    @echo "CI checks completed!"

# Run complete verification
full-check: lint security mypy python-test-cov property-test golden-test rust-clippy rust-test
    @echo "Full verification completed!"

# ==============================================================================
# Development Workflow Recipes
# ==============================================================================

# Complete development setup
dev-setup: install-dev install-pre-commit rust-build julia-instantiate
    @echo "Development environment ready!"

# Run tests on file changes (requires pytest-watch)
watch:
    @echo "Starting test watcher..."
    {{python}} -m pytest_watch -- {{tests_dir}} {{pytest_args}}

# ==============================================================================
# Documentation Recipes
# ==============================================================================

# Generate all documentation
docs: rust-doc
    @echo "Documentation generated!"

# ==============================================================================
# Cleanup Recipes
# ==============================================================================

# Clean all build artifacts
clean: rust-clean
    @echo "Cleaning build artifacts..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ 2>/dev/null || true
    @echo "Clean completed!"

# Clean everything including venv
clean-all: clean
    @echo "Warning: This will remove .venv directory"
    @read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf .venv || true

# ==============================================================================
# Utility Recipes
# ==============================================================================

# Show versions of all tools
version:
    @echo "Tool Versions:"
    @echo -n "Python: " && {{python}} --version
    @echo -n "pip: " && {{pip}} --version | cut -d' ' -f2
    @echo -n "Rust: " && rustc --version 2>/dev/null || echo "not installed"
    @echo -n "Cargo: " && cargo --version 2>/dev/null || echo "not installed"
    @echo -n "Maturin: " && {{maturin}} --version 2>/dev/null || echo "not installed"
    @echo -n "Julia: " && {{julia}} --version 2>/dev/null || echo "not installed"

# Check if all required tools are installed
check-deps:
    @echo "Checking dependencies..."
    @command -v {{python}} >/dev/null 2>&1 || { echo "Python not found"; exit 1; }
    @command -v cargo >/dev/null 2>&1 || { echo "Cargo not found (Rust builds will be skipped)"; }
    @command -v {{julia}} >/dev/null 2>&1 || { echo "Julia not found (Julia tests will be skipped)"; }
    @echo "Dependency check passed!"
