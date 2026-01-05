# Makefile for Omega Trading System
# Task-ID: P4-08 | Phase: 4 – Build-System
# 
# This Makefile provides convenient targets for local development
# across the Python/Rust/Julia hybrid stack.
#
# Usage:
#   make help          - Show all available targets
#   make all           - Build and test everything
#   make rust-build    - Build Rust module with Maturin
#   make julia-test    - Run Julia tests
#   make python-test   - Run Python tests
#
# Requirements:
#   - Python ≥3.12 with pip
#   - Rust ≥1.75 with Cargo
#   - Julia ≥1.9
#   - maturin (pip install maturin)

# ==============================================================================
# Configuration
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Directories
PROJECT_ROOT := $(shell pwd)
RUST_DIR := src/rust_modules/omega_rust
JULIA_DIR := src/julia_modules/omega_julia
PYTHON_SRC := src
TESTS_DIR := tests
BENCHMARKS_DIR := tests/benchmarks
PROPERTY_TESTS_DIR := tests/property_tests
GOLDEN_TESTS_DIR := tests/golden

# Tools
PYTHON := python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
MATURIN := maturin
CARGO := cargo
JULIA := julia

# Rust build options
MATURIN_ARGS := --release
CARGO_TEST_ARGS := --all-features

# Python test options
PYTEST_ARGS := -v
PYTEST_COV_ARGS := --cov=$(PYTHON_SRC) --cov-report=term-missing

# Julia options
JULIA_PROJECT := $(JULIA_DIR)
JULIA_ARGS := --project=$(JULIA_PROJECT)

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

# ==============================================================================
# Help Target
# ==============================================================================

.PHONY: help
help: ## Show this help message
	@echo -e "$(BLUE)Omega Trading System - Development Commands$(NC)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo -e "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo -e "$(GREEN)Quick Start:$(NC)"
	@echo "  make install-dev    # Install Python dev dependencies"
	@echo "  make rust-build     # Build Rust extensions"
	@echo "  make test           # Run all tests"

# ==============================================================================
# Installation Targets
# ==============================================================================

.PHONY: install
install: ## Install Python package in editable mode
	@echo -e "$(BLUE)Installing Omega package...$(NC)"
	$(PIP) install -e .

.PHONY: install-dev
install-dev: ## Install Python package with dev dependencies
	@echo -e "$(BLUE)Installing Omega with dev dependencies...$(NC)"
	$(PIP) install -e ".[dev,analysis]"

.PHONY: install-all
install-all: ## Install Python package with all dependencies (dev + analysis + ml)
	@echo -e "$(BLUE)Installing Omega with all dependencies...$(NC)"
	$(PIP) install -e ".[all]"

.PHONY: install-pre-commit
install-pre-commit: ## Install pre-commit hooks
	@echo -e "$(BLUE)Installing pre-commit hooks...$(NC)"
	$(PIP) install pre-commit
	pre-commit install

# ==============================================================================
# Rust Targets
# ==============================================================================

.PHONY: rust-build
rust-build: ## Build Rust module with Maturin (release mode)
	@echo -e "$(BLUE)Building Rust module...$(NC)"
	@if [ -d "$(RUST_DIR)" ] && [ -f "$(RUST_DIR)/Cargo.toml" ]; then \
		cd $(RUST_DIR) && $(MATURIN) develop $(MATURIN_ARGS); \
	else \
		echo -e "$(YELLOW)Rust module not found at $(RUST_DIR)$(NC)"; \
	fi

.PHONY: rust-build-debug
rust-build-debug: ## Build Rust module in debug mode (faster compilation)
	@echo -e "$(BLUE)Building Rust module (debug)...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(MATURIN) develop; \
	fi

.PHONY: rust-test
rust-test: ## Run Rust unit tests
	@echo -e "$(BLUE)Running Rust tests...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) test $(CARGO_TEST_ARGS); \
	else \
		echo -e "$(YELLOW)Rust module not found$(NC)"; \
	fi

.PHONY: rust-bench
rust-bench: ## Run Rust benchmarks
	@echo -e "$(BLUE)Running Rust benchmarks...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) bench; \
	fi

.PHONY: rust-clippy
rust-clippy: ## Run Clippy linter on Rust code
	@echo -e "$(BLUE)Running Clippy...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) clippy --all-targets --all-features -- -D warnings; \
	fi

.PHONY: rust-fmt
rust-fmt: ## Format Rust code
	@echo -e "$(BLUE)Formatting Rust code...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) fmt; \
	fi

.PHONY: rust-fmt-check
rust-fmt-check: ## Check Rust code formatting
	@echo -e "$(BLUE)Checking Rust formatting...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) fmt -- --check; \
	fi

.PHONY: rust-doc
rust-doc: ## Generate Rust documentation
	@echo -e "$(BLUE)Generating Rust documentation...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) doc --no-deps --open; \
	fi

.PHONY: rust-clean
rust-clean: ## Clean Rust build artifacts
	@echo -e "$(BLUE)Cleaning Rust build artifacts...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) clean; \
	fi

.PHONY: rust-audit
rust-audit: ## Run security audit on Rust dependencies
	@echo -e "$(BLUE)Auditing Rust dependencies...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(CARGO) audit || echo -e "$(YELLOW)Install cargo-audit: cargo install cargo-audit$(NC)"; \
	fi

.PHONY: rust-wheel
rust-wheel: ## Build Rust wheel for distribution
	@echo -e "$(BLUE)Building Rust wheel...$(NC)"
	@if [ -d "$(RUST_DIR)" ]; then \
		cd $(RUST_DIR) && $(MATURIN) build --release; \
	fi

# ==============================================================================
# Julia Targets
# ==============================================================================

.PHONY: julia-instantiate
julia-instantiate: ## Instantiate Julia project dependencies
	@echo -e "$(BLUE)Instantiating Julia dependencies...$(NC)"
	@if [ -d "$(JULIA_DIR)" ]; then \
		$(JULIA) $(JULIA_ARGS) -e 'using Pkg; Pkg.instantiate()'; \
	else \
		echo -e "$(YELLOW)Julia module not found at $(JULIA_DIR)$(NC)"; \
	fi

.PHONY: julia-test
julia-test: ## Run Julia tests
	@echo -e "$(BLUE)Running Julia tests...$(NC)"
	@if [ -d "$(JULIA_DIR)" ]; then \
		$(JULIA) $(JULIA_ARGS) -e 'using Pkg; Pkg.test()'; \
	else \
		echo -e "$(YELLOW)Julia module not found$(NC)"; \
	fi

.PHONY: julia-bench
julia-bench: ## Run Julia benchmarks
	@echo -e "$(BLUE)Running Julia benchmarks...$(NC)"
	@if [ -d "$(JULIA_DIR)" ] && [ -f "$(JULIA_DIR)/benchmark/benchmarks.jl" ]; then \
		$(JULIA) $(JULIA_ARGS) $(JULIA_DIR)/benchmark/benchmarks.jl; \
	else \
		echo -e "$(YELLOW)Julia benchmarks not found$(NC)"; \
	fi

.PHONY: julia-precompile
julia-precompile: ## Precompile Julia packages
	@echo -e "$(BLUE)Precompiling Julia packages...$(NC)"
	@if [ -d "$(JULIA_DIR)" ]; then \
		$(JULIA) $(JULIA_ARGS) -e 'using Pkg; Pkg.precompile()'; \
	fi

.PHONY: julia-update
julia-update: ## Update Julia dependencies
	@echo -e "$(BLUE)Updating Julia dependencies...$(NC)"
	@if [ -d "$(JULIA_DIR)" ]; then \
		$(JULIA) $(JULIA_ARGS) -e 'using Pkg; Pkg.update()'; \
	fi

.PHONY: julia-repl
julia-repl: ## Start Julia REPL with project environment
	@echo -e "$(BLUE)Starting Julia REPL...$(NC)"
	$(JULIA) $(JULIA_ARGS)

# ==============================================================================
# Python Targets
# ==============================================================================

.PHONY: python-test
python-test: ## Run Python tests
	@echo -e "$(BLUE)Running Python tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) $(PYTEST_ARGS)

.PHONY: python-test-cov
python-test-cov: ## Run Python tests with coverage
	@echo -e "$(BLUE)Running Python tests with coverage...$(NC)"
	$(PYTEST) $(TESTS_DIR) $(PYTEST_COV_ARGS) $(PYTEST_ARGS)

.PHONY: python-test-fast
python-test-fast: ## Run Python tests (fast mode, skip slow tests)
	@echo -e "$(BLUE)Running fast Python tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) -m "not slow" $(PYTEST_ARGS)

.PHONY: benchmark
benchmark: ## Run benchmark tests
	@echo -e "$(BLUE)Running benchmark tests...$(NC)"
	$(PYTEST) $(BENCHMARKS_DIR) $(PYTEST_ARGS) --benchmark-only

.PHONY: benchmark-save
benchmark-save: ## Run benchmarks and save results
	@echo -e "$(BLUE)Running benchmarks with JSON output...$(NC)"
	$(PYTEST) $(BENCHMARKS_DIR) $(PYTEST_ARGS) --benchmark-json=benchmark_results.json

.PHONY: property-test
property-test: ## Run property-based tests
	@echo -e "$(BLUE)Running property-based tests...$(NC)"
	$(PYTEST) $(PROPERTY_TESTS_DIR) $(PYTEST_ARGS)

.PHONY: golden-test
golden-test: ## Run golden file tests
	@echo -e "$(BLUE)Running golden file tests...$(NC)"
	$(PYTEST) $(GOLDEN_TESTS_DIR) $(PYTEST_ARGS)

.PHONY: golden-regenerate
golden-regenerate: ## Regenerate golden files
	@echo -e "$(YELLOW)Regenerating golden files...$(NC)"
	$(PYTEST) $(GOLDEN_TESTS_DIR) --regenerate-golden-files $(PYTEST_ARGS)

# ==============================================================================
# Code Quality Targets
# ==============================================================================

.PHONY: lint
lint: ## Run all linters (black, isort, flake8, mypy)
	@echo -e "$(BLUE)Running linters...$(NC)"
	$(PYTHON) -m black --check $(PYTHON_SRC) $(TESTS_DIR)
	$(PYTHON) -m isort --check-only $(PYTHON_SRC) $(TESTS_DIR)
	$(PYTHON) -m flake8 $(PYTHON_SRC) $(TESTS_DIR)
	$(PYTHON) -m mypy $(PYTHON_SRC)

.PHONY: format
format: ## Format Python code (black + isort)
	@echo -e "$(BLUE)Formatting Python code...$(NC)"
	$(PYTHON) -m black $(PYTHON_SRC) $(TESTS_DIR)
	$(PYTHON) -m isort $(PYTHON_SRC) $(TESTS_DIR)

.PHONY: mypy
mypy: ## Run mypy type checker
	@echo -e "$(BLUE)Running mypy...$(NC)"
	$(PYTHON) -m mypy $(PYTHON_SRC)

.PHONY: mypy-strict
mypy-strict: ## Run mypy in strict mode on migrations candidates
	@echo -e "$(BLUE)Running mypy (strict mode)...$(NC)"
	$(PYTHON) -m mypy $(PYTHON_SRC)/backtest_engine --strict

.PHONY: security
security: ## Run security checks (bandit, pip-audit)
	@echo -e "$(BLUE)Running security checks...$(NC)"
	$(PYTHON) -m bandit -r $(PYTHON_SRC) -ll
	$(PYTHON) -m pip_audit

.PHONY: pre-commit
pre-commit: ## Run pre-commit on all files
	@echo -e "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run -a

# ==============================================================================
# Combined Targets
# ==============================================================================

.PHONY: all
all: install-dev rust-build julia-instantiate test ## Build and test everything
	@echo -e "$(GREEN)All builds and tests completed!$(NC)"

.PHONY: test
test: python-test rust-test julia-test ## Run all tests (Python, Rust, Julia)
	@echo -e "$(GREEN)All tests completed!$(NC)"

.PHONY: build
build: rust-build ## Build all native extensions
	@echo -e "$(GREEN)Build completed!$(NC)"

.PHONY: ci
ci: lint rust-clippy rust-fmt-check python-test rust-test ## Run CI checks
	@echo -e "$(GREEN)CI checks completed!$(NC)"

.PHONY: full-check
full-check: lint security mypy python-test-cov property-test golden-test rust-clippy rust-test ## Run complete verification
	@echo -e "$(GREEN)Full verification completed!$(NC)"

# ==============================================================================
# Development Workflow Targets
# ==============================================================================

.PHONY: dev-setup
dev-setup: install-dev install-pre-commit rust-build julia-instantiate ## Complete development setup
	@echo -e "$(GREEN)Development environment ready!$(NC)"

.PHONY: watch
watch: ## Run tests on file changes (requires pytest-watch)
	@echo -e "$(BLUE)Starting test watcher...$(NC)"
	$(PYTHON) -m pytest_watch -- $(TESTS_DIR) $(PYTEST_ARGS)

# ==============================================================================
# Documentation Targets
# ==============================================================================

.PHONY: docs
docs: rust-doc ## Generate all documentation
	@echo -e "$(GREEN)Documentation generated!$(NC)"

# ==============================================================================
# Cleanup Targets
# ==============================================================================

.PHONY: clean
clean: rust-clean ## Clean all build artifacts
	@echo -e "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ 2>/dev/null || true
	@echo -e "$(GREEN)Clean completed!$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean everything including venv
	@echo -e "$(YELLOW)Warning: This will remove .venv directory$(NC)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf .venv || true

# ==============================================================================
# Utility Targets
# ==============================================================================

.PHONY: version
version: ## Show versions of all tools
	@echo -e "$(BLUE)Tool Versions:$(NC)"
	@echo -n "Python: " && $(PYTHON) --version
	@echo -n "pip: " && $(PIP) --version | cut -d' ' -f2
	@echo -n "Rust: " && rustc --version 2>/dev/null || echo "not installed"
	@echo -n "Cargo: " && cargo --version 2>/dev/null || echo "not installed"
	@echo -n "Maturin: " && $(MATURIN) --version 2>/dev/null || echo "not installed"
	@echo -n "Julia: " && $(JULIA) --version 2>/dev/null || echo "not installed"

.PHONY: check-deps
check-deps: ## Check if all required tools are installed
	@echo -e "$(BLUE)Checking dependencies...$(NC)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo -e "$(RED)Python not found$(NC)"; exit 1; }
	@command -v cargo >/dev/null 2>&1 || { echo -e "$(YELLOW)Cargo not found (Rust builds will be skipped)$(NC)"; }
	@command -v $(JULIA) >/dev/null 2>&1 || { echo -e "$(YELLOW)Julia not found (Julia tests will be skipped)$(NC)"; }
	@echo -e "$(GREEN)Dependency check passed!$(NC)"

# Prevent make from interpreting targets as files
.PHONY: all test build clean help
