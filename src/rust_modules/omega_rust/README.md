# Omega Rust Extensions

High-performance Rust implementations for the Omega Trading System.

## Overview

This crate provides Python bindings via PyO3/Maturin for performance-critical numerical algorithms used in the trading system.

### Available Functions

| Function | Description | Typical Speedup |
|----------|-------------|-----------------|
| `ema(prices, period)` | Exponential Moving Average | 10-50x |
| `rsi(prices, period)` | Relative Strength Index | 10-30x |
| `rolling_std(values, window)` | Rolling Standard Deviation | 20-50x |

## Installation

### From Source (Development)

```bash
# Ensure Rust toolchain is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install in development mode
cd src/rust_modules/omega_rust
maturin develop --release
```

### From Wheel (Production)

```bash
# Build release wheel
maturin build --release

# Install wheel
pip install target/wheels/omega_rust-*.whl
```

## Usage

```python
from omega._rust import ema, rsi, rolling_std

# Calculate EMA
prices = [100.0, 101.5, 99.8, 102.3, 103.1]
ema_values = ema(prices, period=3)

# Calculate RSI
rsi_values = rsi(prices, period=14)

# Calculate rolling volatility
returns = [0.01, -0.02, 0.015, -0.005, 0.02]
volatility = rolling_std(returns, window=20)
```

## Development

### Run Tests

```bash
# Rust unit tests
cargo test

# Python integration tests (after maturin develop)
pytest tests/ -m rust_integration
```

### Run Benchmarks

```bash
cargo bench
```

### Code Quality

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-targets --all-features

# Security audit
cargo audit
```

## Architecture

```
src/
├── lib.rs              # PyO3 module entry point
├── error.rs            # Error types
└── indicators/
    ├── mod.rs          # Module exports
    ├── ema.rs          # EMA implementation
    ├── rsi.rs          # RSI implementation
    └── statistics.rs   # Statistical functions
```

## Performance Notes

- All functions are O(n) time complexity
- Minimal memory allocation (pre-allocated vectors)
- SIMD-friendly loops (auto-vectorized by LLVM)
- No Python GIL held during computation

## License

MIT
