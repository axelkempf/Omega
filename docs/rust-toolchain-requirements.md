# Rust Toolchain Requirements

**Task-ID:** P4-01  
**Status:** ✅ Completed (2026-01-05)  
**Phase:** 4 – Build-System

---

## Executive Summary

Dieses Dokument spezifiziert die minimalen Rust-Toolchain-Anforderungen für die Hybrid-Architektur des Omega Trading-Systems. Die Rust-Integration erfolgt über PyO3/Maturin für nahtlose Python-FFI-Bindings mit Fokus auf Performance-kritische numerische Module.

---

## Minimum Requirements

### Rust Version

**Minimum:** `1.75.0` (released 2023-12-28)  
**Recommended:** `1.76+` (stable channel)

**Begründung:**
- PyO3 0.20+ erfordert mindestens Rust 1.63
- Maturin 1.4+ funktioniert optimal mit Rust 1.70+
- Moderne async-Traits und GATs (Generic Associated Types) für bessere Typ-Sicherheit
- Verbesserte Error-Messages und Compile-Zeiten

**Version Pinning-Strategie:**
- Development: `stable` channel (rolling updates)
- CI/CD: Pinned auf spezifische Minor-Version (e.g., `1.76.0`)
- Production Builds: Locked Toolchain via `rust-toolchain.toml`

### Required Toolchain Components

```toml
[toolchain]
channel = "1.76.0"
components = ["rustfmt", "clippy"]
targets = ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin", "x86_64-pc-windows-msvc", "aarch64-apple-darwin"]
profile = "minimal"
```

### Platform-Specific Requirements

| Platform | Target Triple | Notes |
|----------|---------------|-------|
| **Linux (x86_64)** | `x86_64-unknown-linux-gnu` | Primary development platform; glibc 2.31+ |
| **macOS (Intel)** | `x86_64-apple-darwin` | macOS 11.0+ (Big Sur) |
| **macOS (Apple Silicon)** | `aarch64-apple-darwin` | Native ARM64 support; macOS 11.0+ |
| **Windows** | `x86_64-pc-windows-msvc` | MSVC 2019+; Required for MT5 integration |

---

## Core Dependencies

### Essential Crates

#### 1. PyO3 (Python FFI)

```toml
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py310"] }
```

**Features:**
- `extension-module`: Enables Python extension module support
- `abi3-py310`: Stable ABI for Python 3.10+ (forward compatibility)
- **Benefit:** Single binary works across Python 3.10, 3.11, 3.12+

**Key Capabilities:**
- Zero-copy data transfer via buffer protocol
- Python exception mapping
- GIL (Global Interpreter Lock) handling
- Async/await support (PyO3-asyncio)

#### 2. ndarray (Numerical Arrays)

```toml
[dependencies]
ndarray = { version = "0.15", features = ["serde", "rayon"] }
```

**Features:**
- `serde`: Serialization support
- `rayon`: Parallel iterators for multi-threading

**Use Cases:**
- OHLCV data manipulation
- Indicator calculations
- Matrix operations for scoring algorithms

#### 3. Arrow (Zero-Copy FFI)

```toml
[dependencies]
arrow = { version = "50.0", features = ["ffi"] }
arrow-array = "50.0"
arrow-schema = "50.0"
```

**Features:**
- `ffi`: C Data Interface for zero-copy Python ↔ Rust
- Columnar memory layout (cache-friendly)
- Schema validation

**Integration:**
- Python side: `pyarrow.Table` → Arrow C FFI → Rust `RecordBatch`
- No serialization overhead for large datasets

#### 4. Error Handling

```toml
[dependencies]
anyhow = "1.0"          # Context-rich errors
thiserror = "1.0"       # Custom error types
```

**Pattern:**
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OmegaError {
    #[error("Invalid OHLCV data: {reason}")]
    InvalidData { reason: String },
    
    #[error("FFI serialization failed: {0}")]
    SerializationError(#[from] arrow::error::ArrowError),
    
    #[error("Python exception: {0}")]
    PythonError(#[from] pyo3::PyErr),
}

pub type Result<T> = std::result::Result<T, OmegaError>;
```

#### 5. Serialization

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"      # JSON configs
rmp-serde = "1.1"       # msgpack fallback
```

#### 6. Performance & Profiling

```toml
[dependencies]
rayon = "1.8"           # Data parallelism
mimalloc = "0.1"        # Fast allocator

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

---

## Build Tool: Maturin

### Installation

```bash
# Via pip (recommended for Python developers)
pip install maturin

# Via cargo (for Rust developers)
cargo install maturin
```

**Version:** `maturin >= 1.4.0`

### Maturin Configuration

Create `pyproject.toml` in Rust module root:

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "omega-rust"
version = "0.1.0"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[tool.maturin]
python-source = "python"
module-name = "omega._rust"
```

### Build Commands

```bash
# Development build (debug mode)
maturin develop

# Release build
maturin build --release

# Build wheels for all platforms
maturin build --release --out dist --universal2
```

---

## Cargo.toml Template

```toml
[package]
name = "omega-rust"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[lib]
name = "omega_rust"
crate-type = ["cdylib"]

[dependencies]
# Python FFI
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py310"] }

# Numerical Computing
ndarray = { version = "0.15", features = ["serde", "rayon"] }
num-traits = "0.2"

# Arrow Zero-Copy
arrow = { version = "50.0", features = ["ffi"] }
arrow-array = "50.0"
arrow-schema = "50.0"

# Error Handling
anyhow = "1.0"
thiserror = "1.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rmp-serde = "1.1"

# Performance
rayon = "1.8"
mimalloc = { version = "0.1", default-features = false }

[dev-dependencies]
criterion = "0.5"
proptest = "1.4"
approx = "0.5"

[profile.dev]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
panic = "abort"
strip = true

[profile.release-with-debug]
inherits = "release"
debug = true
strip = false

[[bench]]
name = "indicator_bench"
harness = false
```

---

## Development Workflow

### 1. Initialize Rust Module

```bash
# In project root
mkdir -p src/rust_modules/omega_rust
cd src/rust_modules/omega_rust

# Initialize Cargo project
cargo init --lib

# Setup maturin
maturin init --bindings pyo3
```

### 2. Project Structure

```
src/rust_modules/omega_rust/
├── Cargo.toml
├── pyproject.toml
├── rust-toolchain.toml
├── src/
│   ├── lib.rs              # PyO3 entry point
│   ├── indicators/
│   │   ├── mod.rs
│   │   ├── ema.rs
│   │   └── rsi.rs
│   ├── event_engine/
│   │   ├── mod.rs
│   │   └── simulator.rs
│   └── ffi/
│       ├── mod.rs
│       ├── arrow_bridge.rs
│       └── error.rs
├── benches/
│   └── indicator_bench.rs
└── tests/
    └── integration_test.rs
```

### 3. Example Module (`src/rust_modules/omega_rust/src/lib.rs`)

```rust
use pyo3::prelude::*;

/// Calculate Exponential Moving Average
#[pyfunction]
fn ema(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    if period == 0 || period > prices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid period"
        ));
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(prices.len());
    let mut ema_val = prices[0];
    
    for &price in &prices {
        ema_val = alpha * price + (1.0 - alpha) * ema_val;
        result.push(ema_val);
    }
    
    Ok(result)
}

#[pymodule]
fn omega_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    Ok(())
}
```

### 4. Build & Test

```bash
# Install in editable mode
maturin develop

# Run Python tests
pytest tests/

# Run Rust tests
cargo test

# Run benchmarks
cargo bench
```

---

## CI/CD Integration

### GitHub Actions Workflow Snippet

```yaml
name: Rust Build

on: [push, pull_request]

jobs:
  build-rust:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.76.0
          components: rustfmt, clippy
      
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install maturin
        run: pip install maturin
      
      - name: Build wheels
        run: maturin build --release
      
      - name: Run tests
        run: |
          maturin develop
          pytest tests/
```

---

## Cross-Compilation

### Linux → Windows (via MinGW)

```bash
# Install target
rustup target add x86_64-pc-windows-gnu

# Build
cargo build --release --target x86_64-pc-windows-gnu
```

### macOS Universal Binary (Intel + Apple Silicon)

```bash
# Install targets
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin

# Build universal binary
maturin build --release --universal2
```

---

## Performance Optimization

### Compiler Flags

```toml
[profile.release]
opt-level = 3               # Maximum optimization
lto = "thin"                # Link-Time Optimization
codegen-units = 1           # Better optimization, slower compile
panic = "abort"             # Smaller binary, no unwinding
strip = true                # Remove debug symbols
```

### CPU-Specific Optimizations

```bash
# Enable native CPU features (non-portable)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# For portable binaries, use baseline
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release
```

### Memory Allocator

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Benefit:** 10-20% performance improvement vs. system allocator

---

## Security Considerations

### Dependency Auditing

```bash
# Install cargo-audit
cargo install cargo-audit

# Run security audit
cargo audit

# Check for outdated dependencies
cargo outdated
```

### Minimal Attack Surface

- Use `#![forbid(unsafe_code)]` unless absolutely necessary
- Enable Clippy lints in `Cargo.toml`:

```toml
[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
all = "warn"
pedantic = "warn"
nursery = "warn"
```

---

## Troubleshooting

### Common Issues

#### 1. "cannot find -lpython3.12" on Linux

**Solution:**
```bash
sudo apt-get install python3.12-dev
```

#### 2. Maturin build fails on Windows

**Solution:** Install Visual Studio Build Tools 2019+
```powershell
# Check MSVC installation
rustup toolchain list
```

#### 3. ImportError: DLL load failed on Windows

**Solution:** Ensure correct Python ABI match
```bash
maturin build --release --interpreter python
```

---

## Next Steps (Phase 4 Continuation)

- **P4-03:** GitHub Actions Workflow für Rust-Kompilierung ✅ (Template in diesem Dokument)
- **P4-06:** PyO3/Maturin Integration Template (siehe Example Module oben)
- **P4-11:** Cache-Strategie für Cargo in CI

---

## References

- [PyO3 User Guide](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Arrow Rust Implementation](https://docs.rs/arrow/)
- ADR-0002: Serialization Format (`docs/adr/ADR-0002-serialization-format.md`)
- ADR-0003: Error Handling (`docs/adr/ADR-0003-error-handling.md`)

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-05  
**Maintainer:** Axel Kempf
