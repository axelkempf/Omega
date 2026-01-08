# Julia Environment Requirements

**Task-ID:** P4-02  
**Status:** ✅ Completed (2026-01-05)  
**Phase:** 4 – Build-System

---

## Executive Summary

Dieses Dokument spezifiziert die minimalen Julia-Environment-Anforderungen für die Hybrid-Architektur des Omega Trading-Systems. Die Julia-Integration erfolgt über PythonCall.jl für bidirektionale Python ↔ Julia FFI mit Fokus auf Research-Pipelines, Monte-Carlo-Simulationen und explorative Datenanalyse.

---

## Minimum Requirements

### Julia Version

**Minimum:** `1.9.0` (LTS)  
**Recommended:** `1.10.0+` (stable)

**Begründung:**
- Julia 1.9 ist die aktuelle LTS (Long-Term Support) Version
- Julia 1.10 bietet verbesserte Package-Load-Zeiten und modernere stdlib
- PythonCall.jl erfordert mindestens Julia 1.6
- Moderne Threading-APIs und bessere Type-Inference

**Version Pinning-Strategie:**
- Development: `1.10+` (latest stable)
- CI/CD: Pinned auf `1.10.0` für Reproduzierbarkeit
- Production: Locked Julia Version via Docker/Container

### Installation

#### macOS (Homebrew)

```bash
brew install julia
```

#### Linux (juliaup - Recommended)

```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.10
juliaup default 1.10
```

#### Windows (juliaup)

```powershell
winget install julia -s msstore
juliaup add 1.10
juliaup default 1.10
```

#### Verify Installation

```bash
julia --version
# Expected output: julia version 1.10.x
```

---

## Core Dependencies

### Essential Packages

#### 1. PythonCall.jl (Python FFI)

**Version:** `0.9+`

**Installation:**
```julia
using Pkg
Pkg.add("PythonCall")
```

**Key Features:**
- Bidirectional calling: Python ↔ Julia
- Zero-copy data transfer for NumPy arrays
- Automatic type conversion
- GIL-aware (better than PyJulia)
- Conda environment management

**Python Side Setup:**
```bash
pip install juliacall
```

**Usage Example:**
```julia
using PythonCall

# Import Python modules
np = pyimport("numpy")
pd = pyimport("pandas")

# Call Python from Julia
py_array = np.array([1, 2, 3])
println(pytopy(py_array))  # Convert to Julia Array
```

**Python → Julia:**
```python
from juliacall import Main as jl

# Call Julia functions
jl.seval("using Statistics")
result = jl.mean([1, 2, 3, 4, 5])
print(result)  # 3.0
```

#### 2. Arrow.jl (Zero-Copy FFI)

**Version:** `2.7+`

**Installation:**
```julia
Pkg.add("Arrow")
```

**Use Cases:**
- Zero-copy data transfer with Python via Arrow IPC
- Columnar memory layout (cache-friendly)
- Schema validation
- Parquet file I/O

**Integration with PythonCall:**
```julia
using Arrow
using PythonCall

# Read Arrow Table from Python
pyarrow = pyimport("pyarrow")
py_table = pyarrow.table(...)

# Convert to Julia Arrow.Table (zero-copy)
jl_table = Arrow.Table(pytopy(py_table))
```

#### 3. DataFrames.jl (Data Manipulation)

**Version:** `1.6+`

**Installation:**
```julia
Pkg.add("DataFrames")
```

**Features:**
- High-performance tabular data
- Similar API to pandas
- Query language via DataFramesMeta.jl
- Integration with Arrow.jl

**Example:**
```julia
using DataFrames

df = DataFrame(
    timestamp = 1:100,
    close = rand(100),
    volume = rand(100) .* 1000
)

# Filter and transform
filtered = filter(row -> row.close > 0.5, df)
```

#### 4. Statistics & StatsBase.jl

**Built-in:** `Statistics` (stdlib)  
**Version:** `StatsBase 0.34+`

**Installation:**
```julia
# Statistics is built-in
using Statistics

# StatsBase for advanced stats
Pkg.add("StatsBase")
```

**Use Cases:**
- Basic statistics: mean, std, quantile
- Rolling windows
- Correlation, covariance
- Histogram utilities

#### 5. Distributions.jl (Monte-Carlo)

**Version:** `0.25+`

**Installation:**
```julia
Pkg.add("Distributions")
```

**Use Cases:**
- Random number generation for simulations
- Probability distributions (Normal, LogNormal, Student-T)
- Monte-Carlo path generation

**Example:**
```julia
using Distributions

# Generate GBM paths
μ = 0.05  # drift
σ = 0.20  # volatility
dt = 1/252  # daily timestep

dist = Normal(μ * dt, σ * sqrt(dt))
returns = rand(dist, 10000)
prices = cumprod(1 .+ returns)
```

#### 6. CSV.jl (File I/O)

**Version:** `0.10+`

**Installation:**
```julia
Pkg.add("CSV")
```

**Features:**
- Fast CSV parsing
- Type inference
- Stream processing for large files
- Integration with DataFrames.jl

---

## Optional Packages (Research)

### Plotting & Visualization

```julia
Pkg.add("Plots")         # High-level plotting API
Pkg.add("StatsPlots")    # Statistical plots
```

### Time Series Analysis

```julia
Pkg.add("TimeSeries")    # Time series data structures
Pkg.add("MarketData")    # Sample financial data
```

### Machine Learning

```julia
Pkg.add("MLJ")           # Machine Learning framework
Pkg.add("Flux")          # Deep learning
```

### Optimization

```julia
Pkg.add("Optim")         # Numerical optimization
Pkg.add("JuMP")          # Mathematical programming
```

---

## Project.toml Template

```toml
name = "OmegaJulia"
uuid = "12345678-1234-1234-1234-123456789abc"
version = "0.1.0"

[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Arrow = "^2.7"
CSV = "^0.10"
DataFrames = "^1.6"
Distributions = "^0.25"
PythonCall = "^0.9"
StatsBase = "^0.34"
julia = "^1.9"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

---

## Manifest.toml (Lock File)

Julia's `Manifest.toml` is equivalent to Python's `requirements.txt` (pinned versions).

**Best Practice:**
- Commit `Project.toml` (specifies dependencies)
- Commit `Manifest.toml` for reproducibility (locks exact versions)
- For library packages: Do NOT commit `Manifest.toml` (allow version flexibility)
- For applications: COMMIT `Manifest.toml` (ensure exact reproduction)

**Generate Manifest:**
```julia
using Pkg
Pkg.instantiate()  # Creates Manifest.toml
```

---

## Development Workflow

### 1. Initialize Julia Project

```bash
# In project root
mkdir -p src/julia_modules/omega_julia
cd src/julia_modules/omega_julia

# Initialize Julia project
julia --project=. -e 'using Pkg; Pkg.generate("OmegaJulia")'
```

### 2. Project Structure

```
src/julia_modules/omega_julia/
├── Project.toml
├── Manifest.toml
├── src/
│   ├── OmegaJulia.jl       # Main module
│   ├── monte_carlo.jl
│   ├── analysis.jl
│   └── ffi_bridge.jl
├── test/
│   └── runtests.jl
└── examples/
    └── demo.jl
```

### 3. Example Module (`src/julia_modules/omega_julia/src/OmegaJulia.jl`)

```julia
module OmegaJulia

using Statistics
using StatsBase
using Distributions

export monte_carlo_var, rolling_sharpe

"""
Calculate Value-at-Risk using Monte-Carlo simulation.

# Arguments
- `returns::Vector{Float64}`: Historical returns
- `n_sims::Int`: Number of MC simulations
- `confidence::Float64`: Confidence level (default 0.95)

# Returns
- `Float64`: VaR estimate
"""
function monte_carlo_var(
    returns::Vector{Float64},
    n_sims::Int=10_000;
    confidence::Float64=0.95
)
    μ = mean(returns)
    σ = std(returns)
    
    # Bootstrap simulation
    dist = Normal(μ, σ)
    simulated_returns = rand(dist, n_sims)
    
    # Calculate VaR
    var_level = quantile(simulated_returns, 1 - confidence)
    return var_level
end

"""
Calculate rolling Sharpe ratio.

# Arguments
- `returns::Vector{Float64}`: Returns series
- `window::Int`: Rolling window size
- `risk_free_rate::Float64`: Annualized risk-free rate (default 0.02)

# Returns
- `Vector{Float64}`: Rolling Sharpe ratios
"""
function rolling_sharpe(
    returns::Vector{Float64},
    window::Int;
    risk_free_rate::Float64=0.02
)
    n = length(returns)
    sharpe = Vector{Float64}(undef, n - window + 1)
    
    daily_rf = risk_free_rate / 252
    
    for i in 1:(n - window + 1)
        window_returns = @view returns[i:(i + window - 1)]
        excess_returns = window_returns .- daily_rf
        
        μ = mean(excess_returns)
        σ = std(excess_returns)
        
        sharpe[i] = (μ / σ) * sqrt(252)  # Annualized
    end
    
    return sharpe
end

end  # module
```

### 4. Test Suite (`test/runtests.jl`)

```julia
using Test
using OmegaJulia

@testset "Monte Carlo VaR" begin
    # Deterministic test with fixed seed
    using Random
    Random.seed!(42)
    
    returns = randn(1000) .* 0.01
    var = monte_carlo_var(returns, 10_000)
    
    @test var < 0  # VaR should be negative
    @test abs(var) < 0.05  # Reasonable magnitude
end

@testset "Rolling Sharpe" begin
    returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02]
    window = 3
    
    sharpe = rolling_sharpe(returns, window)
    
    @test length(sharpe) == length(returns) - window + 1
    @test all(isfinite.(sharpe))
end
```

### 5. Run Tests

```bash
# Run all tests
julia --project=. test/runtests.jl

# Or using Pkg
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Python ↔ Julia Integration

### Calling Julia from Python

```python
from juliacall import Main as jl

# Load Julia module
jl.seval('using Pkg; Pkg.activate("src/julia_modules/omega_julia")')
jl.seval('using OmegaJulia')

# Call Julia function
import numpy as np
returns = np.random.randn(1000) * 0.01
var = jl.monte_carlo_var(returns.tolist(), 10_000)
print(f"VaR (95%): {var:.4f}")
```

### Calling Python from Julia

```julia
using PythonCall

# Import Python modules
pd = pyimport("pandas")
np = pyimport("numpy")

# Load data from Python
py_df = pd.read_csv("data/prices.csv")
prices = pytopy(py_df["close"].values)  # Convert to Julia Array

# Process in Julia
returns = diff(log.(prices))
var = monte_carlo_var(returns, 10_000)
```

### Arrow Bridge (Zero-Copy)

```julia
# Julia side: Export to Arrow
using Arrow

data = DataFrame(
    timestamp = 1:1000,
    returns = randn(1000) .* 0.01
)

Arrow.write("data.arrow", data)
```

```python
# Python side: Import from Arrow
import pyarrow as pa
import pyarrow.feather as feather

table = feather.read_table("data.arrow")
df = table.to_pandas()
```

---

## CI/CD Integration

### GitHub Actions Workflow Snippet

```yaml
name: Julia Tests

on: [push, pull_request]

jobs:
  test-julia:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        julia-version: ["1.9", "1.10"]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      
      - uses: julia-actions/cache@v1
      
      - name: Install dependencies
        run: |
          julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.instantiate()'
      
      - name: Run tests
        run: |
          julia --project=src/julia_modules/omega_julia test/runtests.jl
      
      - uses: julia-actions/julia-processcoverage@v1
      
      - uses: codecov/codecov-action@v3
```

---

## Performance Optimization

### Pre-compilation

Julia compiles functions on first call (JIT). To reduce latency:

```julia
# In OmegaJulia.jl
using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Trigger compilation of hot functions
        returns = randn(1000) .* 0.01
        monte_carlo_var(returns, 1000)
        rolling_sharpe(returns, 20)
    end
end
```

### Parallelization

```julia
# Enable multi-threading
export JULIA_NUM_THREADS=8

# Parallel Monte-Carlo
using Base.Threads

function parallel_monte_carlo(returns, n_sims)
    results = Vector{Float64}(undef, nthreads())
    
    @threads for i in 1:nthreads()
        chunk_size = div(n_sims, nthreads())
        results[i] = monte_carlo_var(returns, chunk_size)
    end
    
    return mean(results)
end
```

### Type Stability

```julia
# BAD: Type-unstable (performance hit)
function bad_func(x)
    if x > 0
        return x
    else
        return "negative"  # Type changes!
    end
end

# GOOD: Type-stable
function good_func(x::Float64)::Float64
    return max(x, 0.0)
end
```

---

## Package Management

### Add Dependencies

```julia
# Interactive mode
julia> ]  # Enter Pkg mode
pkg> add DataFrames

# Programmatic
using Pkg
Pkg.add("DataFrames")
```

### Update Dependencies

```julia
pkg> update
# Or
Pkg.update()
```

### Pinning Versions

```julia
pkg> add DataFrames@1.6.0
```

---

## Environment Variables

### Python Integration

```bash
# Specify Python executable for PythonCall
export JULIA_CONDAPKG_BACKEND="Null"
export JULIA_PYTHONCALL_EXE="/path/to/python"

# Or in Julia
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/python"
```

### Threading

```bash
# Set number of threads
export JULIA_NUM_THREADS=8

# Verify
julia -e 'using Base.Threads; println(nthreads())'
```

---

## Troubleshooting

### Common Issues

#### 1. PythonCall.jl cannot find Python

**Solution:**
```julia
using PythonCall
PythonCall.C.CTX.exe_path  # Check detected Python
```

Set explicit path:
```bash
export JULIA_PYTHONCALL_EXE="/usr/bin/python3.12"
```

#### 2. "Package X not found" after adding dependency

**Solution:**
```julia
using Pkg
Pkg.resolve()
Pkg.instantiate()
```

#### 3. Precompilation fails on macOS

**Solution:** Increase stack size:
```bash
ulimit -s unlimited
```

#### 4. Julia crashes on Windows with PythonCall

**Solution:** Use matching architecture (x64 Julia + x64 Python)
```powershell
julia -e 'println(Sys.WORD_SIZE)'  # Should match Python (64)
```

---

## Next Steps (Phase 4 Continuation)

- **P4-04:** GitHub Actions Workflow für Julia-Paket-Installation ✅ (Template in diesem Dokument)
- **P4-07:** PyJulia/PythonCall Integration Template (siehe Example Module oben)
- **P4-11:** Cache-Strategie für Julia Depot in CI

---

## References

- [Julia Documentation](https://docs.julialang.org/)
- [PythonCall.jl Documentation](https://cjdoris.github.io/PythonCall.jl/)
- [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [DataFrames.jl](https://dataframes.juliadata.org/)
- [Arrow.jl](https://github.com/apache/arrow-julia)
- ADR-0002: Serialization Format (`docs/adr/ADR-0002-serialization-format.md`)

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-05  
**Maintainer:** Axel Kempf
