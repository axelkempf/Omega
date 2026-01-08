# OmegaJulia

High-performance Julia implementations for the Omega Trading System.

## Overview

OmegaJulia provides optimized implementations for computationally intensive algorithms, particularly Monte Carlo simulations and statistical analysis.

### Available Functions

| Category | Function | Description |
|----------|----------|-------------|
| **Monte Carlo** | `monte_carlo_var` | Value at Risk simulation |
| | `monte_carlo_var_detailed` | VaR with CVaR and statistics |
| | `monte_carlo_portfolio_var` | Correlated portfolio VaR |
| **Rolling Stats** | `rolling_sharpe` | Rolling Sharpe ratio |
| | `rolling_sortino` | Rolling Sortino ratio |
| | `rolling_calmar` | Rolling Calmar ratio |
| **Bootstrap** | `block_bootstrap` | Block resampling |
| | `stationary_bootstrap` | Stationary bootstrap |
| **Risk Metrics** | `sharpe_ratio` | Sharpe ratio |
| | `sortino_ratio` | Sortino ratio |
| | `max_drawdown` | Maximum drawdown |
| | `calmar_ratio` | Calmar ratio |
| | `omega_ratio` | Omega ratio |

## Installation

### Development Setup

```julia
# From the package directory
julia> ]
pkg> activate .
pkg> instantiate
```

### Usage from Python

```python
from juliacall import Main as jl

# Initialize Julia package
jl.seval('push!(LOAD_PATH, "src/julia_modules/omega_julia/src")')
jl.seval("using OmegaJulia")

# Calculate Monte Carlo VaR
returns = [0.01, -0.02, 0.015, -0.005, 0.02, ...]  # Python list
var_95 = jl.monte_carlo_var(returns, 10_000, 0.95)
print(f"95% VaR: {var_95:.4f}")

# Rolling Sharpe ratio
sharpe = jl.rolling_sharpe(returns, 63)
```

### Usage from Julia

```julia
using OmegaJulia

# Monte Carlo VaR
returns = rand(Normal(0.001, 0.02), 252)
var = monte_carlo_var(returns, 10_000, 0.95; seed=42)

# Detailed results
result = monte_carlo_var_detailed(returns, 10_000, 0.95)
println("VaR: $(result.var), CVaR: $(result.cvar)")

# Rolling statistics
sharpe = rolling_sharpe(returns, 63; annualization=252)
sortino = rolling_sortino(returns, 63)
```

## Development

### Run Tests

```bash
cd src/julia_modules/omega_julia
julia --project=. -e "using Pkg; Pkg.test()"
```

### Run Benchmarks

```julia
using BenchmarkTools
using OmegaJulia

returns = rand(10_000)
@btime monte_carlo_var($returns, 100_000, 0.95)
@btime rolling_sharpe($returns, 252)
```

## Performance

Typical speedups compared to Python implementations:

| Function | Python (pandas/numpy) | Julia | Speedup |
|----------|----------------------|-------|---------|
| Monte Carlo VaR (100k sims) | 450ms | 12ms | ~37x |
| Rolling Sharpe (10k points) | 85ms | 2ms | ~42x |
| Block Bootstrap (1k samples) | 320ms | 8ms | ~40x |

## Architecture

```
src/
├── OmegaJulia.jl       # Main module
├── monte_carlo.jl      # Monte Carlo simulations
├── rolling_stats.jl    # Rolling window statistics
├── bootstrap.jl        # Bootstrap methods
└── risk_metrics.jl     # Risk/performance metrics

test/
└── runtests.jl         # Test suite
```

## API Reference

### Monte Carlo

```julia
monte_carlo_var(returns, n_simulations, confidence; seed=nothing)
monte_carlo_var_detailed(returns, n_simulations, confidence; seed=nothing)
monte_carlo_portfolio_var(returns, weights, n_simulations, confidence; seed=nothing)
```

### Rolling Statistics

```julia
rolling_sharpe(returns, window; risk_free_rate=0.0, annualization=252)
rolling_sortino(returns, window; target=0.0, annualization=252)
rolling_calmar(returns, window; annualization=252)
rolling_volatility(returns, window; annualization=252)
```

### Bootstrap

```julia
block_bootstrap(data, block_size, n_samples; seed=nothing)
stationary_bootstrap(data, mean_block_size, n_samples; seed=nothing)
bootstrap_confidence_interval(data, statistic, block_size, n_samples, confidence; seed=nothing)
```

### Risk Metrics

```julia
sharpe_ratio(returns; risk_free_rate=0.0, annualization=252)
sortino_ratio(returns; target=0.0, annualization=252)
max_drawdown(returns)
calmar_ratio(returns; annualization=252)
omega_ratio(returns; threshold=0.0)
information_ratio(returns, benchmark_returns; annualization=252)
```

## License

MIT
