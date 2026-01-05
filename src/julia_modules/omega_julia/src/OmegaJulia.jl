# =============================================================================
# OmegaJulia - High-Performance Julia Extensions
# =============================================================================
# Task-ID: P4-07 | Phase: 4 – Build-System
#
# This module provides Julia implementations for computationally intensive
# algorithms, particularly Monte Carlo simulations and statistical analysis.
#
# Usage from Python:
#   from juliacall import Main as jl
#   jl.seval("using OmegaJulia")
#   var = jl.monte_carlo_var(returns, 10000, 0.95)
# =============================================================================

module OmegaJulia

using Statistics
using StatsBase
using Distributions
using Random
using LinearAlgebra
using Dates
using DataFrames
using Arrow

# Module version
const VERSION = v"0.1.0"

# Export public API
export monte_carlo_var,
    monte_carlo_var_detailed,
    rolling_sharpe,
    rolling_sortino,
    rolling_calmar,
    block_bootstrap,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown

# Export error handling (FFI boundary)
export ErrorCode,
    is_recoverable,
    error_category,
    FfiResult,
    ok_result,
    error_result,
    ffi_safe

# Include submodules
include("error.jl")  # Error codes for FFI (sync with Python/Rust)
include("monte_carlo.jl")
include("rolling_stats.jl")
include("bootstrap.jl")
include("risk_metrics.jl")

# Re-export ErrorCodes module members
using .ErrorCodes

# Module initialization
function __init__()
    # Seed random number generator for reproducibility in tests
    # Production code should use explicit seeds
    @info "OmegaJulia v$VERSION initialized"
end

end # module OmegaJulia
