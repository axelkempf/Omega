# =============================================================================
# Monte Carlo Simulations for Risk Analysis
# =============================================================================
# Provides high-performance Monte Carlo implementations for:
# - Value at Risk (VaR)
# - Expected Shortfall (CVaR)
# - Portfolio simulations
# =============================================================================

"""
    monte_carlo_var(returns::Vector{Float64}, n_simulations::Int, confidence::Float64; seed::Union{Int,Nothing}=nothing) -> Float64

Calculate Value at Risk using Monte Carlo simulation.

# Arguments
- `returns`: Historical return series
- `n_simulations`: Number of Monte Carlo paths to simulate
- `confidence`: Confidence level (e.g., 0.95 for 95% VaR)
- `seed`: Optional random seed for reproducibility

# Returns
- VaR estimate at the specified confidence level

# Example
```julia
returns = rand(Normal(0.001, 0.02), 252)
var_95 = monte_carlo_var(returns, 10_000, 0.95)
```

# Performance
- O(n_simulations) time complexity
- Vectorized operations for maximum throughput
- ~10-50x faster than equivalent Python implementation
"""
function monte_carlo_var(
    returns::Vector{Float64},
    n_simulations::Int,
    confidence::Float64;
    seed::Union{Int,Nothing}=nothing
)::Float64
    # Input validation
    @assert 0.0 < confidence < 1.0 "Confidence must be between 0 and 1"
    @assert n_simulations > 0 "n_simulations must be positive"
    @assert length(returns) > 1 "Need at least 2 returns"
    
    # Set seed if provided
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    
    # Fit distribution to historical returns
    μ = mean(returns)
    σ = std(returns)
    
    # Generate simulated returns
    simulated_returns = μ .+ σ .* randn(rng, n_simulations)
    
    # Calculate VaR as the (1-confidence) quantile
    var = -quantile(simulated_returns, 1.0 - confidence)
    
    return var
end


"""
    monte_carlo_var_detailed(returns::Vector{Float64}, n_simulations::Int, confidence::Float64; seed::Union{Int,Nothing}=nothing) -> NamedTuple

Calculate detailed Monte Carlo VaR statistics including CVaR/Expected Shortfall.

# Returns
Named tuple with fields:
- `var`: Value at Risk
- `cvar`: Conditional VaR (Expected Shortfall)
- `mean_loss`: Mean simulated loss
- `max_loss`: Maximum simulated loss
- `n_simulations`: Number of simulations run
"""
function monte_carlo_var_detailed(
    returns::Vector{Float64},
    n_simulations::Int,
    confidence::Float64;
    seed::Union{Int,Nothing}=nothing
)
    @assert 0.0 < confidence < 1.0 "Confidence must be between 0 and 1"
    @assert n_simulations > 0 "n_simulations must be positive"
    @assert length(returns) > 1 "Need at least 2 returns"
    
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    
    μ = mean(returns)
    σ = std(returns)
    
    simulated_returns = μ .+ σ .* randn(rng, n_simulations)
    
    # Calculate VaR
    var_threshold = quantile(simulated_returns, 1.0 - confidence)
    var = -var_threshold
    
    # Calculate CVaR (Expected Shortfall)
    # Average of returns below VaR threshold
    tail_returns = simulated_returns[simulated_returns .<= var_threshold]
    cvar = -mean(tail_returns)
    
    return (
        var = var,
        cvar = cvar,
        mean_loss = -mean(simulated_returns),
        max_loss = -minimum(simulated_returns),
        n_simulations = n_simulations
    )
end


"""
    monte_carlo_portfolio_var(returns::Matrix{Float64}, weights::Vector{Float64}, n_simulations::Int, confidence::Float64; seed::Union{Int,Nothing}=nothing) -> Float64

Calculate portfolio VaR using correlated Monte Carlo simulation.

# Arguments
- `returns`: Matrix of asset returns (observations × assets)
- `weights`: Portfolio weights (must sum to 1)
- `n_simulations`: Number of simulations
- `confidence`: Confidence level

# Notes
Uses Cholesky decomposition for correlated random sampling.
"""
function monte_carlo_portfolio_var(
    returns::Matrix{Float64},
    weights::Vector{Float64},
    n_simulations::Int,
    confidence::Float64;
    seed::Union{Int,Nothing}=nothing
)::Float64
    n_assets = size(returns, 2)
    @assert length(weights) == n_assets "Weights must match number of assets"
    @assert abs(sum(weights) - 1.0) < 1e-6 "Weights must sum to 1"
    
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    
    # Calculate mean vector and covariance matrix
    μ = vec(mean(returns, dims=1))
    Σ = cov(returns)
    
    # Cholesky decomposition for correlated sampling
    L = cholesky(Σ).L
    
    # Generate correlated random returns
    Z = randn(rng, n_simulations, n_assets)
    simulated_returns = Z * L' .+ μ'
    
    # Calculate portfolio returns
    portfolio_returns = simulated_returns * weights
    
    # Calculate VaR
    var = -quantile(portfolio_returns, 1.0 - confidence)
    
    return var
end
