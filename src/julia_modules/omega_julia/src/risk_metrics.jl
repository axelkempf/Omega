# =============================================================================
# Risk Metrics
# =============================================================================
# Standard risk and performance metrics for trading strategy evaluation.
# =============================================================================

"""
    sharpe_ratio(returns::Vector{Float64}; risk_free_rate::Float64=0.0, annualization::Int=252) -> Float64

Calculate the Sharpe ratio of a return series.

# Arguments
- `returns`: Return series (arithmetic returns)
- `risk_free_rate`: Annual risk-free rate
- `annualization`: Annualization factor (252 for daily, 12 for monthly)

# Returns
Annualized Sharpe ratio
"""
function sharpe_ratio(
    returns::Vector{Float64};
    risk_free_rate::Float64=0.0,
    annualization::Int=252
)::Float64
    @assert length(returns) >= 2 "Need at least 2 returns"
    
    rf_period = risk_free_rate / annualization
    excess_returns = returns .- rf_period
    
    μ = mean(excess_returns)
    σ = std(returns)
    
    if σ < 1e-10
        return 0.0
    end
    
    return (μ / σ) * sqrt(annualization)
end


"""
    sortino_ratio(returns::Vector{Float64}; target::Float64=0.0, annualization::Int=252) -> Float64

Calculate the Sortino ratio using downside deviation.

# Arguments
- `returns`: Return series
- `target`: Target return (MAR)
- `annualization`: Annualization factor
"""
function sortino_ratio(
    returns::Vector{Float64};
    target::Float64=0.0,
    annualization::Int=252
)::Float64
    @assert length(returns) >= 2 "Need at least 2 returns"
    
    μ = mean(returns) - target
    
    # Downside deviation
    downside = filter(r -> r < target, returns)
    
    if length(downside) < 2
        return μ >= 0 ? Inf : -Inf
    end
    
    downside_std = std(downside .- target)
    
    if downside_std < 1e-10
        return μ >= 0 ? Inf : -Inf
    end
    
    return (μ / downside_std) * sqrt(annualization)
end


"""
    max_drawdown(returns::AbstractVector{Float64}) -> Float64

Calculate the maximum drawdown from a return series.

# Returns
Maximum drawdown as a positive value (e.g., 0.15 = 15% drawdown)
"""
function max_drawdown(returns::AbstractVector{Float64})::Float64
    if isempty(returns)
        return 0.0
    end
    
    # Calculate cumulative returns (wealth curve)
    wealth = cumprod(1.0 .+ returns)
    
    # Track running maximum
    running_max = accumulate(max, wealth)
    
    # Calculate drawdowns
    drawdowns = (running_max .- wealth) ./ running_max
    
    return maximum(drawdowns)
end


"""
    calmar_ratio(returns::Vector{Float64}; annualization::Int=252) -> Float64

Calculate the Calmar ratio (annualized return / max drawdown).
"""
function calmar_ratio(
    returns::Vector{Float64};
    annualization::Int=252
)::Float64
    @assert length(returns) >= 2 "Need at least 2 returns"
    
    # Calculate annualized return
    cumulative = prod(1.0 .+ returns) - 1.0
    n_periods = length(returns)
    ann_return = (1.0 + cumulative)^(annualization / n_periods) - 1.0
    
    # Calculate max drawdown
    mdd = max_drawdown(returns)
    
    if mdd < 1e-10
        return ann_return >= 0 ? Inf : -Inf
    end
    
    return ann_return / mdd
end


"""
    omega_ratio(returns::Vector{Float64}; threshold::Float64=0.0) -> Float64

Calculate the Omega ratio.

Omega = Probability-weighted gains above threshold / Probability-weighted losses below threshold
"""
function omega_ratio(
    returns::Vector{Float64};
    threshold::Float64=0.0
)::Float64
    gains = sum(max.(returns .- threshold, 0.0))
    losses = sum(max.(threshold .- returns, 0.0))
    
    if losses < 1e-10
        return gains > 0 ? Inf : 1.0
    end
    
    return gains / losses
end


"""
    information_ratio(returns::Vector{Float64}, benchmark_returns::Vector{Float64}; annualization::Int=252) -> Float64

Calculate the Information Ratio (excess return / tracking error).
"""
function information_ratio(
    returns::Vector{Float64},
    benchmark_returns::Vector{Float64};
    annualization::Int=252
)::Float64
    @assert length(returns) == length(benchmark_returns) "Return series must have same length"
    @assert length(returns) >= 2 "Need at least 2 returns"
    
    excess_returns = returns .- benchmark_returns
    μ = mean(excess_returns)
    tracking_error = std(excess_returns)
    
    if tracking_error < 1e-10
        return 0.0
    end
    
    return (μ / tracking_error) * sqrt(annualization)
end
