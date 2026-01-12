# =============================================================================
# Bootstrap Methods for Statistical Inference
# =============================================================================
# Block bootstrap and related methods for time series analysis with
# autocorrelation preservation.
# =============================================================================

"""
    block_bootstrap(data::Vector{Float64}, block_size::Int, n_samples::Int; seed::Union{Int,Nothing}=nothing) -> Matrix{Float64}

Generate bootstrap samples using block resampling.

Preserves autocorrelation structure by sampling contiguous blocks.

# Arguments
- `data`: Original time series
- `block_size`: Size of each block
- `n_samples`: Number of bootstrap samples to generate
- `seed`: Optional random seed

# Returns
Matrix of shape (length(data), n_samples) containing bootstrap samples.

# Example
```julia
returns = rand(252)
bootstrap_samples = block_bootstrap(returns, 20, 1000)
confidence_intervals = mapslices(x -> quantile(x, [0.025, 0.975]), bootstrap_samples, dims=2)
```
"""
function block_bootstrap(
    data::Vector{Float64},
    block_size::Int,
    n_samples::Int;
    seed::Union{Int,Nothing} = nothing,
)::Matrix{Float64}
    @assert block_size > 0 "block_size must be positive"
    @assert n_samples > 0 "n_samples must be positive"
    @assert length(data) >= block_size "Data must be at least block_size long"

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    n = length(data)
    n_blocks = ceil(Int, n / block_size)
    max_start = n - block_size + 1

    result = Matrix{Float64}(undef, n, n_samples)

    @inbounds for sample = 1:n_samples
        # Sample random block starting positions
        block_starts = rand(rng, 1:max_start, n_blocks)

        # Build bootstrap sample from blocks
        idx = 1
        for start in block_starts
            block_end = min(start + block_size - 1, n)
            block_len = block_end - start + 1

            for j = 0:(block_len-1)
                if idx <= n
                    result[idx, sample] = data[start+j]
                    idx += 1
                end
            end

            if idx > n
                break
            end
        end
    end

    return result
end


"""
    stationary_bootstrap(data::Vector{Float64}, mean_block_size::Float64, n_samples::Int; seed::Union{Int,Nothing}=nothing) -> Matrix{Float64}

Generate bootstrap samples using stationary bootstrap (Politis & Romano, 1994).

Uses geometrically distributed block sizes for stationarity.

# Arguments
- `data`: Original time series
- `mean_block_size`: Expected block size (geometric distribution parameter)
- `n_samples`: Number of bootstrap samples
- `seed`: Optional random seed
"""
function stationary_bootstrap(
    data::Vector{Float64},
    mean_block_size::Float64,
    n_samples::Int;
    seed::Union{Int,Nothing} = nothing,
)::Matrix{Float64}
    @assert mean_block_size > 0 "mean_block_size must be positive"
    @assert n_samples > 0 "n_samples must be positive"

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    n = length(data)
    p = 1.0 / mean_block_size  # Probability of starting new block

    result = Matrix{Float64}(undef, n, n_samples)

    @inbounds for sample = 1:n_samples
        # Start with random position
        pos = rand(rng, 1:n)

        for i = 1:n
            result[i, sample] = data[pos]

            # Decide whether to continue block or start new one
            if rand(rng) < p
                # Start new block at random position
                pos = rand(rng, 1:n)
            else
                # Continue block (wrap around)
                pos = pos < n ? pos + 1 : 1
            end
        end
    end

    return result
end


"""
    bootstrap_confidence_interval(data::Vector{Float64}, statistic::Function, block_size::Int, n_samples::Int, confidence::Float64; seed::Union{Int,Nothing}=nothing) -> Tuple{Float64, Float64}

Calculate bootstrap confidence interval for a statistic.

# Arguments
- `data`: Original data
- `statistic`: Function that computes the statistic of interest
- `block_size`: Block size for bootstrap
- `n_samples`: Number of bootstrap samples
- `confidence`: Confidence level (e.g., 0.95)
- `seed`: Optional random seed

# Returns
Tuple of (lower_bound, upper_bound)
"""
function bootstrap_confidence_interval(
    data::Vector{Float64},
    statistic::Function,
    block_size::Int,
    n_samples::Int,
    confidence::Float64;
    seed::Union{Int,Nothing} = nothing,
)::Tuple{Float64,Float64}
    @assert 0.0 < confidence < 1.0 "Confidence must be between 0 and 1"

    # Generate bootstrap samples
    samples = block_bootstrap(data, block_size, n_samples; seed = seed)

    # Calculate statistic for each sample
    statistics = [statistic(samples[:, i]) for i = 1:n_samples]

    # Calculate percentile confidence interval
    α = (1.0 - confidence) / 2.0
    lower = quantile(statistics, α)
    upper = quantile(statistics, 1.0 - α)

    return (lower, upper)
end
