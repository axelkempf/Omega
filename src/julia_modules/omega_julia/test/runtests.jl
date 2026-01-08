# =============================================================================
# OmegaJulia Test Suite
# =============================================================================
# Run with: julia --project=. -e "using Pkg; Pkg.test()"
# =============================================================================

using Test
using Statistics
using Random

# Add parent directory to load path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using OmegaJulia

@testset "OmegaJulia Tests" begin

    # =========================================================================
    # Monte Carlo Tests
    # =========================================================================
    @testset "Monte Carlo VaR" begin
        # Generate test returns
        Random.seed!(42)
        returns = randn(252) .* 0.02 .+ 0.0005  # Daily returns

        @testset "Basic VaR calculation" begin
            var = monte_carlo_var(returns, 10_000, 0.95; seed = 42)

            @test var isa Float64
            @test var > 0  # VaR should be positive (loss)
            @test var < 0.2  # Reasonable bound for daily VaR
        end

        @testset "VaR reproducibility" begin
            var1 = monte_carlo_var(returns, 10_000, 0.95; seed = 123)
            var2 = monte_carlo_var(returns, 10_000, 0.95; seed = 123)

            @test var1 == var2  # Same seed should give same result
        end

        @testset "VaR confidence levels" begin
            var_90 = monte_carlo_var(returns, 10_000, 0.90; seed = 42)
            var_95 = monte_carlo_var(returns, 10_000, 0.95; seed = 42)
            var_99 = monte_carlo_var(returns, 10_000, 0.99; seed = 42)

            @test var_90 < var_95 < var_99  # Higher confidence = higher VaR
        end

        @testset "Detailed VaR" begin
            result = monte_carlo_var_detailed(returns, 10_000, 0.95; seed = 42)

            @test haskey(result, :var)
            @test haskey(result, :cvar)
            @test result.cvar >= result.var  # CVaR >= VaR always
            @test result.n_simulations == 10_000
        end

        @testset "Input validation" begin
            @test_throws AssertionError monte_carlo_var(returns, 0, 0.95)
            @test_throws AssertionError monte_carlo_var(returns, 1000, 0.0)
            @test_throws AssertionError monte_carlo_var(returns, 1000, 1.0)
            @test_throws AssertionError monte_carlo_var([1.0], 1000, 0.95)
        end
    end

    # =========================================================================
    # Rolling Statistics Tests
    # =========================================================================
    @testset "Rolling Statistics" begin
        Random.seed!(42)
        returns = randn(500) .* 0.02 .+ 0.0005

        @testset "Rolling Sharpe" begin
            sharpe = rolling_sharpe(returns, 63)

            @test length(sharpe) == length(returns)
            @test all(isnan.(sharpe[1:62]))  # First window-1 values NaN
            @test !isnan(sharpe[63])
            @test all(!isnan, sharpe[63:end])
        end

        @testset "Rolling Sortino" begin
            sortino = rolling_sortino(returns, 63)

            @test length(sortino) == length(returns)
            @test all(isnan.(sortino[1:62]))
        end

        @testset "Rolling Calmar" begin
            calmar = rolling_calmar(returns, 63)

            @test length(calmar) == length(returns)
            @test all(isnan.(calmar[1:62]))
        end

        @testset "Input validation" begin
            @test_throws AssertionError rolling_sharpe(returns, 1)  # Window too small
            @test_throws AssertionError rolling_sharpe(returns[1:10], 20)  # Not enough data
        end
    end

    # =========================================================================
    # Risk Metrics Tests
    # =========================================================================
    @testset "Risk Metrics" begin
        @testset "Sharpe Ratio" begin
            # Varying positive returns with some volatility
            Random.seed!(42)
            pos_returns = 0.01 .+ 0.005 .* randn(252)  # Mean 1% with volatility
            sharpe = sharpe_ratio(pos_returns)

            @test sharpe > 0

            # Zero returns
            zero_returns = fill(0.0, 252)
            @test sharpe_ratio(zero_returns) == 0.0
        end

        @testset "Max Drawdown" begin
            # Test known drawdown
            returns = [0.1, -0.05, -0.1, 0.05, 0.02]
            mdd = max_drawdown(returns)

            @test mdd > 0
            @test mdd <= 1.0  # Can't exceed 100%

            # No drawdown for pure gains
            up_returns = fill(0.01, 100)
            @test max_drawdown(up_returns) ≈ 0.0 atol=1e-10
        end

        @testset "Sortino Ratio" begin
            # All positive returns - should be Inf
            pos_returns = fill(0.01, 252)
            @test isinf(sortino_ratio(pos_returns))

            # Mixed returns
            Random.seed!(42)
            mixed_returns = randn(252) .* 0.02
            sortino = sortino_ratio(mixed_returns)
            @test isfinite(sortino)
        end

        @testset "Omega Ratio" begin
            Random.seed!(42)
            returns = randn(252) .* 0.02 .+ 0.001
            omega = omega_ratio(returns)

            @test omega > 0

            # All above threshold
            @test isinf(omega_ratio(fill(0.01, 100); threshold = -0.1))
        end
    end

    # =========================================================================
    # Bootstrap Tests
    # =========================================================================
    @testset "Bootstrap Methods" begin
        Random.seed!(42)
        data = randn(100)

        @testset "Block Bootstrap" begin
            samples = block_bootstrap(data, 10, 100; seed = 42)

            @test size(samples) == (100, 100)
            @test all(isfinite, samples)
        end

        @testset "Bootstrap reproducibility" begin
            samples1 = block_bootstrap(data, 10, 50; seed = 123)
            samples2 = block_bootstrap(data, 10, 50; seed = 123)

            @test samples1 == samples2
        end

        @testset "Stationary Bootstrap" begin
            samples = stationary_bootstrap(data, 10.0, 100; seed = 42)

            @test size(samples) == (100, 100)
            @test all(isfinite, samples)
        end

        @testset "Confidence Interval" begin
            ci = bootstrap_confidence_interval(data, mean, 10, 500, 0.95; seed = 42)

            @test ci[1] < ci[2]  # Lower < Upper
            @test ci[1] < mean(data) < ci[2]  # Sample mean in CI
        end
    end

end  # OmegaJulia Tests

println("\n✅ All tests passed!")
