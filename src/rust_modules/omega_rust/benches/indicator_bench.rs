//! Benchmarks for indicator implementations.
//!
//! Run with: `cargo bench`
//!
//! These benchmarks measure the performance of Rust implementations
//! to compare against Python baselines.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Import internal modules for benchmarking
// Note: This requires the library to expose these for benchmarks
mod ema_bench {
    pub fn ema_impl(prices: &[f64], period: usize) -> Vec<f64> {
        if period == 0 || prices.is_empty() || period > prices.len() {
            return vec![];
        }

        let Ok(period_u32) = u32::try_from(period) else {
            return vec![];
        };

        let alpha = 2.0 / (f64::from(period_u32) + 1.0);
        let mut result = Vec::with_capacity(prices.len());
        let mut ema_val = prices[0];
        result.push(ema_val);

        for &price in &prices[1..] {
            ema_val = alpha.mul_add(price, (1.0 - alpha) * ema_val);
            result.push(ema_val);
        }

        result
    }
}

fn generate_prices(n: usize) -> Vec<f64> {
    // Generate pseudo-random prices for benchmarking
    // Using a simple PRNG for reproducibility
    let mut prices = Vec::with_capacity(n);
    let mut price = 100.0;
    let mut seed = 42u64;

    for _ in 0..n {
        // Simple LCG PRNG
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let upper_bits = u32::try_from(seed >> 33).unwrap_or(0);
        let random = f64::from(upper_bits) / f64::from(u32::MAX);

        // Price change: -1% to +1%
        let change = (random - 0.5) * 0.02;
        price *= 1.0 + change;
        prices.push(price);
    }

    prices
}

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("EMA");

    for size in [1_000_usize, 10_000, 100_000, 1_000_000] {
        let prices = generate_prices(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &prices, |b, prices| {
            b.iter(|| ema_bench::ema_impl(black_box(prices), black_box(20)));
        });
    }

    group.finish();
}

fn bench_ema_varying_period(c: &mut Criterion) {
    let mut group = c.benchmark_group("EMA_Period");
    let prices = generate_prices(100_000);

    for period in [5_usize, 10, 20, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::from_parameter(period),
            &period,
            |b, &period| {
                b.iter(|| ema_bench::ema_impl(black_box(&prices), black_box(period)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_ema, bench_ema_varying_period);
criterion_main!(benches);
