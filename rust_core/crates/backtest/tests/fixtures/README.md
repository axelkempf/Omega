# Backtest Test Fixtures

This directory contains test fixtures for the backtest crate integration tests.

## Structure

```
fixtures/
├── configs/           # Test configuration files
│   ├── minimal.json   # Minimal valid config
│   ├── with_sessions.json  # Config with session windows
│   └── with_costs.json     # Config with custom cost settings
└── README.md
```

## Usage

Load fixtures in tests using:

```rust
fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture_config(name: &str) -> String {
    std::fs::read_to_string(fixtures_root().join("configs").join(name))
        .expect("fixture")
}
```

## Data Files

Market data for tests should be placed in the shared Python fixtures directory
at `python/tests/fixtures/data/` and accessed via the `OMEGA_DATA_PARQUET_ROOT`
environment variable.

## Adding New Fixtures

1. Create a new JSON config in `configs/`
2. Ensure the config is valid according to the V2 schema
3. Add corresponding tests in `integration_test.rs`
