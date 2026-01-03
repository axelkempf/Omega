# Strategy Registry

The **Strategy Registry** module serves as the central directory for mapping user-friendly logical aliases to technical identifiers (such as account numbers or service IDs). This decoupling allows the UI and users to interact with strategies and data feeds using memorable names instead of cryptic IDs.

## Features

-   **Alias Mapping**: Maintains a dictionary `STRATEGY_ALIAS` that maps logical names (e.g., `datafeed`, `account_10928521`) to their corresponding technical IDs.
-   **Resolution Utility**: Provides a `resolve_alias` function to safely retrieve the technical ID for a given alias.

## Configuration

The mapping is defined in `strategy_alias.py`:

```python
STRATEGY_ALIAS = {
    "datafeed": "15582434",
    "dxfeed": "dxfeed01",
    "account_10928521": "10928521",
    # ... other mappings
}
```

## Usage

Use the `resolve_alias` function to translate an alias into its ID. If the alias is not found in the registry, the function returns the input string as-is, allowing direct use of IDs.

```python
from ui_engine.registry.strategy_alias import resolve_alias

# Resolve a known alias
technical_id = resolve_alias("datafeed")
# Result: "15582434"

# Resolve an unknown alias (returns the input)
unknown_id = resolve_alias("unknown_strategy")
# Result: "unknown_strategy"
```

> [!TIP]
> This registry is used by both the `datafeeds` and `strategies` factories to ensure consistent ID resolution across the application.
