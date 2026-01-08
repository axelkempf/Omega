"""Omega Trading Stack - Root namespace package.

This package provides the namespace for FFI extensions:
- omega._rust: Rust-based optimized indicators (PyO3)
- omega._julia: Julia-based numerical routines (PythonCall)
"""

__all__ = []


# Lazy import for optional Rust extension
def __getattr__(name):
    if name == "_rust":
        try:
            from omega import _rust as rust_mod

            return rust_mod
        except ImportError:
            raise ImportError(
                "omega._rust not available. Install with: "
                "pip install -e './src/rust_modules/omega_rust'"
            ) from None
    raise AttributeError(f"module 'omega' has no attribute '{name}'")
