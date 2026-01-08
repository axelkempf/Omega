"""Omega Trading Stack - Root namespace package.

This package provides the namespace for FFI extensions:
- omega._rust: Rust-based optimized indicators (PyO3)
- omega._julia: Julia-based numerical routines (PythonCall)
"""

from __future__ import annotations

import importlib

__all__: list[str] = []


def __getattr__(name: str):
    """Lazy import for optional FFI extensions.

    Uses importlib.import_module to avoid recursive __getattr__ calls.
    """
    if name == "_rust":
        try:
            return importlib.import_module("omega._rust")
        except ImportError:
            raise ImportError(
                "omega._rust not available. Install with: "
                "pip install -e './src/rust_modules/omega_rust'"
            ) from None
    if name == "_julia":
        try:
            return importlib.import_module("omega._julia")
        except ImportError:
            raise ImportError(
                "omega._julia not available. Install Julia FFI with: "
                "pip install juliacall && "
                "julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.instantiate()'"
            ) from None
    raise AttributeError(f"module 'omega' has no attribute '{name}'")
