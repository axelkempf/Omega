"""Omega Trading Stack - Root namespace package.

This package provides the namespace for FFI extensions:
- omega._rust: Rust-based optimized indicators (PyO3, imports omega_rust)
- omega._julia: Julia-based numerical routines (PythonCall)
"""

from __future__ import annotations

import importlib
import sys

__all__: list[str] = []


def __getattr__(name: str):
    """Lazy import for optional FFI extensions.

    Uses importlib.import_module to avoid recursive __getattr__ calls.
    The Rust module is published as 'omega_rust' but we expose it as 'omega._rust'
    for a cleaner API.
    """
    if name == "_rust":
        try:
            # The Rust wheel is published as 'omega_rust' (standalone package)
            # We re-export it as omega._rust for API consistency
            rust_module = importlib.import_module("omega_rust")
            # Cache in sys.modules so subsequent imports are fast
            sys.modules["omega._rust"] = rust_module
            return rust_module
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
