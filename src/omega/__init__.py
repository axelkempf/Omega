from __future__ import annotations

"""Omega top-level Python package.

This repository historically exposed most modules as top-level packages (e.g.
`backtest_engine`, `hf_engine`, `shared`). For Rust/Julia migration work we also
need a stable `omega.*` namespace so extension modules can live under
`omega._rust` and future FFI shims can be added without breaking imports.

The Rust wheel built via maturin installs the compiled extension as
`omega._rust`. This package intentionally does not provide a pure-Python fallback
for that module; consumers should treat it as an optional acceleration layer.
"""

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

# Allow splitting the `omega.*` namespace across multiple distributions.
#
# This is required because the Rust extension is built and distributed as a
# separate wheel (module-name: "omega._rust"). In CI we install that wheel and
# then install this repository in editable mode, which would otherwise make
# `omega` resolve only from `src/omega/` and prevent Python from finding
# `site-packages/omega/_rust.*`.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

try:
    __version__ = version("omega")
except PackageNotFoundError:  # pragma: no cover
    # When running from a source checkout without installing the distribution.
    __version__ = "unknown"

__all__ = ["__version__"]
