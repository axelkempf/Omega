from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pkgutil import extend_path

# Allow composition with separately installed distributions that provide
# additional submodules under the same top-level namespace (e.g. `omega._rust`).
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

# Maturin wheels that expose `module-name = "omega._rust"` often ship only
# `omega/_rust*.so` without an `omega/__init__.py` (PEP 420 implicit namespace).
# When this repository is installed editable, `omega` is imported from `src/omega`.
# In that case, we must explicitly include any other `omega/` directories on
# sys.path so `import omega._rust` resolves reliably in CI.
for _entry in sys.path:
    if not _entry:
        continue
    try:
        _omega_dir = Path(_entry) / "omega"
    except TypeError:
        continue
    if _omega_dir.is_dir():
        _omega_dir_str = str(_omega_dir)
        if _omega_dir_str not in __path__:
            __path__.append(_omega_dir_str)


def _read_package_version() -> str:
    try:
        return version("omega")
    except PackageNotFoundError:
        # Editable installs and source checkouts may import this package without
        # installed distribution metadata.
        return "0.0.0"


__version__ = _read_package_version()

__all__ = ["__version__"]
