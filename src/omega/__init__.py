from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

# Allow composition with separately installed distributions that provide
# additional submodules under the same top-level namespace (e.g. `omega._rust`).
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]


def _read_package_version() -> str:
    try:
        return version("omega")
    except PackageNotFoundError:
        # Editable installs and source checkouts may import this package without
        # installed distribution metadata.
        return "0.0.0"


__version__ = _read_package_version()

__all__ = ["__version__"]
