"""Allow running as `python -m bt`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
