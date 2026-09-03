"""Make ``python -m src.reconstruction._internal.calibration <file>`` work."""

from .diagnostic import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
