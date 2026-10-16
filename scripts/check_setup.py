"""Validate the lightweight Phase 1 development setup."""
import importlib
import platform
import sys

from src.config import PROJECT_ROOT, RANDOM_SEED


def main() -> None:
    modules = ("pandas", "numpy", "sklearn", "fastapi", "pytest", "ruff")
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Random seed: {RANDOM_SEED}")
    if missing:
        raise SystemExit(
            f"Missing dependencies: {', '.join(missing)}. "
            "Run python -m pip install -r requirements-dev.txt"
        )
    print("Setup OK: all Phase 1 dependencies can be imported.")


if __name__ == "__main__":
    main()
