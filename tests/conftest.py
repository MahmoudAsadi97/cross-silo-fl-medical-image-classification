"""Shared pytest fixtures + a skip for tests that need the optional torch stack."""
import importlib.util
import pytest

TORCH = importlib.util.find_spec("torch") is not None


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="torch not installed (optional [torch] extra)")
    for item in items:
        if "torch" in item.keywords and not TORCH:
            item.add_marker(skip)
