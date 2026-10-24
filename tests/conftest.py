"""Fixtures shared by the test modules; the builders they use live in helpers.py."""

from pathlib import Path

import pytest
from scienceme.workspace import Workspace


@pytest.fixture
def public(tmp_path: Path) -> Workspace:
    """An empty public (science.me) workspace."""
    return Workspace.create(tmp_path / "public")


@pytest.fixture
def private(tmp_path: Path) -> Path:
    """An empty private root."""
    root = tmp_path / "private"
    root.mkdir()
    return root
