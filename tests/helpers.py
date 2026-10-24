"""Test helpers shared across test modules."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

NOW = datetime(2026, 10, 1, tzinfo=UTC)
# The public kernel checkout next to this repository, with its seed programs.
DEFAULT_PUBLIC = Path(__file__).resolve().parents[1].parent / "science.me"
# The public seed programs, in the order they freeze (dependencies first, R7).
SEED_PROGRAMS = (
    "lens-admissions",
    "seir-core",
    "hub-baseline",
    "flex-loglinear",
    "seir-x1-s1",
    "seir-x1-s0",
    "seir-x0-s1",
    "seir-x0-s0",
)


def at(hours: float) -> datetime:
    """NOW plus `hours` (an aware UTC datetime)."""
    return NOW + timedelta(hours=hours)
