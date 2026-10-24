"""Test helpers shared across test modules."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from scienceme.model import Intervention

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


def close_schools(start: int, end: int | None, value: float = 1.0, where: tuple[str, ...] | None = ("j1",)) -> Intervention:
    return Intervention(variable="schools_closed", value=value, start_week=start, end_week=end, jurisdictions=where)


def vaccinate(start: int, value: float, where: tuple[str, ...] | None = ("j1",)) -> Intervention:
    return Intervention(variable="vaccinate_fraction", value=value, start_week=start, jurisdictions=where)
