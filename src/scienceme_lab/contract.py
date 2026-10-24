"""The contract every generator keeps: checks a generator's own tests run, one invariant each.

Each check takes a generator and a fresh private directory and raises `AssertionError` with a message
naming what broke.  `CONTRACT` lists them, for a test to parametrize over.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scienceme.model import Intervention, StreamKind, StreamSpec

from scienceme_lab.generator import (
    COVARIATE_X_ROLE,
    SCHOOL_TERM_ROLE,
    SCHOOLS_CLOSED,
    SUPPORT_MAX,
    TRUTH_CLAIMS,
    VACCINATE_FRACTION,
    Generator,
    World,
    generator_of,
)

WEEKS = 24
JURISDICTIONS = 3
SEED = 20261023


def _world(generator: Generator, cache: Path, seed: int = SEED) -> World:
    return generator.draw("contract", WEEKS, JURISDICTIONS, np.random.default_rng(seed), cache)


def _count(world: World, channel: str) -> StreamSpec:
    return StreamSpec(
        id=f"{world.id}/{channel}", world=world.id, role=channel, kind=StreamKind.COUNT,
        jurisdictions=world.jurisdictions, support_max=SUPPORT_MAX,
    )


def _covariate(world: World, name: str, role: str) -> StreamSpec:
    return StreamSpec(id=f"{world.id}/{name}", world=world.id, role=role, kind=StreamKind.COVARIATE, jurisdictions=world.jurisdictions)


def _arm(base: StreamSpec, intervention: Intervention) -> StreamSpec:
    fields = {"id": f"{base.id}~arm", "intervention": intervention, "parent": base.id}
    return StreamSpec.model_validate(base.model_dump() | fields)


def _cells(world: World, spec: StreamSpec, issue: int) -> dict[tuple[str, int], float]:
    return {(r.jurisdiction, r.week): r.value for r in world.vintage(spec, issue).rows}


def draws_reproducibly(generator: Generator, cache: Path) -> None:
    first, second = _world(generator, cache / "a"), _world(generator, cache / "b")
    assert first.dump() == second.dump(), "two draws from the same seed differ"
    channel = min(first.channels)
    assert _cells(first, _count(first, channel), WEEKS) == _cells(second, _count(second, channel), WEEKS), (
        "two worlds drawn from the same seed publish different counts"
    )


def dumps_and_loads_the_same_world(generator: Generator, cache: Path) -> None:
    world = _world(generator, cache / "drawn")
    assert generator_of(world.dump()) == generator.name, "a dump does not name its generator"
    loaded = generator.load(world.dump(), cache / "loaded")
    assert loaded.dump() == world.dump(), "a loaded world dumps differently"
    for channel in sorted(world.channels):
        for issue in (1, WEEKS // 2, WEEKS):
            assert _cells(loaded, _count(loaded, channel), issue) == _cells(world, _count(world, channel), issue), (
                f"a loaded world publishes other {channel} counts at issue {issue}"
            )


def counts_cover_the_weeks_before_the_issue(generator: Generator, cache: Path) -> None:
    world = _world(generator, cache)
    for channel in sorted(world.channels):
        spec = _count(world, channel)
        for issue in (0, 1, WEEKS // 2, WEEKS):
            cells = _cells(world, spec, issue)
            expected = {(j, w) for j in world.jurisdictions for w in range(min(issue, WEEKS))}
            assert set(cells) == expected, f"{channel} at issue {issue} does not show exactly the weeks before it"
            assert all(v == int(v) and 0 <= v <= SUPPORT_MAX for v in cells.values()), (
                f"{channel} at issue {issue} has a count that is not an integer in [0, support_max]"
            )


def counts_are_defined_past_the_last_week(generator: Generator, cache: Path) -> None:
    world = _world(generator, cache)
    for channel in sorted(world.channels):
        cells = _cells(world, _count(world, channel), WEEKS + 5)
        assert set(cells) == {(j, w) for j in world.jurisdictions for w in range(WEEKS)}, (
            f"{channel} past the last week does not show every week"
        )


def arms_match_the_baseline_before_they_start(generator: Generator, cache: Path) -> None:
    world = _world(generator, cache)
    start = WEEKS // 2
    interventions = {
        SCHOOLS_CLOSED: dict(value=1.0, start_week=start, end_week=start + 2),
        VACCINATE_FRACTION: dict(value=0.3, start_week=start),
    }
    for variable in sorted(generator.interventions & interventions.keys()):
        for j in world.jurisdictions:
            intervention = Intervention(variable=variable, jurisdictions=(j,), **interventions[variable])
            for channel in sorted(world.channels):
                base = _count(world, channel)
                for issue in (start, WEEKS):
                    arm, baseline = _cells(world, _arm(base, intervention), issue), _cells(world, base, issue)
                    before = {cell: v for cell, v in baseline.items() if cell[1] < start}
                    assert {cell: v for cell, v in arm.items() if cell[1] < start} == before, (
                        f"an arm of {variable} in {j} differs from the baseline before its start ({channel}, issue {issue})"
                    )


def covariates_are_released(generator: Generator, cache: Path) -> None:
    world = _world(generator, cache)
    assert world.flat_jurisdiction in world.jurisdictions, "the flat jurisdiction is not one of the world's"
    x = _covariate(world, "x", COVARIATE_X_ROLE)
    for issue in (1, WEEKS // 2):
        cells = _cells(world, x, issue)
        assert max(w for _, w in cells) >= issue - 1, f"x at issue {issue} does not reach week {issue - 1}"
        assert all(v == 0 for (j, _), v in cells.items() if j == world.flat_jurisdiction), "x is not zero in the flat jurisdiction"
    term = _cells(world, _covariate(world, "term", SCHOOL_TERM_ROLE), 0)
    assert set(term) == {(j, w) for j in world.jurisdictions for w in range(WEEKS)}, "the term calendar is not whole at issue 0"


def keeps_to_its_own_cache(generator: Generator, cache: Path) -> None:
    """A world drawn into a cache directory another world used publishes its own counts, not the other's."""
    other = _world(generator, cache / "shared", seed=SEED + 1)
    for channel in sorted(other.channels):  # the other world publishes, filling the shared cache
        other.vintage(_count(other, channel), WEEKS)
    shared, fresh = _world(generator, cache / "shared"), _world(generator, cache / "fresh")
    for channel in sorted(shared.channels):
        assert _cells(shared, _count(shared, channel), WEEKS) == _cells(fresh, _count(fresh, channel), WEEKS), (
            f"a world publishes {channel} counts from another world's cache"
        )


def truth_names_the_claims(generator: Generator, cache: Path) -> None:
    truth = _world(generator, cache).truth()
    assert set(truth) == set(TRUTH_CLAIMS), f"truth names {sorted(truth)}, not the claims {list(TRUTH_CLAIMS)}"
    assert all(isinstance(v, bool) for v in truth.values()), "truth is not a bool per claim"


CONTRACT: tuple[Callable[[Generator, Path], None], ...] = (
    draws_reproducibly,
    dumps_and_loads_the_same_world,
    counts_cover_the_weeks_before_the_issue,
    counts_are_defined_past_the_last_week,
    arms_match_the_baseline_before_they_start,
    covariates_are_released,
    keeps_to_its_own_cache,
    truth_names_the_claims,
)
