"""The generator protocol: what the custodian may ask of a world, and how generators are found.

A generator draws worlds; a world answers for its public facts (id, weeks, jurisdictions, the flat
jurisdiction, the channels it can be observed through), publishes any of its streams as a vintage at any
issue (`World.vintage`), says which of the mission's claims hold in it (`World.truth`, for the harness
only), and dumps its configuration for the custodian to keep privately.  Everything else about a world is
the generator's business and stays wherever the generator lives.

Generators are installed packages that register a `Generator` object as an entry point in the group
`GENERATORS_GROUP`, under the generator's own name.  The vocabulary below is the one definition the
custodian, the harness and every generator share.
"""

from __future__ import annotations

import importlib
import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

import numpy as np
from scienceme.model import StreamSpec, Vintage

GENERATORS_GROUP = "scienceme_lab.generators"

# The largest count any count stream can report, the same for every world: a bound derived from the
# populations would disclose one jurisdiction's exact population in the public stream spec.  It sits far
# above any feasible weekly count, while keeping a pmf that spans the whole support small enough to score.
SUPPORT_MAX = 1_000_000

# Stream roles of the covariates every world publishes besides its count channels.
COVARIATE_X_ROLE = "covariate-x"
SCHOOL_TERM_ROLE = "school-term"

# Intervention variables.
SCHOOLS_CLOSED = "schools_closed"
VACCINATE_FRACTION = "vaccinate_fraction"

# Claim ids of the tier 1 mission; a world's truth decides the first two.
X_CLAIM = "covariate-x"
FLAT_CLAIM = "covariate-x-flat"
SCHOOL_CLAIM = "school"
TRUTH_CLAIMS = (X_CLAIM, SCHOOL_CLAIM)


class World(Protocol):
    """One drawn world.  Only what the custodian publishes from it is public."""

    @property
    def id(self) -> str: ...

    @property
    def weeks(self) -> int: ...

    @property
    def jurisdictions(self) -> tuple[str, ...]: ...

    @property
    def flat_jurisdiction(self) -> str:
        """The jurisdiction whose covariate x is identically zero (the negative control)."""
        ...

    @property
    def channels(self) -> frozenset[str]:
        """The count roles the world can be observed through."""
        ...

    def vintage(self, spec: StreamSpec, issue: int) -> Vintage:
        """The vintage of the world's stream `spec` published at `issue`.

        A count stream (a channel's base stream or an arm of one, under `spec.intervention`) shows every
        jurisdiction for the weeks before `min(issue, weeks)`, integer values in `[0, spec.support_max]`,
        for any issue, also past `weeks`.  The covariate x shows at least the weeks before `issue`; the
        school term shows every week.  An arm matches its baseline in every week before its start.
        """
        ...

    def truth(self) -> dict[str, bool]:
        """Whether each claim of `TRUTH_CLAIMS` holds in the world."""
        ...

    def dump(self) -> bytes:
        """The world's configuration: a JSON object whose `"generator"` field names its generator."""
        ...


class Generator(Protocol):
    """Draws worlds and loads them back from their dumps.  `cache` is a private directory a world may use."""

    name: str
    interventions: frozenset[str]

    def draw(self, world_id: str, weeks: int, n_jurisdictions: int, rng: np.random.Generator, cache: Path) -> World: ...

    def load(self, config: bytes, cache: Path) -> World: ...


class GeneratorError(Exception):
    """Raised when a generator cannot be found, or a dump does not name one."""


def generators() -> dict[str, Generator]:
    """Every installed generator by name.  GeneratorError when an entry point's name is not its generator's."""
    found: dict[str, Generator] = {}
    for point in entry_points(group=GENERATORS_GROUP):
        loaded = point.load()
        if loaded.name != point.name:
            raise GeneratorError(f"the entry point {point.name!r} holds the generator {loaded.name!r}")
        found[point.name] = loaded
    return found


def generator_of(config: bytes) -> str:
    """The `"generator"` field of a dumped configuration; GeneratorError when there is none."""
    try:
        value = json.loads(config)
    except ValueError as e:
        raise GeneratorError(f"a world's configuration is not JSON: {e}") from e
    name = value.get("generator") if isinstance(value, dict) else None
    if not isinstance(name, str):
        raise GeneratorError("a world's configuration must be a JSON object with a string field 'generator'")
    return name


def _top_package(generator: Generator) -> str:
    return type(generator).__module__.split(".")[0]


def generator_source(generator: Generator) -> Path:
    """The directory of the top-level package that defines `generator`: what a generator commitment seals."""
    module = importlib.import_module(_top_package(generator))
    assert module.__file__ is not None  # a package imported from files
    return Path(module.__file__).resolve().parent


def source_name(generator: Generator) -> str:
    """The import name of the top-level package that defines `generator`: the name its source is sealed under."""
    return _top_package(generator)
