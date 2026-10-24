"""A toy generator: public, invented dynamics for the lab's tests and as an example of the protocol.

Each jurisdiction has a level of weekly counts.  The covariate x lowers it by `exp(-x_effect * x)`, school
term raises it by `1 + school_effect` (a closure takes the term away), and vaccination scales it by
`1 - fraction` from its start.  Counts are Poisson, final at once (no reporting delay), drawn week by
week so that an arm equals its baseline before its intervention starts.  Nothing here resembles the
private worlds beyond the protocol.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ValidationError
from scienceme.model import Frozen, Intervention, Row, StreamKind, StreamSpec, Vintage

from scienceme_lab.generator import (
    COVARIATE_X_ROLE,
    SCHOOL_CLAIM,
    SCHOOL_TERM_ROLE,
    SCHOOLS_CLOSED,
    VACCINATE_FRACTION,
    X_CLAIM,
    GeneratorError,
)

X_LEAD = 2  # weeks of x released beyond the issue
CHANNEL_SCALES = {"admissions": 1.0, "ed-visits": 3.0}
LEVEL_RANGE = (20.0, 200.0)
EFFECT_RANGE = (0.2, 0.5)
P_NO_EFFECT = 0.5
HOLIDAY_WEEKS = 2


class ToyConfig(Frozen):
    """A toy world.  Per-jurisdiction tuples align with `jurisdictions`; `holidays[j]` are j's weeks out of school."""

    generator: Literal["toy"] = "toy"
    id: str
    seed: int
    weeks: int
    jurisdictions: tuple[str, ...]
    levels: tuple[float, ...]
    x_effect: float
    school_effect: float
    holidays: dict[str, tuple[int, ...]]


def _salt(channel: str) -> int:
    return int.from_bytes(hashlib.sha256(channel.encode()).digest()[:4], "big")


def _acts_in(intervention: Intervention, j: str) -> bool:
    return intervention.jurisdictions is None or j in intervention.jurisdictions


class ToyWorld:
    """A `World` over a `ToyConfig`."""

    def __init__(self, cfg: ToyConfig) -> None:
        self._cfg = cfg

    id = property(lambda self: self._cfg.id)
    weeks = property(lambda self: self._cfg.weeks)
    jurisdictions = property(lambda self: self._cfg.jurisdictions)

    @property
    def flat_jurisdiction(self) -> str:
        return self._cfg.jurisdictions[0]

    @property
    def channels(self) -> frozenset[str]:
        return frozenset(CHANNEL_SCALES)

    def x(self) -> np.ndarray:
        """x per jurisdiction for `weeks + X_LEAD` weeks; zero in the flat jurisdiction."""
        values = np.random.default_rng([self._cfg.seed, 1]).standard_normal((len(self.jurisdictions), self.weeks + X_LEAD))
        values[0] = 0.0
        return values

    def term(self) -> np.ndarray:
        """1 in school weeks, 0 in holidays, per jurisdiction and week."""
        term = np.ones((len(self.jurisdictions), self.weeks))
        for j, name in enumerate(self.jurisdictions):
            term[j, list(self._cfg.holidays[name])] = 0.0
        return term

    def _rates(self, channel: str, intervention: Intervention | None) -> np.ndarray:
        cfg = self._cfg
        term = self.term()
        vaccinated = np.zeros_like(term)
        if intervention is not None:
            for j, name in enumerate(self.jurisdictions):
                if not _acts_in(intervention, name):
                    continue
                if intervention.variable == SCHOOLS_CLOSED:
                    end = self.weeks - 1 if intervention.end_week is None else intervention.end_week
                    term[j, intervention.start_week : end + 1] *= 1.0 - intervention.value
                elif intervention.variable == VACCINATE_FRACTION:
                    vaccinated[j, intervention.start_week :] = intervention.value
                else:
                    raise ValueError(f"the toy generator does not simulate {intervention.variable!r}")
        levels = np.asarray(cfg.levels)[:, None]
        x = self.x()[:, : self.weeks]
        return CHANNEL_SCALES[channel] * levels * np.exp(-cfg.x_effect * x) * (1 + cfg.school_effect * term) * (1 - vaccinated)

    def _counts(self, spec: StreamSpec) -> np.ndarray:
        if spec.role not in CHANNEL_SCALES:
            raise ValueError(f"the toy world has no channel {spec.role!r}")
        rates = self._rates(spec.role, spec.intervention)
        rng = np.random.default_rng([self._cfg.seed, _salt(spec.role)])
        counts = np.empty(rates.shape, dtype=np.int64)
        for week in range(self.weeks):  # week-major: an arm's draws match the baseline's until it diverges
            counts[:, week] = rng.poisson(rates[:, week])
        return counts if spec.support_max is None else np.minimum(counts, spec.support_max)

    def vintage(self, spec: StreamSpec, issue: int) -> Vintage:
        if spec.world != self.id:
            raise ValueError(f"stream {spec.id!r} is not of world {self.id!r}")
        if spec.kind is StreamKind.COUNT:
            values, shown = self._counts(spec), min(issue, self.weeks)
        elif spec.role == COVARIATE_X_ROLE:
            values = self.x()
            shown = min(issue + X_LEAD, values.shape[1])
        elif spec.role == SCHOOL_TERM_ROLE:
            values, shown = self.term(), self.weeks
        else:
            raise ValueError(f"the toy world has no stream of role {spec.role!r}")
        rows = tuple(
            Row(jurisdiction=name, week=week, value=float(values[self.jurisdictions.index(name), week]))
            for name in spec.jurisdictions
            for week in range(max(0, shown))
        )
        return Vintage(stream=spec.id, issue=issue, rows=rows)

    def truth(self) -> dict[str, bool]:
        return {X_CLAIM: self._cfg.x_effect > 0, SCHOOL_CLAIM: self._cfg.school_effect > 0}

    def dump(self) -> bytes:
        return self._cfg.model_dump_json().encode("utf-8")


class ToyGenerator:
    """Draws `ToyWorld`s."""

    name = "toy"
    interventions = frozenset({SCHOOLS_CLOSED, VACCINATE_FRACTION})
    world_class: type[ToyWorld] = ToyWorld

    def draw(self, world_id: str, weeks: int, n_jurisdictions: int, rng: np.random.Generator, cache: Path) -> ToyWorld:
        jurisdictions = tuple(f"j{j}" for j in range(n_jurisdictions))

        def effect() -> float:
            return 0.0 if rng.random() < P_NO_EFFECT else float(rng.uniform(*EFFECT_RANGE))

        cfg = ToyConfig(
            id=world_id,
            seed=int(rng.integers(2**32)),
            weeks=weeks,
            jurisdictions=jurisdictions,
            levels=tuple(float(v) for v in rng.uniform(*LEVEL_RANGE, size=n_jurisdictions)),
            x_effect=effect(),
            school_effect=effect(),
            holidays={j: tuple(sorted(int(w) for w in rng.choice(weeks, HOLIDAY_WEEKS, replace=False))) for j in jurisdictions},
        )
        return self.world_class(cfg)

    def load(self, config: bytes, cache: Path) -> ToyWorld:
        try:
            return self.world_class(ToyConfig.model_validate_json(config))
        except ValidationError as e:
            raise GeneratorError(f"not a toy world's configuration: {e}") from e


GENERATOR = ToyGenerator()
