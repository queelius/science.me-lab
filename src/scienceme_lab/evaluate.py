"""The tier 1 evaluation: the platform's mechanism, run on replicate worlds whose truth only this harness knows.

A run plays the platform forward on one world in its own scratch directory.  It copies the configured
programs into a fresh public workspace and freezes them, in the listed order, an hour before the world
starts at `T0` (R7); the custodian draws the world and publishes it one issue per hour of platform time.
Every issue is forecast (`run_origin`), and at each decision issue one feasible wanted item is fulfilled:
the one `choose` picks from the gain ranking (`gain`), or one drawn uniformly (`random`).  The `all` run
instead fulfills every feasible item at the first decision issue and decides nothing after: the mechanism
with unlimited arms, on which concentration is judged (spec 6.3).  Every run of a replicate draws the same
world, so the runs differ only in what they chose to observe.

At `evaluate_at` a run records the mixture's weights, the claim verdicts, the truth-matching program's
calibration on the intervened units of the arms and on the same units of the baseline, and the mixture's
log score on the intervened jurisdictions of held-out arms nobody observed, scored against the custodian's
private simulation of them (R15).  `criteria` turns the runs into the pre-registered verdicts (spec
criterion 1); `evaluate` runs every run of a config in worker processes.

Every result is stamped with the code it evaluated (`Stamps`): each program's closure hash, which must be
its freeze in the public registry, the scorer's hash, the hash of this package and the hash of each
generator's source.  Worlds are reached only through the generator protocol (`scienceme_lab.generator`).  A stored result is
reused only under the same config and the same stamps.

Nothing reads the clock.  Programs run with `Isolation.NONE`: simulation only, on the trusted seed programs.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import shutil
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast, get_args

import numpy as np
from pydantic import Field, model_validator
from scienceme.archive import Archive, ArchiveView
from scienceme.calibration import Calibration, calibration
from scienceme.claims import ClaimStanding, Verdict, claim_standings
from scienceme.gain import choose, rank_wanted
from scienceme.manifest import ProgramManifest, closure_hash, in_scope, supports
from scienceme.mission import Mission, WantedItem, load_mission, load_wanted, mission_streams
from scienceme.model import Frozen, Intervention, Isolation, StreamKind, StreamSpec, Target
from scienceme.pmf import Pmf, floored_log_prob, mix
from scienceme.predict import run_origin
from scienceme.registry import Registry, RegistryError
from scienceme.scoring import PointScore, Resolution, resolve, target_order
from scienceme.weights import Advantages, normalized_weights, run_mixture
from scienceme.worker import build_order, package_hash, run_order, scorer_hash
from scienceme.worker_entry import Prediction
from scienceme.workspace import Workspace, WriteOnceError, read_model, write_new

from scienceme_lab.custodian import GRID_CLOSURE_WEEKS, GRID_VACCINATE_VALUE, HORIZONS, Custodian
from scienceme_lab.generator import (
    FLAT_CLAIM,
    SCHOOL_CLAIM,
    SCHOOLS_CLOSED,
    VACCINATE_FRACTION,
    X_CLAIM,
    Generator,
    World,
    generator_source,
    generators,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)  # the platform time of issue 0
ISSUE_INTERVAL = timedelta(hours=1)  # issue i is published at T0 + i * ISSUE_INTERVAL
FREEZE_LEAD = timedelta(hours=1)  # the programs freeze this long before T0
CHANNEL = "admissions"
HELDOUT = "heldout"  # the item-id prefix of the held-out arms
RESULT_FILE = "result.json"  # a finished run's ReplicateResult, in its run directory
CONFIG_FILE = "config.json"  # the EvalConfig whose runs a scratch directory holds

Strategy = Literal["gain", "random", "all"]
STRATEGIES: tuple[Strategy, ...] = get_args(Strategy)

# The generators the seed SEIR theories are well specified for: surgery is judged there and only reported elsewhere.
SURGERY_JUDGED = frozenset({"seir"})

# Workers do their linear algebra on one thread: the programs' matrices are tiny, so more BLAS threads only
# spin, and `workers` processes already fill the cores.  OpenBLAS reads this when numpy loads, so it goes in
# the environment the spawned workers inherit.
WORKER_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

_PACKAGE_DIR = Path(__file__).resolve().parent


class EvaluationError(RuntimeError):
    """Raised when a run cannot produce the quantities the criteria judge."""


def matching_program(truth: Mapping[str, bool]) -> str:
    """The seed SEIR theory that matches `truth` on both distinctions, `seir-x<0|1>-s<0|1>`."""
    return f"seir-x{int(truth[X_CLAIM])}-s{int(truth[SCHOOL_CLAIM])}"


# The four seed SEIR theories, one per combination of the two distinctions.
SEIR_THEORIES = tuple(
    matching_program({X_CLAIM: x, SCHOOL_CLAIM: school}) for x in (True, False) for school in (True, False)
)


def matching_share(
    advantages: Mapping[str, float], theories: Iterable[str], truth: Mapping[str, bool], tau: float
) -> float | None:
    """The weight of the theory matching `truth` in the mixture restricted to `theories` (the frozen seed SEIR
    theories, by name), from their cumulative `advantages`.

    The restricted softmax is always defined, however far another program leads them all, where a share
    of weights normalized over every program can underflow to 0/0.  Another theory without an advantage
    (it never scored) counts 0, as a newcomer does in the mixture.  None when the matching theory is not
    among `theories` or has no advantage itself: a theory that scored no target has earned no share, and
    must not inherit one from rivals that scored below the mixture.
    """
    theories = list(theories)
    match = matching_program(truth)
    if match not in theories or match not in advantages:
        return None
    return normalized_weights(advantages, theories, tau)[match]


def lab_hash() -> str:
    """The `package_hash` of the installed `scienceme_lab` package: the custodian and harness that ran."""
    return package_hash(_PACKAGE_DIR)


# --- configuration and results ---


class Thresholds(Frozen):
    """The pre-registered pass marks of `criteria`."""

    gain_beats_random_min: int = 8
    concentration_share: float = 0.5
    concentration_min: int = 8
    surgery_tolerance: float = 0.10
    power_min: float = 0.8
    abstain_min: float = 0.9


# Marks a field that says where and how fast an evaluation runs, not what it evaluates (`machine_fields`).
_MACHINE = {"machine": True}


class EvalConfig(Frozen):
    """A tier 1 evaluation: `replicates` worlds per generator, each run once by gain, `random_seeds` times at
    random and once with every arm (`all`).

    `programs` are copied from `public_root/programs`, must be frozen there, and are frozen again in the
    listed order, dependencies first.
    """

    generators: tuple[str, ...] = ("seir", "abm")
    replicates: int = Field(10, ge=1)
    weeks: int = 30
    jurisdictions: int = Field(4, ge=1)
    decision_issues: tuple[int, ...] = (6, 9, 12)
    evaluate_at: int = 24
    random_seeds: int = Field(2, ge=1)
    programs: tuple[str, ...]
    public_root: Path = Field(description="the checkout of the public repository", json_schema_extra=_MACHINE)
    master_seed: int = Field(ge=0)
    workers: int = Field(10, ge=1, description="how many runs go at once", json_schema_extra=_MACHINE)
    thresholds: Thresholds = Thresholds()

    @model_validator(mode="after")
    def _check_runnable(self) -> EvalConfig:
        """Distinct generators, distinct decisions before `evaluate_at`, and held-out targets inside the world.

        Whether the generators are installed is checked when the evaluation runs (`stamps`), so a config
        parses anywhere, also where only the public harness is installed.
        """
        if not self.generators or len(set(self.generators)) != len(self.generators):
            raise ValueError(f"generators must be distinct and at least one, got {list(self.generators)}")
        if len(set(self.decision_issues)) != len(self.decision_issues) or not all(
            1 <= d < self.evaluate_at for d in self.decision_issues
        ):
            raise ValueError(f"decision issues must be distinct and in 1 .. {self.evaluate_at - 1}")
        last = self.evaluate_at + 1 + max(HORIZONS)
        if last >= self.weeks:
            raise ValueError(f"held-out targets after issue {self.evaluate_at} reach week {last}, past the world's end")
        return self


def machine_fields() -> tuple[str, ...]:
    """The `EvalConfig` fields that say where and how fast an evaluation runs, not what it evaluates.

    A pre-registered config runs on another machine by overriding only these, and runs that differ only in
    them are the same evaluation's.
    """
    return tuple(name for name, field in EvalConfig.model_fields.items() if field.json_schema_extra == _MACHINE)


class Stamps(Frozen):
    """The code a run evaluated: each program's closure hash by name (its freeze in the public registry), the
    scorer's hash (`scienceme.worker.scorer_hash`), this package's (`lab_hash`), and the `package_hash` of
    each configured generator's source by generator name."""

    programs: dict[str, str]
    scorer: str
    lab: str
    generators: dict[str, str]


class ReplicateResult(Frozen):
    """One run, as it stood at `evaluate_at`.

    `heldout_scores` are the mixture's floored log scores (nats) on the held-out targets (the intervened
    jurisdiction of each held-out arm) some program forecast, as (target, score) pairs in `target_order`;
    `heldout_log_score` is their mean over the `heldout_targets` of them, and the `heldout_skipped` others
    had no forecast.  `weights_cumulative` are the mixture weights of every program's cumulative advantage,
    by program name, and `claims` the verdicts by claim id.  `theory_advantages` are the cumulative
    advantages of the seed SEIR theories that scored, by name, and `theory_share` the matching theory's
    weight in the mixture restricted to the theories (`matching_share`; None when undefined).
    `theory_predicted` and `theory_unscored` count each frozen theory's prediction and unscored records.
    `arm_coverage90` is the truth-matching program's 90% coverage on the resolved targets of the fulfilled
    arms in their intervened jurisdictions, and `baseline_coverage90` its coverage on the resolved
    base-channel targets with the same (jurisdiction, reference week, horizon); each comes with the number
    of points behind it (None for none), so runs pool exactly.  `fulfilled` lists the items observed in
    order; `skipped_decisions` the decision issues where no item was feasible.  `stamps` is the code the
    run evaluated.
    """

    generator: str
    replicate: int
    strategy: Strategy
    seed: int
    stamps: Stamps
    heldout_scores: tuple[tuple[Target, float], ...]
    heldout_log_score: float = Field(allow_inf_nan=False)
    heldout_targets: int
    heldout_skipped: int
    weights_cumulative: dict[str, float]
    theory_advantages: dict[str, float]
    theory_share: float | None
    theory_predicted: dict[str, int]
    theory_unscored: dict[str, int]
    claims: dict[str, Verdict]
    truth: dict[str, bool]
    arm_coverage90: float | None
    arm_points: int
    baseline_coverage90: float | None
    baseline_points: int
    fulfilled: tuple[str, ...]
    skipped_decisions: tuple[int, ...]


class CriterionResult(Frozen):
    """One criterion's verdict; `generator` is None for a criterion pooled across generators."""

    name: str
    generator: str | None
    judged: bool
    passed: bool
    detail: dict[str, float]


class EvalReport(Frozen):
    """An evaluation's config, the code it evaluated, every run's result and the criteria judged on them."""

    config: EvalConfig
    stamps: Stamps
    results: tuple[ReplicateResult, ...]
    criteria: tuple[CriterionResult, ...]

    @model_validator(mode="after")
    def _check_stamps(self) -> EvalReport:
        """Every result evaluated the report's code."""
        if any(r.stamps != self.stamps for r in self.results):
            raise ValueError("every result must carry the report's stamps")
        return self

    @property
    def passed(self) -> bool:
        """True when every judged criterion passed."""
        return all(c.passed for c in self.criteria if c.judged)


# --- the pieces of a run ---


def run_dir(scratch: Path, generator: str, replicate: int, strategy: str, seed: int) -> Path:
    """The directory of one run under `scratch`: its public workspace (`public/`) and private root (`private/`)."""
    return scratch / f"{generator}-r{replicate}-{strategy}-s{seed}"


def feasible_items(
    items: Sequence[WantedItem], fulfilled: Collection[str], current_issue: int, weeks: int
) -> list[WantedItem]:
    """The items the custodian would still fulfill, in list order: not fulfilled, and starting after
    `current_issue` but before the world's `weeks` end."""
    return [
        item
        for item in items
        if item.id not in fulfilled and current_issue + 1 <= _start(item.candidate.stream) < weeks
    ]


def heldout_arms(base: StreamSpec, evaluate_at: int) -> list[StreamSpec]:
    """Per jurisdiction of `base`, the arms nobody observes: schools closed for three weeks and 30% vaccinated,
    both from week `evaluate_at + 1`, the first issue they would be published in.

    Ids follow the custodian's `<parent>~<item id>`, with item ids `heldout-close-<j>` and `heldout-vacc-<j>`.
    """
    start = evaluate_at + 1
    arms = []
    for jurisdiction in base.jurisdictions:
        interventions = {
            "close": Intervention(
                variable=SCHOOLS_CLOSED,
                value=1.0,
                start_week=start,
                end_week=start + GRID_CLOSURE_WEEKS - 1,
                jurisdictions=(jurisdiction,),
            ),
            "vacc": Intervention(
                variable=VACCINATE_FRACTION, value=GRID_VACCINATE_VALUE, start_week=start, jurisdictions=(jurisdiction,)
            ),
        }
        for kind, intervention in interventions.items():
            arm_id = f"{base.id}~{HELDOUT}-{kind}-{jurisdiction}"
            fields = {"id": arm_id, "intervention": intervention, "parent": base.id, "first_issue": start}
            arms.append(StreamSpec.model_validate(base.model_dump() | fields))
    return arms


def heldout_targets(arm: StreamSpec, mission: Mission, evaluate_at: int) -> list[Target]:
    """The arm's targets at reference week `evaluate_at + 1` in the jurisdictions its intervention acts in,
    at the mission's horizons and lag (R15).

    Built directly, not by `mission.targets`: the week may lie past the mission's last issue.
    """
    assert arm.intervention is not None  # an arm carries its intervention
    return [
        Target(
            stream=arm.id,
            jurisdiction=jurisdiction,
            reference_week=evaluate_at + 1,
            horizon=horizon,
            resolution_lag=mission.resolution_lag,
        )
        for jurisdiction in arm.jurisdictions
        if _acts_in(arm.intervention, jurisdiction)
        for horizon in sorted(set(mission.horizons))
    ]


def heldout_realized(world: World, arm: StreamSpec, targets: Sequence[Target]) -> list[int]:
    """What the arm's resolution vintages would report for `targets`, asked privately of the world.

    A target resolves in the vintage of issue `target_week + resolution_lag`, which may lie past the world's
    last week; its realized value is that vintage's row for the target's jurisdiction and week.
    """
    vintages: dict[int, dict[tuple[str, int], float]] = {}
    for issue in sorted({t.resolution_issue for t in targets}):
        vintages[issue] = {(r.jurisdiction, r.week): r.value for r in world.vintage(arm, issue).rows}
    return [int(vintages[t.resolution_issue][t.jurisdiction, t.target_week]) for t in targets]


def _acts_in(intervention: Intervention, jurisdiction: str) -> bool:
    """True when `intervention` applies to `jurisdiction` (an intervention listing none applies everywhere)."""
    return intervention.jurisdictions is None or jurisdiction in intervention.jurisdictions


def _start(spec: StreamSpec) -> int:
    assert spec.intervention is not None  # a candidate stream carries its intervention
    return spec.intervention.start_week


def _at(issue: int) -> datetime:
    """The platform time issue `issue` is published at."""
    return T0 + issue * ISSUE_INTERVAL


def stamps(cfg: EvalConfig) -> Stamps:
    """The stamps of `cfg` as things stand: each of `cfg.programs` as frozen in the public registry, whose
    directory must still hash to its freeze, the scorer's and this package's hashes, and each generator's.

    EvaluationError when a program is not frozen in the public registry or its directory changed since, or
    a generator of `cfg` is not installed.
    """
    registry = Registry(Workspace(cfg.public_root))
    programs = {name: _public_freeze(registry, cfg, name) for name in cfg.programs}
    for name, frozen in programs.items():
        try:
            registry.verify(frozen)
        except RegistryError as e:
            raise EvaluationError(f"program {name!r} in {cfg.public_root}: {e}") from e
    return _code(cfg, programs)


def _installed(cfg: EvalConfig) -> dict[str, Generator]:
    """`cfg`'s generators, by name; EvaluationError naming any that is not installed."""
    installed = generators()
    missing = [g for g in cfg.generators if g not in installed]
    if missing:
        raise EvaluationError(f"generators {missing} are not installed; the installed ones are {sorted(installed)}")
    return {g: installed[g] for g in cfg.generators}


def _code(cfg: EvalConfig, programs: dict[str, str]) -> Stamps:
    """`Stamps` of `programs` and of the scorer, this package and `cfg`'s generators as installed now."""
    sources = {g: package_hash(generator_source(generator)) for g, generator in _installed(cfg).items()}
    return Stamps(programs=programs, scorer=scorer_hash(), lab=lab_hash(), generators=sources)


def _public_freeze(registry: Registry, cfg: EvalConfig, name: str) -> str:
    """The closure hash `name` is frozen under in the public registry; EvaluationError when it is not frozen."""
    freeze = registry.by_name(name)
    if freeze is None:
        raise EvaluationError(f"program {name!r} is not frozen in the public registry of {cfg.public_root}")
    return freeze.closure_hash


def _workspace(cfg: EvalConfig, root: Path) -> tuple[Workspace, dict[str, str]]:
    """A fresh public workspace holding `cfg.programs`, frozen `FREEZE_LEAD` before `T0` in the listed order
    (R7), and each program's closure hash by name.

    Each copy must hash to the program's freeze in the public registry, so the run evaluates exactly the
    frozen code: EvaluationError when a program is not frozen there or its copy hashes to something else.
    """
    public = Registry(Workspace(cfg.public_root))
    ws = Workspace.create(root)
    source = Workspace(cfg.public_root).programs
    hashes: dict[str, str] = {}
    for name in cfg.programs:
        frozen = _public_freeze(public, cfg, name)
        shutil.copytree(source / name, ws.programs / name, ignore=shutil.ignore_patterns("__pycache__"))
        copied = closure_hash(ws.programs / name)
        if copied != frozen:
            raise EvaluationError(
                f"program {name!r} is frozen as {frozen} in {cfg.public_root}, but its copy hashes to {copied}"
            )
        hashes[name] = copied
    registry = Registry(ws)
    for name in cfg.programs:
        registry.freeze(name, T0 - FREEZE_LEAD)
    return ws, hashes


def _standing(
    ws: Workspace, mission: Mission, as_of: int, specs: Mapping[str, StreamSpec]
) -> tuple[Resolution, Advantages, list[ClaimStanding]]:
    """The mission's record, mixture advantages and claim standings as of issue `as_of`."""
    resolution = resolve(ws, mission, as_of)
    advantages = run_mixture(resolution.resolved, mission.tau, mission.half_life_weeks, as_of)
    return resolution, advantages, claim_standings(ws, mission, resolution, specs)


def _decide(
    ws: Workspace,
    mission: Mission,
    strategy: Strategy,
    current: int,
    weeks: int,
    now: datetime,
    rng: np.random.Generator,
) -> list[str]:
    """The ids of the wanted items to fulfill at issue `current`; none when no item is feasible.

    gain ranks every open item (`rank_wanted`), keeps the feasible ones and `choose`s one among them with
    the mission's exploration reserve; random draws one feasible item uniformly; all takes every feasible
    item, in wanted-list order.
    """
    specs = {spec.id: spec for spec in Archive(ws).specs()}
    fulfilled = {spec.fulfills for spec in specs.values() if spec.fulfills is not None}
    feasible = feasible_items(load_wanted(ws, mission.name), fulfilled, current, weeks)
    if not feasible:
        return []
    if strategy == "all":
        return [item.id for item in feasible]
    if strategy == "random":
        return [feasible[int(rng.integers(len(feasible)))].id]
    _, advantages, claims = _standing(ws, mission, current, specs)
    ranked = rank_wanted(ws, mission, current, now, Isolation.NONE, advantages, claims)
    ids = {item.id for item in feasible}
    return [choose([r for r in ranked if r.item in ids], mission.exploration_reserve, rng).item]


def _decides_at(cfg: EvalConfig, strategy: Strategy, issue: int) -> bool:
    """Whether a run of `strategy` decides at `issue`: at every decision issue, or for all at the first only."""
    if strategy == "all":
        return issue == min(cfg.decision_issues, default=None)
    return issue in cfg.decision_issues


def coverages(
    resolution: Resolution, specs: Mapping[str, StreamSpec], base: str, program: str | None
) -> tuple[Calibration, Calibration]:
    """The calibration of `program` (a closure hash, None for none) on the arms' intervened units and on the
    baseline's same units (R15).

    Arm points are its scores on resolved targets of arm streams in a jurisdiction the arm's intervention
    acts in (an arm's targets already start at its intervention).  Baseline points are its scores on the
    resolved targets of `base` with the (jurisdiction, reference week, horizon) of one of those arm points,
    so the two sides pair exactly.
    """
    if program is None:
        return calibration([]), calibration([])
    arm: list[PointScore] = []
    units: set[tuple[str, int, int]] = set()
    scored = [k for k in resolution.resolved if program in k.scores]
    for k in scored:
        intervention = specs[k.target.stream].intervention
        if intervention is not None and _acts_in(intervention, k.target.jurisdiction):
            arm.append(k.scores[program])
            units.add(_unit(k.target))
    baseline = [
        k.scores[program]
        for k in scored
        if k.target.stream == base and _unit(k.target) in units
    ]
    return calibration(arm), calibration(baseline)


def _unit(target: Target) -> tuple[str, int, int]:
    """What pairs an arm target with a baseline target: (jurisdiction, reference week, horizon)."""
    return target.jurisdiction, target.reference_week, target.horizon


def _forecasts(
    ws: Workspace,
    mission: Mission,
    view: ArchiveView,
    arm: StreamSpec,
    targets: Sequence[Target],
    manifests: Mapping[str, ProgramManifest],
) -> dict[Target, dict[str, Pmf]]:
    """Each target's forecasts, by closure hash, from every program that conditions on the arm's intervention.

    A program runs once, on the targets in its scope.  A target it fails gets nothing from it, and a program
    whose closure changed after its freeze gives nothing at all.
    """
    forecasts: dict[Target, dict[str, Pmf]] = {target: {} for target in targets}
    for program, manifest in manifests.items():
        mine = [t for t in targets if in_scope(manifest, t, arm)]
        if not mine or not supports(manifest, arm.intervention):
            continue
        try:
            result = run_order(ws, build_order(ws, program, view, mine), Isolation.NONE, mission.run_timeout_s)
        except RegistryError:
            continue
        for r in result.results:
            if isinstance(r, Prediction):
                forecasts[r.target][program] = r.pmf
    return forecasts


def mixture_log_score(
    pmfs: Mapping[str, Pmf], advantages: Advantages, tau: float, epsilon: float, realized: int, support_max: int
) -> float:
    """The floored log score at `realized` of the mixture of `pmfs`, the awake programs' forecasts of one target.

    The awake programs weigh `exp(current advantage / tau)`; a program not in `pmfs` gets no weight whatever
    its advantage.  The score is under `(1 - epsilon) * mixture + epsilon * Uniform{0..support_max}`.
    """
    weights = normalized_weights(advantages.current, pmfs, tau)
    mixture = mix([pmfs[p] for p in weights], list(weights.values()))
    return floored_log_prob(mixture, realized, epsilon, support_max)


def _heldout(
    ws: Workspace,
    world: World,
    mission: Mission,
    base: StreamSpec,
    advantages: Advantages,
    evaluate_at: int,
    now: datetime,
) -> tuple[list[tuple[Target, float]], int]:
    """(the mixture's (target, score) on each held-out target some program forecast, in `target_order`;
    the number of targets nobody forecast).

    Every program active at `now` forecasts each arm from the archive as of `evaluate_at` plus the arm, and
    each target is scored by `mixture_log_score`.  EvaluationError when no target is forecast at all.
    """
    archive, registry = Archive(ws), Registry(ws)
    view = archive.view([s.id for s in mission_streams(mission, archive.specs())], evaluate_at, now)
    manifests = {h: registry.manifest(h) for h in sorted(registry.freezes()) if registry.active(h, now)}
    scores: list[tuple[Target, float]] = []
    skipped = 0
    for arm in heldout_arms(base, evaluate_at):
        targets = heldout_targets(arm, mission, evaluate_at)
        forecasts = _forecasts(ws, mission, view.with_candidate(arm), arm, targets, manifests)
        assert arm.support_max is not None  # a count stream's arm is bounded like its parent
        for target, realized in zip(targets, heldout_realized(world, arm, targets), strict=True):
            if forecasts[target]:
                score = mixture_log_score(forecasts[target], advantages, mission.tau, mission.epsilon, realized, arm.support_max)
                scores.append((target, score))
            else:
                skipped += 1
    if not scores:
        raise EvaluationError(f"no program forecast any of the {skipped} held-out targets")
    return sorted(scores, key=lambda pair: target_order(pair[0])), skipped


# --- a run ---


def run_replicate(
    cfg: EvalConfig, generator: str, replicate: int, strategy: str, seed: int, scratch: Path
) -> ReplicateResult:
    """Play one run in its own directory under `scratch` (`run_dir`) and read it at `cfg.evaluate_at`.

    The world draws from `[master_seed, generator index, replicate]`, so every run of a replicate sees the
    same world; decisions draw from `[master_seed, generator index, replicate, strategy index, seed]`.
    Each issue is published at `T0 + issue` hours and forecast at once; at a decision issue the chosen items
    are fulfilled then (`all` decides at the first decision issue only).  The result is also written to
    `RESULT_FILE` in the run's directory as soon as it is known, so a finished run survives the evaluation
    that started it.  Then the run's forecast record is deleted: it reaches gigabytes, and like any forecast
    record it is a cache (spec 4.7), here rebuilt by playing the run again, which is deterministic; the run's
    inputs (archive, missions, programs, registry, world) stay.  FileExistsError when the run's directory
    exists; ValueError for a generator not in `cfg.generators` or an unknown strategy; EvaluationError when
    a program is not its freeze in the public registry (`_workspace`).
    """
    if generator not in cfg.generators:
        raise ValueError(f"generator {generator!r} is not one of {list(cfg.generators)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; the strategies are {list(STRATEGIES)}")
    run = cast(Strategy, strategy)
    _installed(cfg)  # refused before the run's directory exists
    index = cfg.generators.index(generator)
    root = run_dir(scratch, generator, replicate, run, seed)
    root.mkdir(parents=True)  # never another run's directory
    ws, programs = _workspace(cfg, root / "public")
    custodian = Custodian(ws, root / "private", log=None)
    world_id = f"{generator}-r{replicate}"
    world_rng = np.random.default_rng([cfg.master_seed, index, replicate])
    world = custodian.create(
        generator, world_id, cfg.weeks, cfg.jurisdictions, CHANNEL, T0, rng=world_rng, required_isolation=Isolation.NONE
    )
    mission = load_mission(ws, world_id)
    (base_id,) = [s for s in mission.streams if Archive(ws).spec(s).kind is StreamKind.COUNT]  # the base channel
    rng = np.random.default_rng([cfg.master_seed, index, replicate, STRATEGIES.index(run), seed])

    fulfilled: list[str] = []
    skipped: list[int] = []
    for issue in range(1, cfg.evaluate_at + 1):
        now = _at(issue)
        custodian.tick(world_id, issue, now)
        run_origin(ws, mission, issue, Isolation.NONE, now)
        if not _decides_at(cfg, run, issue):
            continue
        items = _decide(ws, mission, run, custodian.current_issue(world_id), cfg.weeks, now, rng)
        if not items:
            skipped.append(issue)
        for item in items:
            custodian.fulfill(world_id, item, now)
            fulfilled.append(item)

    specs = {spec.id: spec for spec in Archive(ws).specs()}
    resolution, advantages, claims = _standing(ws, mission, cfg.evaluate_at, specs)
    registry = Registry(ws)
    names = {h: freeze.name for h, freeze in registry.freezes().items()}
    truth = world.truth()
    match = registry.by_name(matching_program(truth))
    arm, baseline = coverages(resolution, specs, base_id, None if match is None else match.closure_hash)
    scores, unscored = _heldout(ws, world, mission, specs[base_id], advantages, cfg.evaluate_at, _at(cfg.evaluate_at))
    weights = normalized_weights(advantages.cumulative, advantages.cumulative, mission.tau)
    by_name = {names[h]: a for h, a in advantages.cumulative.items()}
    theories = {p: freeze.closure_hash for p in SEIR_THEORIES if (freeze := registry.by_name(p)) is not None}
    result = ReplicateResult(
        generator=generator,
        replicate=replicate,
        strategy=run,
        seed=seed,
        stamps=_code(cfg, programs),
        heldout_scores=tuple(scores),
        heldout_log_score=float(np.mean([score for _, score in scores])),
        heldout_targets=len(scores),
        heldout_skipped=unscored,
        weights_cumulative={names[h]: w for h, w in weights.items()},
        theory_advantages={p: by_name[p] for p in theories if p in by_name},
        theory_share=matching_share(by_name, theories, truth, mission.tau),
        theory_predicted={p: resolution.predicted.get(h, 0) for p, h in theories.items()},
        theory_unscored={p: resolution.unscored.get(h, 0) for p, h in theories.items()},
        claims={c.claim: c.verdict for c in claims},
        truth=truth,
        arm_coverage90=arm.coverage_90,
        arm_points=arm.n,
        baseline_coverage90=baseline.coverage_90,
        baseline_points=baseline.n,
        fulfilled=tuple(fulfilled),
        skipped_decisions=tuple(skipped),
    )
    write_new(root / RESULT_FILE, result)
    shutil.rmtree(ws.scores)
    return result


# --- the criteria ---


def _mean(values: Iterable[float]) -> float:
    """The mean of `values` (at least one), summed exactly (`math.fsum`), so it does not depend on their order."""
    values = list(values)
    return math.fsum(values) / len(values)


def _gain_beats_random(gain: ReplicateResult, random: Sequence[ReplicateResult]) -> bool | None:
    """Whether gain's mean held-out score exceeds the mean of the random seeds' means, on the targets every one
    of them scored; None when there is no random run or no such target.

    The common targets are taken in `target_order` and summed exactly, so the verdict depends on nothing
    but the scores (not on set order, which follows the hash seed).  A tie is not a win.
    """
    if not random:
        return None
    scored = [dict(r.heldout_scores) for r in (gain, *random)]
    common = sorted(set.intersection(*(set(s) for s in scored)), key=target_order)
    if not common:
        return None
    gain_mean = _mean(scored[0][t] for t in common)
    random_mean = _mean(_mean(s[t] for t in common) for s in scored[1:])
    return gain_mean > random_mean


def _gain_vs_random(generator: str, runs: Sequence[ReplicateResult], t: Thresholds) -> CriterionResult:
    """Replicates where gain's held-out score exceeds the mean of its random seeds' on their common targets."""
    random: defaultdict[int, list[ReplicateResult]] = defaultdict(list)
    for r in runs:
        if r.strategy == "random":
            random[r.replicate].append(r)
    outcomes = [_gain_beats_random(r, random[r.replicate]) for r in runs if r.strategy == "gain"]
    beats = sum(outcome is True for outcome in outcomes)
    return CriterionResult(
        name="gain-vs-random",
        generator=generator,
        judged=True,
        passed=beats >= t.gain_beats_random_min,
        detail={
            "beats": beats,
            "replicates": len(outcomes),
            "incomparable": sum(outcome is None for outcome in outcomes),
            "min": t.gain_beats_random_min,
        },
    )


def _concentration(
    name: str, generator: str, runs: Sequence[ReplicateResult], judged: bool, t: Thresholds
) -> CriterionResult:
    """Runs where the theory matching the truth holds at least `concentration_share` of the theories'
    restricted weight; a run whose share is undefined is not concentrated, and is counted."""
    concentrated = sum(r.theory_share is not None and r.theory_share >= t.concentration_share for r in runs)
    return CriterionResult(
        name=name,
        generator=generator,
        judged=judged,
        passed=concentrated >= t.concentration_min,
        detail={
            "concentrated": concentrated,
            "runs": len(runs),
            "undefined": sum(r.theory_share is None for r in runs),
            "min": t.concentration_min,
        },
    )


def _pooled(coverages: Iterable[tuple[float | None, int]]) -> tuple[float | None, int]:
    """(coverage over all the points, number of points) from per-run (coverage, points); None for no points.

    Coverage is a mean over points, so weighting each run by its points pools them exactly.
    """
    pairs = [(c, n) for c, n in coverages if c is not None and n > 0]
    points = sum(n for _, n in pairs)
    return (sum(c * n for c, n in pairs) / points if points else None), points


def _surgery(generator: str, runs: Sequence[ReplicateResult], t: Thresholds) -> CriterionResult:
    """The matching theory's 90% coverage on arms is within `surgery_tolerance` of its coverage on the baseline,
    pooled over every run of either strategy (it is about the program, not how the arms were chosen)."""
    arm, arm_points = _pooled((r.arm_coverage90, r.arm_points) for r in runs)
    baseline, baseline_points = _pooled((r.baseline_coverage90, r.baseline_points) for r in runs)
    detail = {"arm_points": arm_points, "baseline_points": baseline_points, "tolerance": t.surgery_tolerance}
    passed = False
    if arm is not None:
        detail["arm_coverage90"] = arm
    if baseline is not None:
        detail["baseline_coverage90"] = baseline
    if arm is not None and baseline is not None:
        detail["difference"] = arm - baseline
        passed = abs(arm - baseline) <= t.surgery_tolerance
    return CriterionResult(
        name="surgery", generator=generator, judged=generator in SURGERY_JUDGED, passed=passed, detail=detail
    )


def _share_with(runs: Sequence[ReplicateResult], claim: str, verdict: Verdict) -> float | None:
    """The share of `runs` whose `claim` has `verdict`; None for no runs."""
    return sum(r.claims.get(claim) == verdict for r in runs) / len(runs) if runs else None


def _power(gain: Sequence[ReplicateResult], t: Thresholds) -> CriterionResult:
    """The x claim is supported where x acts and contradicted where it does not; the flat claim abstains."""
    with_x = [r for r in gain if r.truth[X_CLAIM]]
    without_x = [r for r in gain if not r.truth[X_CLAIM]]
    rates = {
        "p_supports_given_x": (_share_with(with_x, X_CLAIM, Verdict.SUPPORTS), t.power_min),
        "p_contradicts_given_no_x": (_share_with(without_x, X_CLAIM, Verdict.CONTRADICTS), t.power_min),
        "p_abstains_flat": (_share_with(gain, FLAT_CLAIM, Verdict.INSUFFICIENT), t.abstain_min),
    }
    detail: dict[str, float] = {"runs_x": len(with_x), "runs_no_x": len(without_x), "runs": len(gain)}
    detail |= {name: rate for name, (rate, _) in rates.items() if rate is not None}
    passed = all(rate is not None and rate >= least for rate, least in rates.values())
    return CriterionResult(name="power", generator=None, judged=True, passed=passed, detail=detail)


def criteria(cfg: EvalConfig, results: Sequence[ReplicateResult]) -> tuple[CriterionResult, ...]:
    """The pre-registered criteria: per generator gain-vs-random, concentration (judged on the `all` runs:
    spec 6.3 asks for it "even with unlimited arms"), the gain runs' budget-limited concentration
    (`concentration-gain`, reported, not judged) and surgery (every run of every strategy); then power over
    every generator's gain runs."""
    t = cfg.thresholds
    judged: list[CriterionResult] = []
    for generator in cfg.generators:
        runs = [r for r in results if r.generator == generator]
        by_strategy = {s: [r for r in runs if r.strategy == s] for s in STRATEGIES}
        judged += [
            _gain_vs_random(generator, runs, t),
            _concentration("concentration", generator, by_strategy["all"], True, t),
            _concentration("concentration-gain", generator, by_strategy["gain"], False, t),
            _surgery(generator, runs, t),
        ]
    judged.append(_power([r for r in results if r.strategy == "gain"], t))
    return tuple(judged)


# --- the evaluation ---


def runs(cfg: EvalConfig) -> list[tuple[str, int, Strategy, int]]:
    """Every run of `cfg` as (generator, replicate, strategy, seed), in report order: per generator and
    replicate, the gain run (seed 0), then each random seed, then the all run (seed 0)."""
    seeds: dict[Strategy, int] = {"gain": 1, "random": cfg.random_seeds, "all": 1}
    return [
        (generator, replicate, strategy, seed)
        for generator in cfg.generators
        for replicate in range(cfg.replicates)
        for strategy in STRATEGIES
        for seed in range(seeds[strategy])
    ]


@contextmanager
def worker_environment() -> Iterator[None]:
    """`WORKER_ENVIRONMENT` in `os.environ` while the context lasts; afterwards every variable is as it was."""
    saved = {name: os.environ.get(name) for name in WORKER_ENVIRONMENT}
    os.environ.update(WORKER_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _finished(scratch: Path, run: tuple[str, int, Strategy, int], current: Stamps) -> ReplicateResult:
    """The result a finished run left in its directory; EvaluationError when it is not that run's, or it
    evaluated other code than `current`."""
    path = run_dir(scratch, *run) / RESULT_FILE
    result = read_model(path, ReplicateResult)
    if (result.generator, result.replicate, result.strategy, result.seed) != run:
        raise EvaluationError(f"{path} holds the result of another run")
    _require_stamps(path, result.stamps, current)
    return result


def _require_stamps(source: Path, found: Stamps, current: Stamps) -> None:
    """EvaluationError naming what differs when the result from `source` evaluated other code than `current`."""
    differing = [field for field in Stamps.model_fields if getattr(found, field) != getattr(current, field)]
    if differing:
        raise EvaluationError(
            f"{source} evaluated other code than the current one: its {', '.join(differing)} stamps differ"
        )


def evaluate(cfg: EvalConfig, scratch: Path) -> EvalReport:
    """Run every run of `cfg` in `cfg.workers` processes, each in its own directory under `scratch`, and judge them.

    The evaluation is recoverable: a run whose directory holds its result (`RESULT_FILE`) is not run again,
    its result is read back.  `scratch` records the config (`CONFIG_FILE`) so that results are reused only
    by the same evaluation (its `machine_fields` may differ), and only when they evaluated the same code
    (`stamps`).  Before anything runs or is written: EvaluationError when a program is not its freeze in
    the public registry; WriteOnceError when `scratch` holds another config's runs, results with no
    recorded config (they are never adopted), or a run directory without a result (an interrupted run);
    EvaluationError when a run directory holds another run's result or one with other stamps.  Workers are
    spawned, not forked (each is a fresh interpreter, never a copy of a process holding threads), in
    `worker_environment`.  A run that raises propagates once every run has ended, and EvaluationError
    follows when the code changed while the runs were going.
    """
    current = stamps(cfg)
    recorded = scratch / CONFIG_FILE
    here = {name: getattr(cfg, name) for name in machine_fields()}
    if recorded.is_file() and read_model(recorded, EvalConfig).model_copy(update=here) != cfg:
        raise WriteOnceError(f"{scratch} holds the runs of another evaluation config")
    finished = [run for run in runs(cfg) if (run_dir(scratch, *run) / RESULT_FILE).is_file()]
    if finished and not recorded.is_file():
        raise WriteOnceError(
            f"refusing to adopt {len(finished)} results under {scratch}: no evaluation config is recorded there"
        )
    taken = [run_dir(scratch, *run).name for run in runs(cfg) if run not in finished and run_dir(scratch, *run).exists()]
    if taken:
        raise WriteOnceError(
            f"refusing to reuse {len(taken)} run directories without a result under {scratch}, e.g. {taken[0]}"
        )
    results = {run: _finished(scratch, run, current) for run in finished}
    if not recorded.is_file():
        write_new(recorded, cfg)
    spawn = multiprocessing.get_context("spawn")
    with worker_environment(), ProcessPoolExecutor(max_workers=cfg.workers, mp_context=spawn) as pool:
        futures = {run: pool.submit(run_replicate, cfg, *run, scratch) for run in runs(cfg) if run not in results}
        results |= {run: future.result() for run, future in futures.items()}
    for run in futures:
        _require_stamps(run_dir(scratch, *run) / RESULT_FILE, results[run].stamps, current)
    ordered = tuple(results[run] for run in runs(cfg))
    return EvalReport(config=cfg, stamps=current, results=ordered, criteria=criteria(cfg, ordered))
