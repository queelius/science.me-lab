"""The custodian: the only bridge from a world's private truth to the public archive.

A world's configuration (`worlds/<id>/config.json`, its `World.dump`) and anything its generator caches
(`worlds/<id>/cache/`) stay under the private root.  The custodian publishes only what a surveillance
system would show: count vintages, the covariate x released ahead, and the school calendar, each asked of
the world through the generator protocol (`World.vintage`).  Every capture is published at the `now` it is
given, which is after any program frozen earlier, so nothing a program was frozen on can have seen the
data, and nothing about the world's parameters is written to the public workspace.

With a commitment log the custodian seals before it publishes: `create` seals the generator's current
source (when that version is not sealed yet) and then the world's configuration, before writing either
privately or anything publicly; `tick`, `fulfill` and `add_channel` seal the generator's source first
whenever it has changed since its last seal.  The nonces stay under the private root (`commitments/`).
`reveal` opens a world's seal once the world has published its last issue.

The custodian reads no clock: every operation takes `now`.
"""

from __future__ import annotations

import secrets
import shutil
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from scienceme.archive import Archive
from scienceme.mission import (
    Candidate,
    Claim,
    Mission,
    WantedItem,
    append_wanted,
    list_missions,
    load_mission,
    load_wanted,
    save_mission,
)
from scienceme.model import Capture, Intervention, Isolation, StreamKind, StreamSpec, TargetFilter, Vintage
from scienceme.worker import package_hash
from scienceme.workspace import Workspace

from scienceme_lab.commitments import Commitment, CommitmentLog, Reveal, Subject, SubjectKind
from scienceme_lab.generator import (
    COVARIATE_X_ROLE,
    FLAT_CLAIM,
    SCHOOL_CLAIM,
    SCHOOL_TERM_ROLE,
    SCHOOLS_CLOSED,
    SUPPORT_MAX,
    VACCINATE_FRACTION,
    X_CLAIM,
    Generator,
    GeneratorError,
    World,
    generator_of,
    generator_source,
    generators,
    source_name,
)

X_STREAM = "x"
TERM_STREAM = "term"

# The mission template.
CAPTAIN = "llm-captain"
HORIZONS = (0, 1, 2, 3)
FIRST_ISSUE = 4
RESOLUTION_LAG = 4
EPSILON = 0.01
HALF_LIFE_WEEKS = 8.0
COVERAGE_MIN = 0.8
CLAIM_THRESHOLD_NATS = 3.0
REGIME_FLAG_TV = 0.5
EXPLORATION_RESERVE = 0.2

# The wanted grid: per jurisdiction and start week, a three-week closure and a 30% vaccination.
GRID_PROPOSER = "custodian-grid"
GRID_STARTS = (8, 12, 16, 20)
GRID_CLOSURE_WEEKS = 3
GRID_VACCINATE_VALUE = 0.3
GRID_COST = 1.0

CONFIG_FILE = "config.json"
REVEALED_WORLDS = Path("revealed") / "worlds"  # where a world's configuration is revealed, in the log's directory
_SEAL_DIGITS = 12  # hex digits of a generator's package hash in its nonce file's name


class CustodianError(Exception):
    """Raised when the custodian refuses an operation on a world."""


def _count_spec(world: World, channel: str, **fields: object) -> StreamSpec:
    """The base COUNT stream of `channel` in the world; `fields` add to or override the defaults."""
    values: dict[str, object] = {
        "id": f"{world.id}/{channel}",
        "world": world.id,
        "role": channel,
        "kind": StreamKind.COUNT,
        "jurisdictions": world.jurisdictions,
        "support_max": SUPPORT_MAX,
    }
    return StreamSpec.model_validate(values | fields)


def _covariate_spec(world: World, name: str, role: str) -> StreamSpec:
    return StreamSpec(
        id=f"{world.id}/{name}", world=world.id, role=role, kind=StreamKind.COVARIATE, jurisdictions=world.jurisdictions
    )


def _mission(world: World, channel: str, required_isolation: Isolation) -> Mission:
    """The mission template of a world observed through `channel`.

    Its streams are every base stream of the world: the channel, the covariate x and the school term.  The
    views programs forecast from are built from the mission's streams, so a program can condition on the
    covariates only if they are listed (R16); a covariate stream has no targets of its own.
    """
    varying = tuple(j for j in world.jurisdictions if j != world.flat_jurisdiction)
    return Mission(
        name=world.id,
        captain=CAPTAIN,
        streams=(f"{world.id}/{channel}", f"{world.id}/{X_STREAM}", f"{world.id}/{TERM_STREAM}"),
        horizons=HORIZONS,
        first_issue=FIRST_ISSUE,
        last_issue=world.weeks - max(HORIZONS) - RESOLUTION_LAG,  # the last target resolves at issue `weeks`
        resolution_lag=RESOLUTION_LAG,
        epsilon=EPSILON,
        tau=float(len(world.jurisdictions)),
        half_life_weeks=HALF_LIFE_WEEKS,
        coverage_min=COVERAGE_MIN,
        claim_threshold_nats=CLAIM_THRESHOLD_NATS,
        regime_flag_tv=REGIME_FLAG_TV,
        exploration_reserve=EXPLORATION_RESERVE,
        required_isolation=required_isolation,
        claims=(
            Claim(
                id=X_CLAIM,
                statement="The covariate x lowers transmission, so conditioning on it improves forecasts.",
                scope=TargetFilter(roles=(channel,), jurisdictions=varying),
                weight=0.5,
            ),
            Claim(
                id=FLAT_CLAIM,
                statement="The covariate x improves forecasts where it is identically zero (a negative control).",
                scope=TargetFilter(roles=(channel,), jurisdictions=(world.flat_jurisdiction,)),
                weight=0.0,
                refines=X_CLAIM,
            ),
            Claim(
                id=SCHOOL_CLAIM,
                statement="School term raises transmission, so closing schools lowers the counts.",
                scope=TargetFilter(roles=(channel,)),
                weight=1.0,
            ),
        ),
    )


def _arm_spec(parent: StreamSpec, arm_id: str, intervention: Intervention, **fields: object) -> StreamSpec:
    """The stream of `parent` under `intervention`: everything but the identity is the parent's."""
    values = parent.model_dump() | {"id": arm_id, "intervention": intervention, "parent": parent.id}
    return StreamSpec.model_validate(values | fields)


def _grid(mission: Mission, base: StreamSpec) -> list[WantedItem]:
    """The wanted grid: for each jurisdiction and start, a school closure and a vaccination item.

    Candidate targets come from `mission.targets` at `reference_week = start`, so they carry the
    mission's horizons and resolution lag, and a start outside the mission window has none.
    """
    items: list[WantedItem] = []
    for jurisdiction in base.jurisdictions:
        for start in GRID_STARTS:
            closure = Intervention(
                variable=SCHOOLS_CLOSED,
                value=1.0,
                start_week=start,
                end_week=start + GRID_CLOSURE_WEEKS - 1,
                jurisdictions=(jurisdiction,),
            )
            vaccination = Intervention(
                variable=VACCINATE_FRACTION, value=GRID_VACCINATE_VALUE, start_week=start, jurisdictions=(jurisdiction,)
            )
            for kind, intervention in (("close", closure), ("vacc", vaccination)):
                item_id = f"{kind}-{jurisdiction}-w{start}"
                arm = _arm_spec(base, f"{base.id}~{item_id}", intervention)
                targets = tuple(mission.targets(arm, start))
                items.append(
                    WantedItem(
                        id=item_id,
                        proposed_by=GRID_PROPOSER,
                        candidate=Candidate(stream=arm, targets=targets),
                        cost=GRID_COST,
                    )
                )
    return items


def _check_intervention(generator: Generator, world: World, intervention: Intervention, current: int) -> None:
    """Refuse what the generator would not simulate or would mis-handle (`current` is the latest issue).

    The variable must be one the generator simulates; the value a fraction in [0, 1]; every listed
    jurisdiction in the world; `end_week`, when given, not before `start_week`; and the start after the
    current issue and before the world's last week.
    """
    if intervention.variable not in generator.interventions:
        raise CustodianError(
            f"the {generator.name} generator does not simulate {intervention.variable!r}; "
            f"it simulates {sorted(generator.interventions)}"
        )
    if not 0.0 <= intervention.value <= 1.0:
        raise CustodianError(f"value {intervention.value} is not a fraction in [0, 1]")
    unknown = [j for j in intervention.jurisdictions or () if j not in world.jurisdictions]
    if unknown:
        raise CustodianError(f"unknown jurisdictions {unknown}; the world has {list(world.jurisdictions)}")
    if intervention.end_week is not None and intervention.end_week < intervention.start_week:
        raise CustodianError(f"end week {intervention.end_week} is before start week {intervention.start_week}")
    if intervention.start_week < current + 1:
        raise CustodianError(
            f"start week {intervention.start_week} is not after the current issue {current}; "
            f"the earliest start is week {current + 1}"
        )
    if intervention.start_week >= world.weeks:
        raise CustodianError(f"start week {intervention.start_week} is not before the world's {world.weeks} weeks end")


class Custodian:
    """Holds the private truth of the worlds under `private_root` and publishes their vintages to `public`.

    `log` is the commitment log it seals into, or None for a simulation run that seals nothing;
    `generators` are the generators it can draw and load worlds with (default: every installed one).
    """

    def __init__(
        self,
        public: Workspace,
        private_root: Path,
        log: CommitmentLog | None,
        generators: Mapping[str, Generator] | None = None,
    ) -> None:
        self._public = public
        self._private = private_root
        self._archive = Archive(public)
        self._log = log
        self._generators = dict(generators) if generators is not None else _installed()

    # --- creating a world ---

    def create(
        self,
        generator: str,
        world_id: str,
        weeks: int,
        n_jurisdictions: int,
        channel: str,
        now: datetime,
        rng: np.random.Generator | None = None,
        required_isolation: Isolation = Isolation.CONTAINER,
    ) -> World:
        """Draw a world, seal it, keep its truth privately, and publish its streams, mission and wanted grid.

        With no `rng` the draw uses OS entropy, so nobody knows the truth.  The world id becomes the
        mission name and the stream-id prefix, so it must be one stream-id segment.  Publishes the
        whole school calendar at issue 0; count and covariate issues follow with `tick`.
        """
        maker = self._generator(generator)
        if weeks < 1 or n_jurisdictions < 1:
            raise CustodianError(f"a world needs at least one week and one jurisdiction, not {weeks} and {n_jurisdictions}")
        if "/" in world_id:
            raise CustodianError(f"{world_id!r} is not a valid world id: it must not contain '/'")
        if rng is None:
            rng = np.random.default_rng(secrets.randbits(128))
        world = maker.draw(world_id, weeks, n_jurisdictions, rng, self._cache(world_id))
        if channel not in world.channels:
            raise CustodianError(f"channel {channel!r} is not observable; the channels are {sorted(world.channels)}")
        try:
            base = _count_spec(world, channel)
            x_spec = _covariate_spec(world, X_STREAM, COVARIATE_X_ROLE)
            term_spec = _covariate_spec(world, TERM_STREAM, SCHOOL_TERM_ROLE)
        except ValidationError as e:
            raise CustodianError(f"{world_id!r} is not a valid world id: {_describe(e)}") from e
        if (
            self._config_path(world_id).exists()
            or world_id in list_missions(self._public)
            or any(spec.world == world_id or spec.id.startswith(f"{world_id}/") for spec in self._archive.specs())
        ):
            raise CustodianError(f"world {world_id!r} already exists")
        if self._log is not None and self._world_nonce(world_id).exists():
            raise CustodianError(f"world {world_id!r} was sealed but never written; choose another id")

        mission = _mission(world, channel, required_isolation)
        grid = _grid(mission, base)  # everything that can fail is built before the first write
        term = world.vintage(term_spec, 0)
        config = world.dump()
        self._seal_generator(maker, now)
        if self._log is not None:
            path = self._config_path(world_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_suffix(".sealing")  # the commitment reads the content from a file
            staged.write_bytes(config)
            try:
                self._log.commit(Subject(kind=SubjectKind.WORLD, name=world_id), staged, self._world_nonce(world_id), now)
            finally:
                staged.unlink()
        self._write_config(world_id, config)
        for spec in (base, x_spec, term_spec):
            self._archive.create_stream(spec)
        self._archive.add(_capture(term, now, world_id))
        save_mission(self._public, mission)
        for item in grid:
            append_wanted(self._public, world_id, item)
        return world

    # --- reading a world ---

    def world(self, world_id: str) -> World:
        """The world `world_id`, loaded from its private configuration."""
        return self._load(world_id)[1]

    def current_issue(self, world_id: str) -> int:
        """The latest issue published for the world's base channel; 0 when none is."""
        return max(self._archive.issues(self._base_spec(world_id).id), default=0)

    # --- publishing ---

    def tick(self, world_id: str, to_issue: int, now: datetime) -> list[Capture]:
        """Publish every stream's issues after its own latest one, up to `to_issue`, all at `now`.

        The streams are the count streams (the base channel, its arms and any added channel) and the
        covariate x.  Each starts from its own latest issue plus one, so a stream an interrupted tick left
        behind is caught up by the next.  `CustodianError` when `to_issue` is past the world's last week.
        Nothing is written until every capture is built.
        """
        maker, world = self._load(world_id)
        if to_issue > world.weeks:
            raise CustodianError(f"world {world_id!r} has {world.weeks} weeks; cannot publish issue {to_issue}")
        captures: list[Capture] = []
        for spec in self._world_specs(world_id):
            if spec.kind is StreamKind.COUNT:
                captures += self._vintages(world, spec, self._unpublished(spec.id, to_issue), now)
        x_spec = self._archive.spec(f"{world_id}/{X_STREAM}")
        captures += self._vintages(world, x_spec, self._unpublished(x_spec.id, to_issue), now)
        captures.sort(key=lambda c: c.vintage.issue)  # stable: ascending issue; within one, count streams by id, then x
        if captures:
            self._seal_generator(maker, now)
        for capture in captures:
            self._archive.add(capture)
        return captures

    def fulfill(self, world_id: str, item_id: str, now: datetime) -> StreamSpec:
        """Run the wanted item's arm and publish it as a stream, with the history up to the current issue.

        The arm is a stream of the item's parent under the item's intervention, with `fulfills` and
        `first_issue = current + 1`.  Its id is always `<parent id>~<item id>`: the proposal's own stream
        id must equal that, so a proposal cannot name a stream outside the world.  `CustodianError` when
        the item is unknown or already fulfilled, its parent is not a base count stream of this world,
        or its intervention is refused by `_check_intervention`.
        """
        maker, world = self._load(world_id)
        item = next((i for i in load_wanted(self._public, world_id) if i.id == item_id), None)
        if item is None:
            raise CustodianError(f"world {world_id!r} has no wanted item {item_id!r}")
        specs = {spec.id: spec for spec in self._world_specs(world_id)}
        if any(spec.fulfills == item_id for spec in specs.values()):
            raise CustodianError(f"wanted item {item_id!r} is already fulfilled")
        proposed = item.candidate.stream
        intervention = proposed.intervention
        assert intervention is not None and proposed.parent is not None  # Candidate and StreamSpec guarantee both
        parent = specs.get(proposed.parent)
        if parent is None or parent.kind is not StreamKind.COUNT or parent.intervention is not None:
            raise CustodianError(f"item {item_id!r}: parent {proposed.parent!r} is not a base count stream of {world_id!r}")
        arm_id = f"{parent.id}~{item_id}"
        if proposed.id != arm_id:
            raise CustodianError(f"item {item_id!r}: stream id {proposed.id!r} must be {arm_id!r}")
        current = self.current_issue(world_id)
        _check_intervention(maker, world, intervention, current)
        arm = _arm_spec(parent, arm_id, intervention, first_issue=current + 1, fulfills=item_id)
        captures = self._vintages(world, arm, range(1, current + 1), now)
        self._seal_generator(maker, now)
        self._archive.create_stream(arm)
        for capture in captures:
            self._archive.add(capture)
        return arm

    def add_channel(self, world_id: str, channel: str, now: datetime) -> StreamSpec:
        """Start publishing the world through another channel, with its history up to the current issue.

        The new base COUNT stream has `first_issue = current + 1`, and its id is appended to the
        mission's streams so that it is forecast from then on.  `CustodianError` when the world cannot be
        observed through the channel, already is, or has no mission.
        """
        maker, world = self._load(world_id)
        if channel not in world.channels:
            raise CustodianError(f"channel {channel!r} is not observable; the channels are {sorted(world.channels)}")
        if any(spec.id == f"{world_id}/{channel}" for spec in self._world_specs(world_id)):
            raise CustodianError(f"world {world_id!r} already has channel {channel!r}")
        try:
            mission = load_mission(self._public, world_id)
        except FileNotFoundError as e:
            raise CustodianError(f"world {world_id!r} has no mission file to add {channel!r} to") from e
        current = self.current_issue(world_id)
        spec = _count_spec(world, channel, first_issue=current + 1)
        captures = self._vintages(world, spec, range(1, current + 1), now)
        self._seal_generator(maker, now)
        self._archive.create_stream(spec)
        for capture in captures:
            self._archive.add(capture)
        save_mission(self._public, Mission.model_validate(mission.model_dump() | {"streams": (*mission.streams, spec.id)}))
        return spec

    # --- revealing ---

    def reveal(self, world_id: str, now: datetime) -> Reveal:
        """Open the world's seal: publish its configuration in the log's directory, with the kept nonce.

        `CustodianError` when there is no log, the world has not published its last issue yet, or it was
        never sealed.
        """
        if self._log is None:
            raise CustodianError("this custodian has no commitment log to reveal into")
        world = self.world(world_id)
        if self.current_issue(world_id) < world.weeks:
            raise CustodianError(
                f"world {world_id!r} is not closed: it is at issue {self.current_issue(world_id)} of {world.weeks}"
            )
        sealed = [
            e for e in self._log.entries()
            if isinstance(e, Commitment) and e.subject == Subject(kind=SubjectKind.WORLD, name=world_id)
        ]
        if not sealed or not self._world_nonce(world_id).is_file():
            raise CustodianError(f"world {world_id!r} was never sealed in {self._log.path}")
        published = self._log.path.parent / REVEALED_WORLDS / world_id / CONFIG_FILE
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._config_path(world_id), published)
        nonce = bytes.fromhex(self._world_nonce(world_id).read_text(encoding="utf-8").strip())
        return self._log.reveal(sealed[-1].seal, nonce, published, now)

    # --- internals ---

    def _generator(self, name: str) -> Generator:
        if name not in self._generators:
            raise CustodianError(f"no generator {name!r} is installed; the generators are {sorted(self._generators)}")
        return self._generators[name]

    def _load(self, world_id: str) -> tuple[Generator, World]:
        path = self._config_path(world_id)
        if not path.is_file():
            raise CustodianError(f"no world {world_id!r} under {self._private}")
        config = path.read_bytes()
        try:
            maker = self._generator(generator_of(config))
            return maker, maker.load(config, self._cache(world_id))
        except GeneratorError as e:
            raise CustodianError(f"world {world_id!r}: {e}") from e

    def _world_dir(self, world_id: str) -> Path:
        return self._private / "worlds" / world_id

    def _config_path(self, world_id: str) -> Path:
        return self._world_dir(world_id) / CONFIG_FILE

    def _cache(self, world_id: str) -> Path:
        return self._world_dir(world_id) / "cache"

    def _world_nonce(self, world_id: str) -> Path:
        return self._private / "commitments" / f"world-{world_id}.nonce"

    def _write_config(self, world_id: str, config: bytes) -> None:
        path = self._config_path(world_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(config)

    def _seal_generator(self, generator: Generator, now: datetime) -> None:
        """Seal the generator's current source, unless that version already has a kept nonce."""
        if self._log is None:
            return
        source = generator_source(generator)
        digest = package_hash(source).removeprefix("sha256:")[:_SEAL_DIGITS]
        keep = self._private / "commitments" / f"generator-{source_name(generator)}-{digest}.nonce"
        if not keep.exists():
            self._log.commit(Subject(kind=SubjectKind.GENERATOR, name=source_name(generator)), source, keep, now)

    def _world_specs(self, world_id: str) -> list[StreamSpec]:
        """Every stream of the world, sorted by id."""
        return [spec for spec in self._archive.specs() if spec.world == world_id]

    def _base_spec(self, world_id: str) -> StreamSpec:
        """The world's base channel: its oldest base COUNT stream (an added channel starts later)."""
        bases = [s for s in self._world_specs(world_id) if s.kind is StreamKind.COUNT and s.intervention is None]
        if not bases:
            raise CustodianError(f"no world {world_id!r} in the public archive")
        return min(bases, key=lambda s: (s.first_issue, s.id))

    def _unpublished(self, stream: str, to_issue: int) -> range:
        """The issues of `stream` after its own latest one, up to `to_issue`."""
        return range(max(self._archive.issues(stream), default=0) + 1, to_issue + 1)

    def _vintages(self, world: World, spec: StreamSpec, issues: Iterable[int], now: datetime) -> list[Capture]:
        """The vintages of `spec` at `issues`, as captures published at `now`."""
        return [_capture(world.vintage(spec, issue), now, world.id) for issue in issues]


def _installed() -> dict[str, Generator]:
    try:
        return generators()
    except GeneratorError as e:
        raise CustodianError(str(e)) from e


def _capture(vintage: Vintage, now: datetime, world_id: str) -> Capture:
    """`vintage` as published by the custodian of `world_id` at `now`."""
    return Capture.make(vintage, now, f"custodian:{world_id}")


def _describe(error: ValueError) -> str:
    """`error` on one line."""
    return " ".join(str(error).split())
