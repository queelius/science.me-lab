"""The custodian: private truth in, public vintages out, always published after the programs froze."""

import shutil
import typing
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from helpers import DEFAULT_PUBLIC, NOW, SEED_PROGRAMS, at
from scienceme.archive import Archive
from scienceme.mission import (
    Candidate,
    WantedItem,
    append_wanted,
    load_mission,
    load_wanted,
    mission_streams,
)
from scienceme.model import (
    Intervention,
    Isolation,
    StreamKind,
    StreamSpec,
    Target,
    TargetFilter,
)
from scienceme.predict import run_origin
from scienceme.record import PredictionRecord, RecordStore
from scienceme.registry import Registry
from scienceme.workspace import Workspace

from scienceme_lab import custodian as custodian_module
from scienceme_lab.commitments import CommitmentLog, SubjectKind, seal
from scienceme_lab.custodian import Custodian, CustodianError
from scienceme_lab.generator import SUPPORT_MAX, World
from scienceme_lab.toy import GENERATOR, ToyConfig

WORLD = "w1"
BASE = f"{WORLD}/admissions"
WEEKS = 30
JURISDICTIONS = 3
GRID_STARTS = (8, 12, 16, 20)
ABM_SEED = 7  # an ABM world whose vaccinated jurisdiction has an outbreak the arm changes


@pytest.fixture
def custodian(public: Workspace, private: Path) -> Custodian:
    return Custodian(public, private, log=None)


def make_world(
    custodian: Custodian,
    world_id: str = WORLD,
    generator: str = "toy",
    weeks: int = WEEKS,
    jurisdictions: int = JURISDICTIONS,
    channel: str = "admissions",
    seed: int = 7,
    **kwargs: typing.Any,
) -> World:
    return custodian.create(
        generator, world_id, weeks, jurisdictions, channel, NOW, rng=np.random.default_rng(seed), **kwargs
    )


def files(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


def propose(
    public: Workspace,
    item_id: str,
    intervention: Intervention,
    parent: str = BASE,
    world: str = WORLD,
    arm_id: str | None = None,
) -> None:
    """Append a hand-made wanted item (an agent's proposal) whose arm is `parent` under `intervention`.

    The arm's stream id is the derived `<parent>~<item id>` unless `arm_id` says otherwise."""
    spec = Archive(public).spec(parent)
    arm = StreamSpec(
        id=arm_id or f"{parent}~{item_id}",
        world=world,
        role=spec.role,
        kind=StreamKind.COUNT,
        jurisdictions=spec.jurisdictions,
        support_max=1 if spec.support_max is None else spec.support_max,
        intervention=intervention,
        parent=parent,
    )
    append_wanted(public, world, WantedItem(id=item_id, proposed_by="test", candidate=Candidate(stream=arm, targets=())))


def rows_by_week(vintage_rows: tuple, before: int | None = None) -> dict:
    return {(r.jurisdiction, r.week): r.value for r in vintage_rows if before is None or r.week < before}


def last_vintage(public: Workspace, stream: str, issue: int):
    """The vintage of `stream` published in `issue`, as the archive shows it."""
    history = Archive(public).view([stream], as_of=issue, published_by=at(10_000)).history(stream)
    (found,) = [v for v in history if v.issue == issue]
    return found


def assert_arm_matches_baseline_before_start(
    public: Workspace, arm: str, start: int, jurisdiction: str, issues: typing.Sequence[int], n_jurisdictions: int
) -> None:
    """In every published issue the arm's rows for weeks before `start` equal the baseline's, in every
    jurisdiction; by the last issue the arm differs from it in `jurisdiction`, and only from `start` on."""
    for issue in issues:
        baseline = rows_by_week(last_vintage(public, BASE, issue).rows, before=start)
        assert rows_by_week(last_vintage(public, arm, issue).rows, before=start) == baseline
        assert len(baseline) == n_jurisdictions * min(start, issue)
    final_baseline = rows_by_week(last_vintage(public, BASE, issues[-1]).rows)
    final_arm = rows_by_week(last_vintage(public, arm, issues[-1]).rows)
    differing = {cell for cell in final_arm if final_arm[cell] != final_baseline[cell]}
    assert any(j == jurisdiction for j, _ in differing)
    assert all(week >= start for _, week in differing)


def assert_nothing_leaks(public: Workspace, world: World) -> None:
    """No parameter of the world's truth is written anywhere in the public workspace."""
    cfg = ToyConfig.model_validate_json(world.dump())
    secrets_ = [*cfg.levels, *(v for v in (cfg.x_effect, cfg.school_effect) if v > 0)]
    for path in public.root.rglob("*"):
        if path.is_file():
            text = path.read_text()
            for value in secrets_:
                assert repr(value) not in text, f"{repr(value)} leaked into {path}"
    for spec in Archive(public).specs():
        assert spec.support_max in (None, SUPPORT_MAX), f"{spec.id} bounds its counts by {spec.support_max}"


# --- create -------------------------------------------------------------------------------------


def test_create_publishes_three_streams(custodian: Custodian, public: Workspace) -> None:
    cfg = make_world(custodian)
    specs = {s.id: s for s in Archive(public).specs()}
    assert set(specs) == {BASE, "w1/x", "w1/term"}
    base, x, term = specs[BASE], specs["w1/x"], specs["w1/term"]
    assert (base.world, base.role, base.kind) == (WORLD, "admissions", StreamKind.COUNT)
    assert base.jurisdictions == cfg.jurisdictions
    assert base.support_max == SUPPORT_MAX
    assert (base.intervention, base.parent, base.first_issue, base.fulfills) == (None, None, 0, None)
    assert (x.world, x.role, x.kind, x.jurisdictions) == (WORLD, "covariate-x", StreamKind.COVARIATE, cfg.jurisdictions)
    assert (term.world, term.role, term.kind) == (WORLD, "school-term", StreamKind.COVARIATE)


def test_the_mission_follows_the_template(custodian: Custodian, public: Workspace) -> None:
    cfg = make_world(custodian)
    mission = load_mission(public, WORLD)
    assert mission.name == WORLD
    assert mission.captain == "llm-captain"
    assert mission.streams == (BASE, "w1/x", "w1/term")
    assert mission.horizons == (0, 1, 2, 3)
    assert (mission.first_issue, mission.last_issue, mission.resolution_lag) == (4, WEEKS - 3 - 4, 4)
    assert (mission.epsilon, mission.tau, mission.half_life_weeks) == (0.01, float(JURISDICTIONS), 8.0)
    assert (mission.coverage_min, mission.claim_threshold_nats) == (0.8, 3.0)
    assert (mission.regime_flag_tv, mission.exploration_reserve) == (0.5, 0.2)
    assert mission.required_isolation is Isolation.CONTAINER
    claims = {c.id: c for c in mission.claims}
    assert list(claims) == ["covariate-x", "covariate-x-flat", "school"]
    assert claims["covariate-x"].scope == TargetFilter(roles=("admissions",), jurisdictions=("j1", "j2"))
    assert claims["covariate-x"].weight == 0.5
    assert claims["covariate-x"].refines is None
    assert claims["covariate-x-flat"].scope == TargetFilter(roles=("admissions",), jurisdictions=(cfg.flat_jurisdiction,))
    assert claims["covariate-x-flat"].weight == 0.0
    assert claims["covariate-x-flat"].refines == "covariate-x"
    assert claims["school"].scope == TargetFilter(roles=("admissions",))
    assert claims["school"].weight == 1.0
    assert claims["school"].refines is None


def test_claims_are_scoped_to_the_world_channel(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian, channel="ed-visits")
    mission = load_mission(public, WORLD)
    assert mission.streams == ("w1/ed-visits", "w1/x", "w1/term")
    assert all(c.scope.roles == ("ed-visits",) for c in mission.claims)


def test_the_required_isolation_is_a_parameter(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian, required_isolation=Isolation.NONE)
    assert load_mission(public, WORLD).required_isolation is Isolation.NONE


def test_the_wanted_grid_has_two_items_per_jurisdiction_and_start(custodian: Custodian, public: Workspace) -> None:
    cfg = make_world(custodian)
    items = {i.id: i for i in load_wanted(public, WORLD)}
    expected = {
        f"{kind}-{j}-w{start}" for kind in ("close", "vacc") for j in cfg.jurisdictions for start in GRID_STARTS
    }
    assert set(items) == expected
    assert len(load_wanted(public, WORLD)) == 2 * JURISDICTIONS * len(GRID_STARTS)
    for item in items.values():
        assert (item.proposed_by, item.cost, item.fulfiller) == ("custodian-grid", 1.0, "custodian")


def test_a_closure_item_closes_schools_for_three_weeks(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    item = {i.id: i for i in load_wanted(public, WORLD)}["close-j1-w12"]
    arm = item.candidate.stream
    assert arm.id == "w1/admissions~close-j1-w12"
    assert arm.parent == BASE
    assert arm.intervention == Intervention(
        variable="schools_closed", value=1.0, start_week=12, end_week=14, jurisdictions=("j1",)
    )
    assert (arm.world, arm.role, arm.kind) == (WORLD, "admissions", StreamKind.COUNT)
    assert arm.jurisdictions == ("j0", "j1", "j2")
    assert arm.fulfills is None


def test_a_vaccination_item_vaccinates_thirty_percent_from_its_start(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    item = {i.id: i for i in load_wanted(public, WORLD)}["vacc-j2-w20"]
    arm = item.candidate.stream
    assert arm.id == "w1/admissions~vacc-j2-w20"
    assert arm.parent == BASE
    assert arm.intervention == Intervention(variable="vaccinate_fraction", value=0.3, start_week=20, jurisdictions=("j2",))


def test_grid_targets_sit_at_the_start_and_resolve_after_the_mission_lag(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    item = {i.id: i for i in load_wanted(public, WORLD)}["close-j1-w12"]
    expected = tuple(
        Target(stream="w1/admissions~close-j1-w12", jurisdiction=j, reference_week=12, horizon=h, resolution_lag=4)
        for j in ("j0", "j1", "j2")
        for h in (0, 1, 2, 3)
    )
    assert item.candidate.targets == expected
    assert {t.resolution_issue - t.target_week for i in load_wanted(public, WORLD) for t in i.candidate.targets} == {4}


def test_every_target_the_mission_asks_resolves_by_the_last_issue_the_world_publishes(
    custodian: Custodian, public: Workspace
) -> None:
    make_world(custodian)
    mission = load_mission(public, WORLD)
    spec = Archive(public).spec(BASE)
    asked = [t for ref in range(mission.first_issue, mission.last_issue + 1) for t in mission.targets(spec, ref)]
    assert max(t.resolution_issue for t in asked) == WEEKS  # the window is tight: the last target resolves at the last issue
    beyond = Target(stream=BASE, jurisdiction="j0", reference_week=mission.last_issue + 1, horizon=3, resolution_lag=4)
    assert beyond.resolution_issue == WEEKS + 1
    assert mission.targets(spec, mission.last_issue + 1) == []


def test_a_start_outside_the_mission_window_has_no_targets(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian, weeks=20, jurisdictions=2)  # the window is weeks 4 to 13
    items = {i.id: i for i in load_wanted(public, WORLD)}
    assert len(items) == 2 * 2 * len(GRID_STARTS)
    assert len(items["close-j0-w8"].candidate.targets) == 2 * 4
    assert len(items["vacc-j1-w12"].candidate.targets) == 2 * 4
    assert items["close-j0-w16"].candidate.targets == ()
    assert items["vacc-j1-w20"].candidate.targets == ()


@pytest.mark.parametrize("world_id", ["W1", "w/1", "..", "", "w 1", "a_b"])
def test_an_invalid_world_id_is_refused_before_anything_is_written(
    custodian: Custodian, public: Workspace, private: Path, world_id: str
) -> None:
    with pytest.raises(CustodianError, match="world id"):
        make_world(custodian, world_id=world_id)
    assert files(private) == [] and files(public.root) == []


def test_a_world_id_may_use_every_character_of_a_stream_id_segment(custodian: Custodian) -> None:
    make_world(custodian, world_id="seir-adm.1~b")
    assert custodian.world("seir-adm.1~b").id == "seir-adm.1~b"


def test_a_public_world_is_not_recreated_from_another_private_root(
    custodian: Custodian, public: Workspace, tmp_path: Path
) -> None:
    make_world(custodian)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(CustodianError, match="already exists"):
        make_world(Custodian(public, elsewhere, log=None), seed=9)
    assert files(elsewhere) == []


def test_a_stream_already_under_the_world_prefix_blocks_creation_before_anything_private_is_written(
    custodian: Custodian, public: Workspace, private: Path
) -> None:
    foreign = StreamSpec(
        id="w9/admissions", world="w1", role="admissions", kind=StreamKind.COUNT, jurisdictions=("j0",), support_max=SUPPORT_MAX
    )
    Archive(public).create_stream(foreign)
    with pytest.raises(CustodianError, match="already exists"):
        make_world(custodian, world_id="w9")
    assert files(private) == []
    assert [s.id for s in Archive(public).specs()] == ["w9/admissions"]


def test_a_channel_without_a_lens_is_refused(custodian: Custodian, public: Workspace, private: Path) -> None:
    with pytest.raises(CustodianError, match="channel"):
        make_world(custodian, channel="school-term")
    assert files(private) == [] and files(public.root) == []


def test_an_unknown_world_is_an_error(custodian: Custodian) -> None:
    for call in (
        lambda: custodian.world("nope"),
        lambda: custodian.current_issue("nope"),
        lambda: custodian.tick("nope", 3, NOW),
        lambda: custodian.fulfill("nope", "x", NOW),
        lambda: custodian.add_channel("nope", "ed-visits", NOW),
    ):
        with pytest.raises(CustodianError, match="nope"):
            call()


# --- tick ---------------------------------------------------------------------------------------


def test_tick_publishes_issues_one_to_n_of_every_stream_at_now(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    captures = custodian.tick(WORLD, 5, at(1))
    archive = Archive(public)
    for stream in (BASE, "w1/x"):
        assert archive.issues(stream) == [1, 2, 3, 4, 5]
        assert all(archive.published_at(stream, issue) == at(1) for issue in range(1, 6))
    assert archive.issues("w1/term") == [0]
    assert len(captures) == 10
    assert [c.vintage.issue for c in captures] == sorted(c.vintage.issue for c in captures)
    assert {c.published_at for c in captures} == {at(1)}
    assert {c.source for c in captures} == {"custodian:w1"}
    assert custodian.current_issue(WORLD) == 5


def test_tick_continues_from_the_current_issue(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 3, at(1))
    captures = custodian.tick(WORLD, 6, at(2))
    assert sorted({c.vintage.issue for c in captures}) == [4, 5, 6]
    archive = Archive(public)
    assert archive.issues(BASE) == [1, 2, 3, 4, 5, 6]
    assert [archive.published_at(BASE, i) for i in (3, 4)] == [at(1), at(2)]
    assert custodian.current_issue(WORLD) == 6


def test_tick_to_an_issue_already_published_does_nothing(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 4, at(1))
    assert custodian.tick(WORLD, 4, at(2)) == []
    assert custodian.tick(WORLD, 2, at(2)) == []
    assert Archive(public).issues(BASE) == [1, 2, 3, 4]


def test_tick_repairs_streams_left_behind_by_an_interrupted_tick(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 4, at(1))
    arm = custodian.fulfill(WORLD, "close-j1-w8", at(1))
    custodian.tick(WORLD, 6, at(2))
    # as if the tick had stopped after the base channel: the covariate and the arm miss their last issues
    (public.root / "archive/w1/x/00005.json").unlink()
    (public.root / "archive/w1/x/00006.json").unlink()
    (public.root / "archive" / arm.id / "00006.json").unlink()
    archive = Archive(public)
    assert (archive.issues("w1/x"), archive.issues(arm.id), custodian.current_issue(WORLD)) == (
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        6,
    )
    captures = custodian.tick(WORLD, 8, at(3))
    assert sorted((c.vintage.stream, c.vintage.issue) for c in captures) == sorted(
        [("w1/x", i) for i in (5, 6, 7, 8)] + [(arm.id, i) for i in (6, 7, 8)] + [(BASE, i) for i in (7, 8)]
    )
    assert archive.issues("w1/x") == archive.issues(arm.id) == archive.issues(BASE) == list(range(1, 9))
    assert [archive.published_at("w1/x", i) for i in (4, 5)] == [at(1), at(3)]


def test_captures_come_back_by_issue_then_count_streams_then_the_covariate(custodian: Custodian) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 2, at(1))
    custodian.add_channel(WORLD, "ed-visits", at(2))
    order = [(c.vintage.issue, c.vintage.stream) for c in custodian.tick(WORLD, 4, at(3))]
    assert order == [
        (3, BASE), (3, "w1/ed-visits"), (3, "w1/x"),
        (4, BASE), (4, "w1/ed-visits"), (4, "w1/x"),
    ]  # fmt: skip


def test_tick_past_the_last_week_is_refused_and_publishes_nothing(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 5, at(1))
    with pytest.raises(CustodianError, match="weeks"):
        custodian.tick(WORLD, WEEKS + 1, at(2))
    assert Archive(public).issues(BASE) == [1, 2, 3, 4, 5]
    custodian.tick(WORLD, WEEKS, at(2))
    assert custodian.current_issue(WORLD) == WEEKS
    assert Archive(public).issues(BASE) == list(range(1, WEEKS + 1))


def test_ticking_leaks_no_parameters(custodian: Custodian, public: Workspace) -> None:
    cfg = make_world(custodian)
    custodian.tick(WORLD, WEEKS, at(1))
    assert_nothing_leaks(public, cfg)


# --- fulfill ------------------------------------------------------------------------------------


def test_fulfill_creates_the_arm_stream_from_the_wanted_item(custodian: Custodian, public: Workspace) -> None:
    cfg = make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    arm = custodian.fulfill(WORLD, "vacc-j1-w8", at(2))
    assert arm.id == "w1/admissions~vacc-j1-w8"
    assert arm.parent == BASE
    assert arm.intervention == Intervention(variable="vaccinate_fraction", value=0.3, start_week=8, jurisdictions=("j1",))
    assert arm.fulfills == "vacc-j1-w8"
    assert arm.first_issue == 7
    assert (arm.world, arm.role, arm.kind) == (WORLD, "admissions", StreamKind.COUNT)
    assert arm.jurisdictions == cfg.jurisdictions
    assert arm.support_max == SUPPORT_MAX
    assert Archive(public).spec(arm.id) == arm


def test_fulfill_releases_the_history_the_parent_has_published(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    arm = custodian.fulfill(WORLD, "close-j2-w8", at(2))
    archive = Archive(public)
    assert archive.issues(arm.id) == [1, 2, 3, 4, 5, 6]
    assert {archive.published_at(arm.id, i) for i in range(1, 7)} == {at(2)}
    for issue in range(1, 7):
        assert last_vintage(public, arm.id, issue).rows == last_vintage(public, BASE, issue).rows
    assert custodian.current_issue(WORLD) == 6


def test_fulfill_at_the_first_unpublished_week_is_allowed(custodian: Custodian) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 7, at(1))
    assert custodian.fulfill(WORLD, "close-j0-w8", at(2)).first_issue == 8


def test_fulfill_refuses_a_start_that_is_not_after_the_current_issue(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 8, at(1))
    with pytest.raises(CustodianError, match="start"):
        custodian.fulfill(WORLD, "close-j1-w8", at(2))
    assert [s.id for s in Archive(public).specs() if s.fulfills] == []
    custodian.tick(WORLD, 12, at(3))
    with pytest.raises(CustodianError, match="start"):
        custodian.fulfill(WORLD, "vacc-j1-w12", at(4))


def test_the_arm_matches_the_baseline_before_its_start_in_every_jurisdiction(
    custodian: Custodian, public: Workspace
) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    arm = custodian.fulfill(WORLD, "vacc-j1-w8", at(2))
    custodian.tick(WORLD, WEEKS, at(3))
    assert_arm_matches_baseline_before_start(public, arm.id, 8, "j1", (8, 12, WEEKS), JURISDICTIONS)


def test_an_arm_is_ticked_with_the_rest(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    arm = custodian.fulfill(WORLD, "close-j1-w8", at(2))
    captures = custodian.tick(WORLD, 9, at(3))
    assert Archive(public).issues(arm.id) == list(range(1, 10))
    assert sum(c.vintage.stream == arm.id for c in captures) == 3
    assert {c.published_at for c in captures} == {at(3)}


def test_mission_streams_include_the_arms_after_fulfillment(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    mission = load_mission(public, WORLD)
    assert [s.id for s in mission_streams(mission, Archive(public).specs())] == [BASE, "w1/term", "w1/x"]
    arm = custodian.fulfill(WORLD, "close-j1-w8", at(2))
    assert [s.id for s in mission_streams(mission, Archive(public).specs())] == [BASE, arm.id, "w1/term", "w1/x"]


def test_programs_forecasting_the_mission_see_the_covariates(custodian: Custodian, public: Workspace) -> None:
    """The seed SEIR theories that assert and deny the covariate x differ where x varies, and only there (R16)."""
    for name in SEED_PROGRAMS:
        shutil.copytree(DEFAULT_PUBLIC / "programs" / name, public.programs / name, ignore=shutil.ignore_patterns("__pycache__"))
        Registry(public).freeze(name, NOW - timedelta(hours=1))
    cfg = make_world(custodian, required_isolation=Isolation.NONE)
    mission = load_mission(public, WORLD)
    assert {"w1/x", "w1/term"} <= {s.id for s in mission_streams(mission, Archive(public).specs())}
    custodian.tick(WORLD, 8, at(1))
    run_origin(public, mission, 8, Isolation.NONE, at(1))
    names = {h: f.name for h, f in Registry(public).freezes().items()}
    pmfs = {
        (names[r.program], r.target.jurisdiction, r.target.horizon): r.pmf
        for r in RecordStore(public).all(WORLD)
        if isinstance(r, PredictionRecord)
    }

    def differ(jurisdiction: str) -> list[bool]:
        return [pmfs["seir-x1-s1", jurisdiction, h] != pmfs["seir-x0-s1", jurisdiction, h] for h in mission.horizons]

    assert all(differ("j1")) and all(differ("j2"))
    assert not any(differ(cfg.flat_jurisdiction))  # x is identically zero there, so the theories agree


def test_fulfilling_leaks_no_parameters(custodian: Custodian, public: Workspace) -> None:
    cfg = make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    custodian.fulfill(WORLD, "close-j1-w8", at(2))
    custodian.tick(WORLD, WEEKS, at(3))
    assert_nothing_leaks(public, cfg)


def test_an_item_is_fulfilled_once(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    custodian.fulfill(WORLD, "close-j1-w8", at(2))
    with pytest.raises(CustodianError, match="already"):
        custodian.fulfill(WORLD, "close-j1-w8", at(3))
    assert len([s for s in Archive(public).specs() if s.fulfills]) == 1


def test_an_unknown_item_is_refused(custodian: Custodian) -> None:
    make_world(custodian)
    with pytest.raises(CustodianError, match="no-such-item"):
        custodian.fulfill(WORLD, "no-such-item", at(1))


@pytest.mark.parametrize("jurisdictions", [("j9",), ("j1", "nowhere")])
def test_an_unknown_jurisdiction_is_refused_and_nothing_is_created(
    custodian: Custodian, public: Workspace, jurisdictions: tuple[str, ...]
) -> None:
    make_world(custodian)
    propose(public, "bad", Intervention(variable="schools_closed", value=1.0, start_week=10, jurisdictions=jurisdictions))
    with pytest.raises(CustodianError, match="jurisdiction"):
        custodian.fulfill(WORLD, "bad", at(1))
    assert [s.id for s in Archive(public).specs() if s.fulfills] == []


def test_an_intervention_without_jurisdictions_applies_to_every_one(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    propose(public, "everywhere", Intervention(variable="schools_closed", value=1.0, start_week=10, end_week=12))
    arm = custodian.fulfill(WORLD, "everywhere", at(1))
    assert arm.intervention is not None and arm.intervention.jurisdictions is None


@pytest.mark.parametrize("variable", ["schools_closed", "vaccinate_fraction"])
def test_an_intervention_must_start_within_the_season(custodian: Custodian, public: Workspace, variable: str) -> None:
    make_world(custodian)
    propose(public, "late", Intervention(variable=variable, value=0.3, start_week=WEEKS, jurisdictions=("j1",)))
    with pytest.raises(CustodianError, match="start"):
        custodian.fulfill(WORLD, "late", at(1))
    propose(public, "last", Intervention(variable=variable, value=0.3, start_week=WEEKS - 1, jurisdictions=("j1",)))
    assert custodian.fulfill(WORLD, "last", at(1)).intervention is not None


def test_an_unknown_variable_is_refused(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    propose(public, "masks", Intervention(variable="mask_wearing", value=1.0, start_week=10))
    with pytest.raises(CustodianError, match="mask_wearing"):
        custodian.fulfill(WORLD, "masks", at(1))


def test_an_arm_of_an_arm_or_of_another_world_is_refused(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    make_world(custodian, world_id="w2", seed=8)
    custodian.tick(WORLD, 3, at(1))
    arm = custodian.fulfill(WORLD, "close-j1-w8", at(1))
    iv = Intervention(variable="schools_closed", value=1.0, start_week=12, jurisdictions=("j1",))
    propose(public, "nested", iv, parent=arm.id)
    propose(public, "foreign", iv, parent="w2/admissions")
    propose(public, "covariate", iv, parent="w1/x")
    for item in ("nested", "foreign", "covariate"):
        with pytest.raises(CustodianError, match="parent"):
            custodian.fulfill(WORLD, item, at(2))


def test_a_proposed_arm_is_named_from_its_parent_and_its_item(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    propose(public, "mine", Intervention(variable="schools_closed", value=1.0, start_week=10, jurisdictions=("j1",)))
    assert custodian.fulfill(WORLD, "mine", at(1)).id == "w1/admissions~mine"


@pytest.mark.parametrize(
    "arm_id",
    ["other/world~x", "squat", "w9/admissions", "w1/admissions~other", "w1/proposed~x", "w1/admissions~close-j1-w8"],
)
def test_a_proposal_whose_stream_id_is_not_parent_and_item_is_refused(
    custodian: Custodian, public: Workspace, arm_id: str
) -> None:
    make_world(custodian)
    propose(public, "x", Intervention(variable="schools_closed", value=1.0, start_week=10), arm_id=arm_id)
    with pytest.raises(CustodianError, match="stream id"):
        custodian.fulfill(WORLD, "x", at(1))
    assert [s.id for s in Archive(public).specs() if s.fulfills] == []
    assert all(s.id.startswith("w1/") for s in Archive(public).specs())


def test_a_refused_proposal_does_not_block_a_later_world_from_being_created(
    custodian: Custodian, public: Workspace, private: Path
) -> None:
    make_world(custodian)
    propose(public, "x", Intervention(variable="schools_closed", value=1.0, start_week=10), arm_id="w9/admissions")
    with pytest.raises(CustodianError, match="stream id"):
        custodian.fulfill(WORLD, "x", at(1))
    make_world(custodian, world_id="w9", seed=8)
    assert custodian.world("w9").id == "w9"
    assert files(private) == [Path("worlds/w1/config.json"), Path("worlds/w9/config.json")]


@pytest.mark.parametrize("variable", ["schools_closed", "vaccinate_fraction"])
@pytest.mark.parametrize("value", [-0.1, 1.01, 2.0, -5.0])
def test_a_value_outside_zero_to_one_is_refused(
    custodian: Custodian, public: Workspace, variable: str, value: float
) -> None:
    make_world(custodian)
    propose(public, "v", Intervention(variable=variable, value=value, start_week=10, jurisdictions=("j1",)))
    with pytest.raises(CustodianError, match="value"):
        custodian.fulfill(WORLD, "v", at(1))
    assert [s.id for s in Archive(public).specs() if s.fulfills] == []


@pytest.mark.parametrize("variable", ["schools_closed", "vaccinate_fraction"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_the_end_values_zero_and_one_are_allowed(
    custodian: Custodian, public: Workspace, variable: str, value: float
) -> None:
    make_world(custodian)
    propose(public, "v", Intervention(variable=variable, value=value, start_week=10, jurisdictions=("j1",)))
    assert custodian.fulfill(WORLD, "v", at(1)).fulfills == "v"


def test_an_end_week_before_the_start_week_is_refused(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    propose(public, "back", Intervention(variable="schools_closed", value=1.0, start_week=12, end_week=11))
    with pytest.raises(CustodianError, match="end week"):
        custodian.fulfill(WORLD, "back", at(1))
    propose(public, "day", Intervention(variable="schools_closed", value=1.0, start_week=12, end_week=12))
    assert custodian.fulfill(WORLD, "day", at(1)).fulfills == "day"


# --- add_channel --------------------------------------------------------------------------------


def test_add_channel_creates_a_base_stream_that_starts_after_the_current_issue(
    custodian: Custodian, public: Workspace
) -> None:
    cfg = make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    spec = custodian.add_channel(WORLD, "ed-visits", at(2))
    assert spec.id == "w1/ed-visits"
    assert (spec.world, spec.role, spec.kind) == (WORLD, "ed-visits", StreamKind.COUNT)
    assert spec.jurisdictions == cfg.jurisdictions
    assert spec.support_max == SUPPORT_MAX
    assert (spec.intervention, spec.parent, spec.fulfills) == (None, None, None)
    assert spec.first_issue == 7
    assert Archive(public).spec(spec.id) == spec


def test_an_added_channel_is_ticked_with_the_rest(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 4, at(1))
    custodian.add_channel(WORLD, "ed-visits", at(2))
    custodian.tick(WORLD, 7, at(3))
    archive = Archive(public)
    assert archive.issues("w1/ed-visits") == list(range(1, 8))
    assert archive.issues(BASE) == list(range(1, 8))
    assert [archive.published_at("w1/ed-visits", i) for i in (4, 5)] == [at(2), at(3)]


def test_an_added_channel_joins_the_mission_streams(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 4, at(1))
    before = load_mission(public, WORLD)
    custodian.add_channel(WORLD, "ed-visits", at(2))
    mission = load_mission(public, WORLD)
    assert mission.streams == (BASE, "w1/x", "w1/term", "w1/ed-visits")
    assert mission == before.model_copy(update={"streams": (*before.streams, "w1/ed-visits")})
    assert custodian.current_issue(WORLD) == 4
    streams = [s.id for s in mission_streams(mission, Archive(public).specs())]
    assert streams == [BASE, "w1/ed-visits", "w1/term", "w1/x"]
    # its targets start at its own first issue, whatever the mission window says
    spec = Archive(public).spec("w1/ed-visits")
    assert mission.targets(spec, 4) == []
    assert mission.targets(spec, 5) != []


def test_add_channel_needs_the_worlds_mission_and_creates_nothing_without_it(
    custodian: Custodian, public: Workspace
) -> None:
    make_world(custodian)
    (public.root / "missions/w1/mission.json").unlink()
    with pytest.raises(CustodianError, match="mission"):
        custodian.add_channel(WORLD, "ed-visits", at(1))
    assert "w1/ed-visits" not in {s.id for s in Archive(public).specs()}


def test_a_world_starts_its_added_channel_at_issue_one_when_nothing_is_published(
    custodian: Custodian,
) -> None:
    make_world(custodian)
    spec = custodian.add_channel(WORLD, "ed-visits", at(1))
    assert spec.first_issue == 1
    assert custodian.current_issue(WORLD) == 0
    custodian.tick(WORLD, 3, at(2))
    assert custodian.current_issue(WORLD) == 3


def test_a_channel_is_added_once(custodian: Custodian) -> None:
    make_world(custodian)
    custodian.add_channel(WORLD, "ed-visits", at(1))
    with pytest.raises(CustodianError, match="already"):
        custodian.add_channel(WORLD, "ed-visits", at(2))
    with pytest.raises(CustodianError, match="already"):
        custodian.add_channel(WORLD, "admissions", at(2))


def test_a_channel_without_a_lens_cannot_be_added(custodian: Custodian, public: Workspace) -> None:
    make_world(custodian)
    with pytest.raises(CustodianError, match="channel"):
        custodian.add_channel(WORLD, "school-term", at(1))
    assert "w1/school-term" not in {s.id for s in Archive(public).specs()}



# --- create, through the protocol ----------------------------------------------------------------


def test_create_writes_the_truth_privately_and_everything_else_publicly(
    custodian: Custodian, public: Workspace, private: Path
) -> None:
    world = make_world(custodian)
    assert files(private) == [Path("worlds/w1/config.json")]
    assert (private / "worlds/w1/config.json").read_bytes() == world.dump()
    assert custodian.world(WORLD).dump() == world.dump()
    assert files(public.root) == [
        Path("archive/w1/admissions/stream.json"),
        Path("archive/w1/term/00000.json"),
        Path("archive/w1/term/stream.json"),
        Path("archive/w1/x/stream.json"),
        Path("missions/w1/epochs.jsonl"),
        Path("missions/w1/mission.json"),
        Path("missions/w1/wanted.jsonl"),
    ]
    assert_nothing_leaks(public, world)


def test_create_draws_the_world_from_the_given_generator(custodian: Custodian, tmp_path: Path) -> None:
    world = make_world(custodian, jurisdictions=2)
    assert world.dump() == GENERATOR.draw(WORLD, WEEKS, 2, np.random.default_rng(7), tmp_path).dump()
    assert world.jurisdictions == ("j0", "j1")


def test_without_an_rng_the_truth_comes_from_the_operating_system(
    custodian: Custodian, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asked: list[int] = []

    def randbits(k: int) -> int:
        asked.append(k)
        return 12345

    monkeypatch.setattr(custodian_module.secrets, "randbits", randbits)
    world = custodian.create("toy", WORLD, WEEKS, JURISDICTIONS, "admissions", NOW)
    assert asked == [128]
    assert world.dump() == GENERATOR.draw(WORLD, WEEKS, JURISDICTIONS, np.random.default_rng(12345), tmp_path).dump()


def test_two_worlds_created_without_an_rng_differ(custodian: Custodian) -> None:
    first = custodian.create("toy", "a1", WEEKS, JURISDICTIONS, "admissions", NOW)
    second = custodian.create("toy", "a2", WEEKS, JURISDICTIONS, "admissions", NOW)
    assert ToyConfig.model_validate_json(first.dump()).seed != ToyConfig.model_validate_json(second.dump()).seed


def test_create_publishes_the_whole_term_calendar_at_issue_zero(custodian: Custodian, public: Workspace) -> None:
    world = make_world(custodian, jurisdictions=2)
    archive = Archive(public)
    assert archive.issues("w1/term") == [0]
    assert archive.published_at("w1/term", 0) == NOW
    assert last_vintage(public, "w1/term", 0) == world.vintage(archive.spec("w1/term"), 0)
    assert {r.week for r in last_vintage(public, "w1/term", 0).rows} == set(range(WEEKS))
    assert archive.issues(BASE) == [] and archive.issues("w1/x") == []
    assert custodian.current_issue(WORLD) == 0


@pytest.mark.parametrize(("generator", "weeks", "jurisdictions"), [("sir", 24, 3), ("toy", 0, 3), ("toy", 24, 0)])
def test_an_invalid_world_shape_is_refused(
    custodian: Custodian, public: Workspace, private: Path, generator: str, weeks: int, jurisdictions: int
) -> None:
    with pytest.raises(CustodianError):
        make_world(custodian, generator=generator, weeks=weeks, jurisdictions=jurisdictions)
    assert files(private) == [] and files(public.root) == []


def test_a_world_is_created_once(custodian: Custodian, public: Workspace, private: Path) -> None:
    world = make_world(custodian)
    before = (files(private), files(public.root))
    with pytest.raises(CustodianError, match="already exists"):
        make_world(custodian, seed=8)
    assert (files(private), files(public.root)) == before
    assert custodian.world(WORLD).dump() == world.dump()


# --- tick and add_channel publish the world's own vintages ----------------------------------------


def test_count_and_covariate_vintages_are_the_worlds(custodian: Custodian, public: Workspace) -> None:
    world = make_world(custodian)
    custodian.tick(WORLD, 9, at(1))
    archive = Archive(public)
    for stream in (BASE, "w1/x"):
        for issue in (1, 5, 9):
            assert last_vintage(public, stream, issue) == world.vintage(archive.spec(stream), issue)
    assert any(r.value > 0 for r in last_vintage(public, BASE, 9).rows)


def test_add_channel_releases_the_history_at_now(custodian: Custodian, public: Workspace) -> None:
    world = make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    spec = custodian.add_channel(WORLD, "ed-visits", at(2))
    archive = Archive(public)
    assert archive.issues(spec.id) == [1, 2, 3, 4, 5, 6]
    assert {archive.published_at(spec.id, i) for i in range(1, 7)} == {at(2)}
    assert last_vintage(public, spec.id, 6) == world.vintage(spec, 6)
    assert {r.value for r in last_vintage(public, spec.id, 6).rows} != {r.value for r in last_vintage(public, BASE, 6).rows}


def test_the_private_root_holds_only_the_config(custodian: Custodian, private: Path) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 6, at(1))
    custodian.fulfill(WORLD, "close-j1-w8", at(2))
    custodian.add_channel(WORLD, "ed-visits", at(3))
    assert files(private) == [Path("worlds/w1/config.json")]  # the toy world caches nothing


# --- sealing --------------------------------------------------------------------------------------


@pytest.fixture
def log(tmp_path: Path) -> CommitmentLog:
    (tmp_path / "lab").mkdir()
    return CommitmentLog(tmp_path / "lab")


@pytest.fixture
def sealing(public: Workspace, private: Path, log: CommitmentLog) -> Custodian:
    return Custodian(public, private, log)


def nonce(path: Path) -> bytes:
    return bytes.fromhex(path.read_text())


def test_create_seals_the_generator_then_the_world_before_publishing(
    sealing: Custodian, log: CommitmentLog, private: Path
) -> None:
    make_world(sealing)
    generator, world = log.entries()
    assert (generator.subject.kind, generator.subject.name) == (SubjectKind.GENERATOR, "scienceme-lab")
    assert (world.subject.kind, world.subject.name) == (SubjectKind.WORLD, WORLD)
    assert world.seal == seal(nonce(private / "commitments/world-w1.nonce"), (private / "worlds/w1/config.json").read_bytes())
    assert sorted(p.name for p in (private / "commitments").iterdir()) == [
        p.name for p in sorted((private / "commitments").glob("generator-scienceme-lab-*"))
    ] + ["world-w1.nonce"]


def test_a_generator_version_is_sealed_once(sealing: Custodian, log: CommitmentLog) -> None:
    make_world(sealing)
    make_world(sealing, world_id="w2", seed=8)
    sealing.tick(WORLD, 6, at(1))
    sealing.fulfill(WORLD, "close-j1-w8", at(2))
    kinds = [e.subject.kind for e in log.entries()]
    assert kinds == [SubjectKind.GENERATOR, SubjectKind.WORLD, SubjectKind.WORLD]


def test_a_changed_generator_is_sealed_before_it_publishes(
    public: Workspace, private: Path, log: CommitmentLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "pkgs" / "editablegen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from scienceme_lab.toy import ToyGenerator\n\n\nclass Edited(ToyGenerator):\n    pass\n\n\nGENERATOR = Edited()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "pkgs"))
    import editablegen  # noqa: PLC0415

    custodian = Custodian(public, private, log, generators={"toy": editablegen.GENERATOR})
    make_world(custodian)
    custodian.tick(WORLD, 3, at(1))
    assert [e.subject.kind for e in log.entries()] == [SubjectKind.GENERATOR, SubjectKind.WORLD]
    (package / "dynamics.py").write_text("RATE = 2\n")  # the generator's source changes between ticks
    captures = custodian.tick(WORLD, 5, at(2))
    entries = log.entries()
    assert [e.subject.kind for e in entries] == [SubjectKind.GENERATOR, SubjectKind.WORLD, SubjectKind.GENERATOR]
    assert entries[2].committed_at == at(2) and {c.published_at for c in captures} == {at(2)}
    assert (entries[2].subject.name, entries[0].seal != entries[2].seal) == ("editablegen", True)


def test_without_a_log_nothing_is_sealed(custodian: Custodian, private: Path) -> None:
    make_world(custodian)
    custodian.tick(WORLD, 3, at(1))
    assert not (private / "commitments").exists()
    with pytest.raises(CustodianError, match="no commitment log"):
        custodian.reveal(WORLD, at(2))


def test_a_world_of_an_uninstalled_generator_is_refused(custodian: Custodian, public: Workspace, private: Path) -> None:
    make_world(custodian)
    (private / "worlds/w1/config.json").write_text('{"generator": "absent"}')
    for call in (lambda: custodian.world(WORLD), lambda: custodian.tick(WORLD, 3, at(1))):
        with pytest.raises(CustodianError, match="not installed.*toy") as refused:
            call()
        assert "absent" not in str(refused.value)  # which generator a world uses stays private
    assert Archive(public).issues(BASE) == []


def test_a_sealed_but_unwritten_world_is_refused_clearly(
    sealing: Custodian, public: Workspace, private: Path, log: CommitmentLog
) -> None:
    (private / "commitments").mkdir()
    (private / "commitments/world-w1.nonce").write_text("00" * 32)
    with pytest.raises(CustodianError, match="sealed"):
        make_world(sealing)
    assert files(public.root) == [] and log.entries() == []


def test_a_world_is_revealed_only_once_it_has_closed(sealing: Custodian, log: CommitmentLog, private: Path) -> None:
    make_world(sealing, weeks=10)
    sealing.tick(WORLD, 9, at(1))
    with pytest.raises(CustodianError, match="not closed"):
        sealing.reveal(WORLD, at(2))
    sealing.tick(WORLD, 10, at(3))
    reveal = sealing.reveal(WORLD, at(4))
    assert reveal.path == "revealed/worlds/w1/config.json"
    assert (log.path.parent / reveal.path).read_bytes() == (private / "worlds/w1/config.json").read_bytes()
    assert log.problems() == []


def test_an_unsealed_world_cannot_be_revealed(custodian: Custodian, public: Workspace, private: Path, log: CommitmentLog) -> None:
    make_world(custodian, weeks=10)  # created without a log
    custodian.tick(WORLD, 10, at(1))
    with pytest.raises(CustodianError, match="never sealed"):
        Custodian(public, private, log).reveal(WORLD, at(2))


def test_a_corrupt_world_configuration_is_one_custodian_error(custodian: Custodian, private: Path) -> None:
    make_world(custodian)
    (private / "worlds/w1/config.json").write_text("not json")
    with pytest.raises(CustodianError, match="w1.*not JSON"):
        custodian.world(WORLD)


def test_a_misregistered_generator_is_a_custodian_error(
    public: Workspace, private: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scienceme_lab.generator import GeneratorError  # noqa: PLC0415

    def broken() -> dict:
        raise GeneratorError("the entry point 'other' holds the generator 'toy'")

    monkeypatch.setattr(custodian_module, "generators", broken)
    with pytest.raises(CustodianError, match="other"):
        Custodian(public, private, log=None)



def test_a_generator_nonce_missing_from_the_log_is_sealed_there(public: Workspace, private: Path, tmp_path: Path) -> None:
    first = CommitmentLog(tmp_path / "first")
    make_world(Custodian(public, private, first))
    (sealed, _) = first.entries()
    later = CommitmentLog(tmp_path / "later")  # the nonce is kept, but this log never saw the commitment
    captures = Custodian(public, private, later).tick(WORLD, 3, at(1))
    assert captures and [e.seal for e in later.entries()] == [sealed.seal]


def test_a_sealed_generator_source_is_kept_privately(sealing: Custodian, private: Path) -> None:
    from scienceme.worker import package_hash  # noqa: PLC0415

    import scienceme_lab  # noqa: PLC0415

    make_world(sealing)
    (kept,) = [p for p in (private / "commitments").iterdir() if p.is_dir()]
    assert kept.name.startswith("generator-scienceme-lab-")
    assert package_hash(kept) == package_hash(Path(scienceme_lab.__file__).parent)
