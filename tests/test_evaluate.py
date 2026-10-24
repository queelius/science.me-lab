"""The tier 1 evaluation harness: criteria at their threshold edges, the held-out truth, and a tiny end-to-end run."""

import itertools
import math
import os
import shutil
import typing
from pathlib import Path

import numpy as np
import pytest
from helpers import NOW, SEED_PROGRAMS, at, close_schools, vaccinate
from pydantic import ValidationError
from scienceme.archive import Archive
from scienceme.claims import Verdict
from scienceme.mission import Candidate, WantedItem, append_wanted, load_mission, load_wanted
from scienceme.registry import Registry
from scienceme.model import Intervention, StreamKind, StreamSpec, Target
from scienceme.pmf import Pmf
from scienceme.scoring import PointScore, Resolution, ResolvedTarget, target_order
from scienceme.weights import Advantages, normalized_weights
from scienceme.worker import package_hash, scorer_hash
from scienceme.workspace import read_model, write_new
from scienceme.workspace import Workspace, WriteOnceError

import scienceme_lab
from helpers import DEFAULT_PUBLIC
from scienceme_lab.custodian import Custodian
from scienceme_lab.evaluate import (
    CONFIG_FILE,
    RESULT_FILE,
    SEIR_THEORIES,
    STRATEGIES,
    CriterionResult,
    EvalConfig,
    EvalReport,
    EvaluationError,
    ReplicateResult,
    Stamps,
    Thresholds,
    coverages,
    criteria,
    evaluate,
    feasible_items,
    heldout_arms,
    heldout_realized,
    heldout_targets,
    machine_fields,
    matching_program,
    matching_share,
    mixture_log_score,
    run_dir,
    run_replicate,
    stamps,
    lab_hash,
    worker_environment,
)
from scienceme_lab.generator import SUPPORT_MAX

REPO_ROOT = Path(scienceme_lab.__file__).resolve().parents[2]
LAB_SOURCE = Path(scienceme_lab.__file__).resolve().parent

X, NO_X = {"covariate-x": True, "school": False}, {"covariate-x": False, "school": True}
# Stamps for hand-built results: no real code.
STAMPS = Stamps(
    programs={"seir-x1-s0": "sha256:" + "1" * 64},
    scorer="sha256:" + "2" * 64,
    lab="sha256:" + "3" * 64,
    generators={"seir": "sha256:" + "4" * 64},
)
# The tiny end-to-end evaluation: one toy replicate, its gain, random and all runs.
TINY = {
    "generators": ("toy",),
    "replicates": 1,
    "weeks": 14,
    "jurisdictions": 2,
    "decision_issues": (5,),
    "evaluate_at": 9,
    "random_seeds": 1,
    "workers": 1,
}


def config(**overrides: typing.Any) -> EvalConfig:
    fields: dict[str, typing.Any] = {"programs": SEED_PROGRAMS, "public_root": DEFAULT_PUBLIC, "master_seed": 1}
    return EvalConfig(**(fields | overrides))


def heldout(*scores: float) -> tuple[tuple[Target, float], ...]:
    """Held-out scores on targets 0, 1, ... (horizons of one held-out arm)."""
    def target(horizon: int) -> Target:
        return Target(stream="w/adm~heldout-close-j0", jurisdiction="j0", reference_week=25, horizon=horizon)

    return tuple((target(h), v) for h, v in enumerate(scores))


def result(**overrides: typing.Any) -> ReplicateResult:
    """A hand-built run: a gain run of replicate 0 of an SEIR world with x and no school effect."""
    truth = overrides.get("truth", X)
    scores = overrides.get("heldout_scores", heldout(-3.0))
    fields: dict[str, typing.Any] = {
        "generator": "seir",
        "replicate": 0,
        "strategy": "gain",
        "seed": 0,
        "stamps": STAMPS,
        "heldout_scores": scores,
        "heldout_log_score": float(np.mean([v for _, v in scores])),
        "heldout_targets": len(scores),
        "heldout_skipped": 0,
        "weights_cumulative": {matching_program(truth): 1.0},
        "theory_advantages": {matching_program(truth): 0.0},
        "theory_share": 1.0,
        "theory_predicted": {matching_program(truth): 1},
        "theory_unscored": {matching_program(truth): 0},
        "claims": {"covariate-x": Verdict.SUPPORTS, "covariate-x-flat": Verdict.INSUFFICIENT, "school": Verdict.INSUFFICIENT},
        "truth": truth,
        "arm_coverage90": 0.9,
        "arm_points": 10,
        "baseline_coverage90": 0.9,
        "baseline_points": 10,
        "fulfilled": (),
        "skipped_decisions": (),
    }
    return ReplicateResult(**(fields | overrides))


def criterion(found: typing.Sequence[CriterionResult], name: str, generator: str | None = "seir") -> CriterionResult:
    (match,) = [c for c in found if c.name == name and c.generator == generator]
    return match


# --- criteria at their threshold edges ---


def gain_vs_random_runs(beats: int, generator: str = "seir") -> list[ReplicateResult]:
    """Ten replicates whose random seeds score -2 and -4 (mean -3) on four targets; gain beats that mean in
    `beats` of them.  A gain score of -2.5 is worse than one random seed but better than their mean; -3.0
    only ties it.
    """
    runs = []
    for replicate in range(10):
        gain = -2.5 if replicate < beats else -3.0
        runs.append(result(generator=generator, replicate=replicate, heldout_scores=heldout(*[gain] * 4)))
        for seed, score in enumerate((-2.0, -4.0)):
            scores = heldout(*[score] * 4)
            runs.append(result(generator=generator, replicate=replicate, strategy="random", seed=seed, heldout_scores=scores))
    return runs


@pytest.mark.parametrize(("beats", "passed"), [(8, True), (7, False)])
def test_gain_must_beat_the_mean_of_the_random_seeds_in_eight_of_ten_replicates(beats: int, passed: bool) -> None:
    found = criterion(criteria(config(), gain_vs_random_runs(beats)), "gain-vs-random")
    assert (found.judged, found.passed) == (True, passed)
    assert (found.detail["beats"], found.detail["incomparable"]) == (beats, 0)


def test_gain_and_random_are_compared_on_the_heldout_targets_both_scored() -> None:
    """Gain did badly on a target the random run skipped, and the random run well on one gain skipped:
    over everything each scored gain loses, on the targets both scored it wins."""
    (t0, _), (t1, _), (t2, _) = heldout(0.0, 0.0, 0.0)
    gain = result(heldout_scores=((t0, -100.0), (t1, -1.0)))
    random = result(strategy="random", heldout_scores=((t1, -2.0), (t2, 0.0)))
    assert gain.heldout_log_score < random.heldout_log_score
    found = criterion(criteria(config(), [gain, random]), "gain-vs-random")
    assert found.detail["beats"] == 1
    disjoint = result(strategy="random", heldout_scores=((t2, -9.0),))
    found = criterion(criteria(config(), [gain, disjoint, result(replicate=1)]), "gain-vs-random")
    assert (found.detail["beats"], found.detail["replicates"], found.detail["incomparable"]) == (0, 2, 2)


def test_gain_beats_random_by_exact_means_whatever_order_the_targets_come_in() -> None:
    """Scores of 1e16, 1 and -1e16 sum to 1 exactly, but to 0 when the 1 is added next to a 1e16 in floats.

    Each replicate gives gain the three scores in another order over the same three targets, so a sum that
    follows the targets' iteration order loses the 1 in four of the six orders, whatever the hash seed makes
    of a set's order.  Exactly, gain's mean of 1/3 beats the random seeds' mean of means, 1/6 (the means of
    0.5 and 0), in all six.  A seventh replicate ties exactly, and a tie is not a win.
    """
    t0, t1, t2 = (t for t, _ in heldout(0.0, 0.0, 0.0))
    runs = []
    for replicate, order in enumerate(itertools.permutations((1e16, 1.0, -1e16))):
        runs.append(result(replicate=replicate, heldout_scores=tuple(zip((t0, t1, t2), order))))
        for seed, scores in enumerate(((0.5, 0.0, 0.0), (0.25, 0.25, 0.0))):
            runs.append(result(replicate=replicate, strategy="random", seed=seed, heldout_scores=tuple(zip((t0, t1, t2), scores))))
    runs.append(result(replicate=6, heldout_scores=((t2, -1e16), (t0, 1e16), (t1, 1.0))))
    runs.append(result(replicate=6, strategy="random", heldout_scores=((t0, 1.0), (t1, 0.0), (t2, 0.0))))
    runs.append(result(replicate=6, strategy="all", heldout_scores=((t0, 9.0), (t1, 9.0), (t2, 9.0))))  # not compared
    found = criterion(criteria(config(), runs), "gain-vs-random")
    assert (found.detail["beats"], found.detail["replicates"], found.detail["incomparable"]) == (6, 7, 0)


def concentration_runs(strategy: str, concentrated: int, undefined: int = 0) -> list[ReplicateResult]:
    """Ten runs of `strategy`: in `concentrated` the matching theory's restricted share is exactly one half, in
    the next `undefined` it is undefined, in the rest just under one half.  The runs of the other strategies,
    none of them concentrated, do not count."""
    runs = []
    for replicate in range(10):
        share = 0.5 if replicate < concentrated else None if replicate < concentrated + undefined else 0.4999
        runs.append(result(replicate=replicate, strategy=strategy, theory_share=share))
        runs += [result(replicate=replicate, strategy=s, theory_share=0.0) for s in STRATEGIES if s != strategy]
    return runs


EDGES = [(8, 0, True), (7, 0, False), (8, 2, True), (7, 3, False)]


@pytest.mark.parametrize(("concentrated", "undefined", "passed"), EDGES)
def test_the_matching_theory_must_hold_half_the_theories_weight_in_eight_runs_with_every_arm(
    concentrated: int, undefined: int, passed: bool
) -> None:
    """Concentration is judged with unlimited arms (spec 6.3): on the all runs."""
    found = criterion(criteria(config(), concentration_runs("all", concentrated, undefined)), "concentration")
    assert (found.judged, found.passed) == (True, passed)
    assert (found.detail["concentrated"], found.detail["undefined"], found.detail["runs"]) == (concentrated, undefined, 10)


@pytest.mark.parametrize(("concentrated", "undefined", "passed"), EDGES)
def test_the_gain_runs_budget_limited_concentration_is_reported_not_judged(
    concentrated: int, undefined: int, passed: bool
) -> None:
    found = criterion(criteria(config(), concentration_runs("gain", concentrated, undefined)), "concentration-gain")
    assert (found.judged, found.passed) == (False, passed)
    assert (found.detail["concentrated"], found.detail["undefined"], found.detail["runs"]) == (concentrated, undefined, 10)


def test_the_matching_share_is_a_softmax_restricted_to_the_theories() -> None:
    """A baseline far ahead of every theory underflows their weights over all programs to zero; the share
    among the theories alone is still defined."""
    tau = 4.0
    advantages = {"hub-baseline": 0.0, "seir-x1-s0": -4000.0, "seir-x1-s1": -4000.0 - tau * math.log(3)}
    theories = list(SEIR_THEORIES)
    assert all(w == 0.0 for p, w in normalized_weights(advantages, [*advantages], tau).items() if p != "hub-baseline")
    # the two theories without an advantage count 0, as newcomers do, so they take all the weight here
    assert matching_share(advantages, theories, X, tau) == 0.0
    scored = {p: a for p, a in advantages.items() if p != "hub-baseline"} | {"seir-x0-s1": -5000.0, "seir-x0-s0": -5000.0}
    assert matching_share(scored, theories, X, tau) == pytest.approx(0.75)
    assert matching_share({"hub-baseline": 0.0}, theories, X, tau) is None  # no theory scored
    assert matching_share(scored, ["seir-x0-s1", "seir-x0-s0"], X, tau) is None  # the match is not frozen


def test_a_matching_theory_that_never_scored_has_no_share() -> None:
    """Rivals that scored below the mixture must not hand the share to a match that failed every target."""
    rivals = {p: -30.0 for p in SEIR_THEORIES if p != matching_program(X)}
    assert matching_share(rivals, SEIR_THEORIES, X, 4.0) is None
    assert matching_share(rivals | {matching_program(X): -30.0}, SEIR_THEORIES, X, 4.0) == pytest.approx(0.25)


def test_the_four_theories_and_the_matching_one_follow_the_truth() -> None:
    assert SEIR_THEORIES == ("seir-x1-s1", "seir-x1-s0", "seir-x0-s1", "seir-x0-s0")
    assert matching_program({"covariate-x": True, "school": False}) == "seir-x1-s0"
    assert matching_program({"covariate-x": False, "school": True}) == "seir-x0-s1"


def surgery_runs(generator: str, baseline: float) -> list[ReplicateResult]:
    """A gain run and a random run whose arm coverages pool, weighted by their points, to 0.625 (unweighted
    they average 0.5): surgery pools both strategies."""
    return [
        result(generator=generator, replicate=0, arm_coverage90=0.25, arm_points=1, baseline_coverage90=baseline, baseline_points=4),
        result(
            generator=generator,
            replicate=0,
            strategy="random",
            arm_coverage90=0.75,
            arm_points=3,
            baseline_coverage90=None,
            baseline_points=0,
        ),
    ]


@pytest.mark.parametrize(("baseline", "passed"), [(0.875, True), (0.375, True), (0.9375, False), (0.25, False)])
def test_surgery_compares_the_pooled_arm_and_baseline_coverage_within_the_tolerance(baseline: float, passed: bool) -> None:
    cfg = config(thresholds=Thresholds(surgery_tolerance=0.25))
    found = criterion(criteria(cfg, surgery_runs("seir", baseline)), "surgery")
    assert (found.judged, found.passed) == (True, passed)
    assert found.detail["arm_coverage90"] == 0.625
    assert found.detail["baseline_coverage90"] == baseline
    assert (found.detail["arm_points"], found.detail["baseline_points"]) == (4, 4)


def test_surgery_is_judged_for_seir_and_only_reported_for_abm() -> None:
    cfg = config(thresholds=Thresholds(surgery_tolerance=0.25))
    found = criteria(cfg, surgery_runs("seir", 0.25) + surgery_runs("abm", 0.25))
    assert (criterion(found, "surgery", "seir").judged, criterion(found, "surgery", "abm").judged) == (True, False)
    assert not criterion(found, "surgery", "abm").passed


def test_surgery_with_no_arm_points_fails() -> None:
    runs = [result(arm_coverage90=None, arm_points=0)]
    found = criterion(criteria(config(), runs), "surgery")
    assert not found.passed
    assert "arm_coverage90" not in found.detail


def power_runs(supports: int, contradicts: int, abstains: int) -> list[ReplicateResult]:
    """Ten gain runs over two generators: five with x (in `supports` the x claim is supported), five without
    (in `contradicts` it is contradicted); the flat claim abstains in `abstains` of the ten.  The random runs
    say the opposite and must not count."""
    runs = []
    for i in range(10):
        has_x = i < 5
        right = (i < supports) if has_x else (i - 5 < contradicts)
        verdict = (Verdict.SUPPORTS if has_x else Verdict.CONTRADICTS) if right else Verdict.INSUFFICIENT
        claims = {"covariate-x": verdict, "covariate-x-flat": Verdict.INSUFFICIENT if i < abstains else Verdict.SUPPORTS}
        generator = ("seir", "abm")[i % 2]
        runs.append(result(generator=generator, replicate=i, truth=X if has_x else NO_X, claims=claims))
        wrong = {"covariate-x": Verdict.MIXED, "covariate-x-flat": Verdict.MIXED}
        runs.append(result(generator=generator, replicate=i, strategy="random", truth=X if has_x else NO_X, claims=wrong))
    return runs


@pytest.mark.parametrize(
    ("supports", "contradicts", "abstains", "passed"),
    [(4, 4, 9, True), (3, 5, 10, False), (5, 3, 10, False), (5, 5, 8, False)],
)
def test_power_pools_the_gain_runs_of_every_generator(supports: int, contradicts: int, abstains: int, passed: bool) -> None:
    found = criterion(criteria(config(), power_runs(supports, contradicts, abstains)), "power", None)
    assert (found.judged, found.passed) == (True, passed)
    assert found.detail["p_supports_given_x"] == supports / 5
    assert found.detail["p_contradicts_given_no_x"] == contradicts / 5
    assert found.detail["p_abstains_flat"] == abstains / 10


def test_power_without_a_world_lacking_x_fails() -> None:
    runs = [result(replicate=i) for i in range(10)]
    found = criterion(criteria(config(), runs), "power", None)
    assert not found.passed
    assert "p_contradicts_given_no_x" not in found.detail


def test_criteria_come_per_generator_then_power() -> None:
    found = criteria(config(), [])
    assert [(c.name, c.generator, c.judged) for c in found] == [
        ("gain-vs-random", "seir", True),
        ("concentration", "seir", True),
        ("concentration-gain", "seir", False),
        ("surgery", "seir", True),
        ("gain-vs-random", "abm", True),
        ("concentration", "abm", True),
        ("concentration-gain", "abm", False),
        ("surgery", "abm", False),
        ("power", None, True),
    ]


def test_the_report_passes_when_every_judged_criterion_passes() -> None:
    def report(*found: CriterionResult) -> EvalReport:
        return EvalReport(config=config(), stamps=STAMPS, results=(), criteria=found)

    judged = CriterionResult(name="a", generator="seir", judged=True, passed=True, detail={})
    reported = CriterionResult(name="b", generator="abm", judged=False, passed=False, detail={})
    assert report(judged, reported).passed
    assert not report(judged.model_copy(update={"passed": False}), reported).passed


def test_the_default_thresholds_are_the_preregistered_ones() -> None:
    assert Thresholds() == Thresholds(
        gain_beats_random_min=8,
        concentration_share=0.5,
        concentration_min=8,
        surgery_tolerance=0.10,
        power_min=0.8,
        abstain_min=0.9,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"generators": ()},
        {"generators": ("toy", "toy")},
        {"decision_issues": (6, 24)},
        {"decision_issues": (0,)},
        {"evaluate_at": 26},
        {"replicates": 0},
        {"random_seeds": 0},
        {"workers": 0},
    ],
)
def test_a_config_the_harness_cannot_run_is_refused(overrides: dict[str, typing.Any]) -> None:
    with pytest.raises(ValidationError):
        config(**overrides)


def test_the_default_config_is_the_tier_1_design() -> None:
    cfg = config()
    assert (cfg.generators, cfg.replicates, cfg.weeks, cfg.jurisdictions) == (("seir", "abm"), 10, 30, 4)
    assert (cfg.decision_issues, cfg.evaluate_at, cfg.random_seeds, cfg.workers) == ((6, 9, 12), 24, 2, 10)


# --- the pieces of a run ---


@pytest.fixture
def custodian(public: Workspace, private: Path) -> Custodian:
    return Custodian(public, private, log=None)


def test_feasible_items_start_after_the_current_issue_and_before_the_world_ends(
    custodian: Custodian, public: Workspace
) -> None:
    custodian.create("toy", "w", 14, 2, "admissions", NOW, rng=np.random.default_rng(3))
    items = load_wanted(public, "w")
    starts = {item.id: item.candidate.stream.intervention.start_week for item in items}
    assert sorted({starts[i.id] for i in feasible_items(items, set(), 5, 14)}) == [8, 12]
    assert sorted({starts[i.id] for i in feasible_items(items, set(), 8, 14)}) == [12]
    assert feasible_items(items, set(), 12, 14) == []
    chosen = feasible_items(items, {"close-j0-w8"}, 5, 14)
    assert "close-j0-w8" not in [i.id for i in chosen]
    assert len(chosen) == 7
    assert chosen == [i for i in items if i in chosen]  # in wanted-list order


def test_heldout_arms_close_schools_and_vaccinate_in_each_jurisdiction(custodian: Custodian, public: Workspace) -> None:
    custodian.create("toy", "w", 30, 3, "admissions", NOW, rng=np.random.default_rng(3))
    base = Archive(public).spec("w/admissions")
    arms = heldout_arms(base, 24)
    assert [a.id for a in arms] == [f"w/admissions~heldout-{kind}-{j}" for j in ("j0", "j1", "j2") for kind in ("close", "vacc")]
    by_id = {a.id: a for a in arms}
    close, vacc = by_id["w/admissions~heldout-close-j1"], by_id["w/admissions~heldout-vacc-j1"]
    assert close.intervention == Intervention(
        variable="schools_closed", value=1.0, start_week=25, end_week=27, jurisdictions=("j1",)
    )
    assert vacc.intervention == Intervention(variable="vaccinate_fraction", value=0.3, start_week=25, jurisdictions=("j1",))
    for arm in arms:
        assert (arm.parent, arm.first_issue, arm.jurisdictions) == (base.id, 25, base.jurisdictions)
        assert arm.model_dump(exclude={"id", "intervention", "parent", "first_issue"}) == base.model_dump(
            exclude={"id", "intervention", "parent", "first_issue"}
        )


def test_heldout_targets_are_where_the_arm_intervenes_past_the_mission_window(
    custodian: Custodian, public: Workspace
) -> None:
    custodian.create("toy", "w", 30, 3, "admissions", NOW, rng=np.random.default_rng(3))
    mission = load_mission(public, "w")
    arms = heldout_arms(Archive(public).spec("w/admissions"), 24)
    assert 25 > mission.last_issue  # the mission itself asks nothing at week 25
    for arm in arms:
        targets = heldout_targets(arm, mission, 24)
        assert arm.intervention is not None and arm.intervention.jurisdictions is not None
        assert [(t.jurisdiction, t.horizon) for t in targets] == [(arm.intervention.jurisdictions[0], h) for h in range(4)]
        assert all((t.stream, t.reference_week, t.resolution_lag) == (arm.id, 25, mission.resolution_lag) for t in targets)
    everywhere = arms[0].model_copy(update={"intervention": vaccinate(25, 0.3, where=None)})
    assert [t.jurisdiction for t in heldout_targets(everywhere, mission, 24)] == [j for j in ("j0", "j1", "j2") for _ in range(4)]


@pytest.mark.parametrize("kind", ["close", "vacc"])
def test_the_heldout_truth_is_what_the_resolution_vintage_shows(
    custodian: Custodian, public: Workspace, kind: str
) -> None:
    """Fulfilled as a wanted item and published, the arm's resolution vintages hold exactly the harness's truth."""
    evaluate_at = 12
    custodian.create("toy", "w", 30, 3, "admissions", NOW, rng=np.random.default_rng(7))
    custodian.tick("w", evaluate_at, at(1))
    mission = load_mission(public, "w")
    arm = next(a for a in heldout_arms(Archive(public).spec("w/admissions"), evaluate_at) if a.id.endswith(f"{kind}-j1"))
    item_id = arm.id.split("~")[1]
    append_wanted(public, "w", WantedItem(id=item_id, proposed_by="test", candidate=Candidate(stream=arm, targets=())))
    custodian.fulfill("w", item_id, at(2))
    targets = heldout_targets(arm, mission, evaluate_at)
    custodian.tick("w", max(t.resolution_issue for t in targets), at(3))
    published = [Archive(public).realized(t) for t in targets]
    assert heldout_realized(custodian.world("w"), arm, targets) == published
    assert all(isinstance(y, int) for y in published)


def test_an_evaluation_refuses_a_scratch_directory_holding_its_runs_before_running_any(tmp_path: Path) -> None:
    cfg = config(**TINY)
    run_dir(tmp_path, "toy", 0, "random", 0).mkdir()
    with pytest.raises(WriteOnceError, match="toy-r0-random-s0"):
        evaluate(cfg, tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["toy-r0-random-s0"]


def test_coverage_pairs_the_arms_intervened_targets_with_the_same_baseline_targets() -> None:
    base = StreamSpec(
        id="w/adm", world="w", role="admissions", kind=StreamKind.COUNT, jurisdictions=("j0", "j1"), support_max=9
    )
    other = base.model_copy(update={"id": "w/ed", "role": "ed-visits"})
    arm = base.model_copy(update={"id": "w/adm~a", "parent": base.id, "intervention": close_schools(8, 10)})  # in j1
    inside, outside = PointScore(log_p=-1.0, cdf_below=0.4, cdf_at=0.6), PointScore(log_p=-5.0, cdf_below=0.96, cdf_at=0.99)

    def resolved(stream: StreamSpec, jurisdiction: str, week: int, horizon: int, scores: dict[str, PointScore]) -> ResolvedTarget:
        target = Target(stream=stream.id, jurisdiction=jurisdiction, reference_week=week, horizon=horizon)
        return ResolvedTarget(target=target, realized=1, scores=scores)

    resolution = Resolution(
        resolved=(
            resolved(arm, "j1", 8, 0, {"p": inside}),  # intervened: (j1, origin 8, horizon 0) and (j1, 8, 1)
            resolved(arm, "j1", 8, 1, {"p": outside}),
            resolved(arm, "j0", 8, 0, {"p": outside}),  # the arm where it does not intervene: left out
            resolved(base, "j1", 8, 0, {"p": inside, "q": inside}),  # the baseline's same targets
            resolved(base, "j1", 8, 1, {"p": inside}),
            resolved(base, "j1", 7, 1, {"p": outside}),  # the same target week from another origin: not a pair
            resolved(base, "j1", 9, 0, {"p": outside}),
            resolved(base, "j0", 8, 0, {"p": outside}),  # a jurisdiction no arm intervened in
            resolved(base, "j1", 8, 3, {"p": outside}),  # a horizon no arm point has
            resolved(other, "j1", 8, 0, {"p": outside}),  # another channel
        ),
        pending=0,
        predicted={},
        unscored={},
    )
    specs = {s.id: s for s in (base, other, arm)}
    arms, baseline = coverages(resolution, specs, base.id, "p")
    assert (arms.coverage_90, arms.n, baseline.coverage_90, baseline.n) == (0.5, 2, 1.0, 2)
    arms, baseline = coverages(resolution, specs, base.id, "q")  # no arm point, so no baseline pair either
    assert (arms.coverage_90, arms.n, baseline.coverage_90, baseline.n) == (None, 0, None, 0)
    arms, baseline = coverages(resolution, specs, base.id, None)
    assert (arms.n, baseline.n) == (0, 0)


def test_the_heldout_mixture_weighs_the_awake_programs_by_their_current_advantage() -> None:
    """Current advantages favor b 3:1 over a; cumulative ones favor a, and the asleep c leads both."""
    tau, eps = 4.0, 0.01
    pmfs = {"a": Pmf.from_probs(10, [0.5, 0.5]), "b": Pmf.point(20)}
    advantages = Advantages(
        cumulative={"a": 100.0, "b": 0.0, "c": 1000.0}, current={"a": 0.0, "b": tau * math.log(3), "c": 1000.0}, steps=()
    )
    floor = eps / (SUPPORT_MAX + 1)
    assert mixture_log_score(pmfs, advantages, tau, eps, 20, SUPPORT_MAX) == pytest.approx(math.log((1 - eps) * 0.75 + floor))
    assert mixture_log_score(pmfs, advantages, tau, eps, 10, SUPPORT_MAX) == pytest.approx(math.log((1 - eps) * 0.125 + floor))
    assert mixture_log_score(pmfs, advantages, tau, eps, 500, SUPPORT_MAX) == pytest.approx(math.log(floor))  # only the floor
    alone = {"b": pmfs["b"]}  # the mixture is over the awake programs only
    assert mixture_log_score(alone, advantages, tau, eps, 20, SUPPORT_MAX) == pytest.approx(math.log((1 - eps) + floor))
    assert mixture_log_score(pmfs, advantages, tau, eps, 500, 99) == pytest.approx(math.log(eps / 100))  # the stream's bound


# A 12-week single-jurisdiction world: only the wanted items starting in week 8 can be fulfilled (week 12 is its end).
SMALL = {"generators": ("toy",), "replicates": 1, "weeks": 12, "jurisdictions": 1, "evaluate_at": 7, "workers": 1}


def test_a_decision_with_no_feasible_item_is_skipped_and_recorded(tmp_path: Path) -> None:
    cfg = config(**SMALL, decision_issues=(4, 5, 6))
    run = run_replicate(cfg, "toy", 0, "random", 0, tmp_path)
    assert sorted(run.fulfilled) == ["close-j0-w8", "vacc-j0-w8"]
    assert run.skipped_decisions == (6,)
    public = Workspace(run_dir(tmp_path, "toy", 0, "random", 0) / "public")
    arms = {s.fulfills: s.first_issue for s in Archive(public).specs() if s.fulfills is not None}
    assert arms == {run.fulfilled[0]: 5, run.fulfilled[1]: 6}
    assert math.isfinite(run.heldout_log_score) and run.heldout_targets == len(run.heldout_scores) == 2 * 4
    assert read_model(run_dir(tmp_path, "toy", 0, "random", 0) / RESULT_FILE, ReplicateResult) == run
    assert not public.scores.exists() and public.archive.is_dir()  # the forecast cache goes; its inputs stay


def test_the_all_run_fulfills_every_feasible_item_at_the_first_decision_and_decides_no_more(tmp_path: Path) -> None:
    cfg = config(**SMALL, decision_issues=(5, 4, 6))  # the first decision issue is the earliest, 4
    run = run_replicate(cfg, "toy", 0, "all", 0, tmp_path)
    assert run.strategy == "all"
    assert run.fulfilled == ("close-j0-w8", "vacc-j0-w8")  # every item feasible at issue 4, in wanted-list order
    assert run.skipped_decisions == ()  # issues 5 and 6 are no decisions for it
    public = Workspace(run_dir(tmp_path, "toy", 0, "all", 0) / "public")
    arms = {s.fulfills: s.first_issue for s in Archive(public).specs() if s.fulfills is not None}
    assert arms == {"close-j0-w8": 5, "vacc-j0-w8": 5}


def test_an_all_run_without_decision_issues_observes_nothing(tmp_path: Path) -> None:
    run = run_replicate(config(**SMALL, decision_issues=()), "toy", 0, "all", 0, tmp_path)
    assert (run.fulfilled, run.skipped_decisions) == ((), ())


def test_a_run_where_no_program_can_forecast_the_heldout_arms_is_an_error(tmp_path: Path) -> None:
    cfg = config(**SMALL, decision_issues=(), programs=("lens-admissions", "hub-baseline"))
    with pytest.raises(EvaluationError, match="no program forecast any of the 8 held-out targets"):
        run_replicate(cfg, "toy", 0, "random", 0, tmp_path)


@pytest.mark.parametrize(("generator", "strategy"), [("abm", "gain"), ("toy", "greedy")])
def test_a_run_of_an_unconfigured_generator_or_unknown_strategy_is_refused(
    tmp_path: Path, generator: str, strategy: str
) -> None:
    with pytest.raises(ValueError, match=generator if generator == "abm" else strategy):
        run_replicate(config(**SMALL, decision_issues=()), generator, 0, strategy, 0, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_evaluation_refuses_the_runs_of_another_config(tmp_path: Path) -> None:
    write_new(tmp_path / CONFIG_FILE, config(**TINY))
    with pytest.raises(WriteOnceError, match="another evaluation config"):
        evaluate(config(**TINY, master_seed=2), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [CONFIG_FILE]


def test_an_evaluation_refuses_a_result_left_by_another_run(tmp_path: Path) -> None:
    write_new(tmp_path / CONFIG_FILE, config(**TINY))
    write_new(run_dir(tmp_path, "toy", 0, "gain", 0) / RESULT_FILE, result(generator="toy", seed=4))
    with pytest.raises(EvaluationError, match="another run"):
        evaluate(config(**TINY), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([CONFIG_FILE, "toy-r0-gain-s0"])


def test_run_directories_are_distinct_per_run(tmp_path: Path) -> None:
    dirs = {run_dir(tmp_path, g, r, s, seed) for g in ("seir", "abm") for r in (0, 1) for s in STRATEGIES for seed in (0, 1)}
    assert len(dirs) == 24
    assert all(d.parent == tmp_path for d in dirs)


def test_workers_get_one_blas_thread_and_the_environment_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with worker_environment():
        assert (os.environ["OPENBLAS_NUM_THREADS"], os.environ["OMP_NUM_THREADS"]) == ("1", "1")
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


# --- the code a run evaluates ---


@pytest.fixture
def public_copy(tmp_path: Path) -> Path:
    """A copy of the public repository's programs and registry: the seed programs, frozen."""
    root = tmp_path / "public-copy"
    shutil.copytree(DEFAULT_PUBLIC / "programs", root / "programs", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(DEFAULT_PUBLIC / "registry", root / "registry")
    return root


def test_the_stamps_are_the_public_freezes_the_scorer_this_package_and_the_generators(public_copy: Path) -> None:
    cfg = config(public_root=public_copy, generators=("toy",))
    found = stamps(cfg)
    public = Registry(Workspace(public_copy))
    assert found.programs == {name: public.by_name(name).closure_hash for name in SEED_PROGRAMS}
    assert list(found.programs) == list(SEED_PROGRAMS)
    assert found.scorer == scorer_hash()
    assert found.lab == lab_hash() == package_hash(LAB_SOURCE)
    assert found.generators == {"toy": package_hash(LAB_SOURCE)}  # the toy generator lives in this package


def test_a_program_not_frozen_in_the_public_registry_is_refused(public_copy: Path, tmp_path: Path) -> None:
    shutil.copytree(public_copy / "programs" / "hub-baseline", public_copy / "programs" / "hub-draft")
    manifest = public_copy / "programs" / "hub-draft" / "program.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"hub-baseline"', '"hub-draft"'), encoding="utf-8")
    cfg = config(**SMALL, decision_issues=(), public_root=public_copy, programs=(*SEED_PROGRAMS, "hub-draft"))
    with pytest.raises(EvaluationError, match="'hub-draft' is not frozen in the public registry"):
        stamps(cfg)
    with pytest.raises(EvaluationError, match="'hub-draft' is not frozen in the public registry"):
        run_replicate(cfg, "toy", 0, "gain", 0, tmp_path / "scratch")
    with pytest.raises(EvaluationError, match="'hub-draft' is not frozen"):
        evaluate(cfg, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()  # refused before anything is written


def test_a_program_edited_after_its_public_freeze_is_refused(public_copy: Path, tmp_path: Path) -> None:
    cfg = config(**SMALL, decision_issues=(), public_root=public_copy)
    with (public_copy / "programs" / "hub-baseline" / "program.py").open("a", encoding="utf-8") as f:
        f.write("# edited after its freeze\n")
    with pytest.raises(EvaluationError, match="'hub-baseline' is frozen as sha256:[0-9a-f]+ in .*, but its copy hashes to"):
        run_replicate(cfg, "toy", 0, "gain", 0, tmp_path / "scratch")
    with pytest.raises(EvaluationError, match="hub-baseline.*modified-after-freeze"):
        evaluate(cfg, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def stamped(cfg: EvalConfig, **overrides: typing.Any) -> list[ReplicateResult]:
    """One hand-built result per run of `cfg` (TINY: gain, random, all), stamped with the current code."""
    current = stamps(cfg)
    return [result(generator="toy", strategy=strategy, stamps=current, **overrides) for strategy in STRATEGIES]


def test_an_evaluation_reuses_the_results_its_runs_left(tmp_path: Path, public_copy: Path) -> None:
    cfg = config(**TINY)
    finished = stamped(cfg)
    write_new(tmp_path / CONFIG_FILE, cfg)
    for run in finished:
        write_new(run_dir(tmp_path, run.generator, run.replicate, run.strategy, run.seed) / RESULT_FILE, run)
    report = evaluate(cfg, tmp_path)  # nothing left to run
    assert report.results == tuple(finished) and report.stamps == stamps(cfg)
    assert evaluate(cfg.model_copy(update={"workers": 3}), tmp_path).results == tuple(finished)
    elsewhere = cfg.model_copy(update={"public_root": public_copy})  # another checkout of the same freezes
    assert evaluate(elsewhere, tmp_path).results == tuple(finished)


def test_machine_fields_say_where_and_how_fast_not_what() -> None:
    assert machine_fields() == ("public_root", "workers")


def test_an_evaluation_never_adopts_results_no_recorded_config_vouches_for(tmp_path: Path) -> None:
    cfg = config(**TINY)
    run = stamped(cfg)[0]
    write_new(run_dir(tmp_path, run.generator, run.replicate, run.strategy, run.seed) / RESULT_FILE, run)
    with pytest.raises(WriteOnceError, match="refusing to adopt 1 results .* no evaluation config is recorded"):
        evaluate(cfg, tmp_path)
    assert not (tmp_path / CONFIG_FILE).exists()


@pytest.mark.parametrize("field", ["programs", "scorer", "lab", "generators"])
def test_an_evaluation_refuses_a_result_of_other_code(tmp_path: Path, field: str) -> None:
    cfg = config(**TINY)
    current = stamps(cfg)
    other = {"programs": {**current.programs, "hub-baseline": "sha256:" + "0" * 64}, "scorer": "sha256:" + "0" * 64,
             "lab": "sha256:" + "0" * 64, "generators": {"toy": "sha256:" + "0" * 64}}[field]
    results = stamped(cfg)
    results[1] = results[1].model_copy(update={"stamps": current.model_copy(update={field: other})})
    write_new(tmp_path / CONFIG_FILE, cfg)
    for run in results:
        write_new(run_dir(tmp_path, run.generator, run.replicate, run.strategy, run.seed) / RESULT_FILE, run)
    with pytest.raises(EvaluationError, match=f"evaluated other code than the current one: its {field} stamps differ"):
        evaluate(cfg, tmp_path)


def test_a_report_holds_only_results_of_its_own_code() -> None:
    other = result(stamps=STAMPS.model_copy(update={"scorer": "sha256:" + "9" * 64}))
    with pytest.raises(ValidationError, match="the report's stamps"):
        EvalReport(config=config(), stamps=STAMPS, results=(result(), other), criteria=())


# --- a tiny end-to-end evaluation ---


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[EvalConfig, Path, EvalReport]:
    """The tiny evaluation, run once for the module: (config, scratch directory, report)."""
    scratch = tmp_path_factory.mktemp("scratch")
    cfg = config(**TINY, master_seed=20261023)
    return cfg, scratch, evaluate(cfg, scratch)


def test_a_tiny_evaluation_runs_a_gain_a_random_and_an_all_run(tiny: tuple[EvalConfig, Path, EvalReport]) -> None:
    cfg, _, report = tiny
    assert report.config == cfg
    assert [(r.generator, r.replicate, r.strategy, r.seed) for r in report.results] == [
        ("toy", 0, "gain", 0),
        ("toy", 0, "random", 0),
        ("toy", 0, "all", 0),
    ]
    gain, random, every = report.results
    assert report.stamps == stamps(cfg) and all(run.stamps == report.stamps for run in report.results)
    assert list(report.stamps.programs) == list(cfg.programs)
    for run in report.results:
        assert math.isfinite(run.heldout_log_score)
        assert run.heldout_targets == len(run.heldout_scores) == (2 * 2) * 4  # 2 arms per jurisdiction, where each acts
        assert [t for t, _ in run.heldout_scores] == sorted((t for t, _ in run.heldout_scores), key=target_order)
        assert set(run.theory_advantages) == set(SEIR_THEORIES) and run.theory_share is not None
        assert set(run.theory_predicted) == set(run.theory_unscored) == set(SEIR_THEORIES)
        assert all(n > 0 for n in run.theory_predicted.values()) and all(n >= 0 for n in run.theory_unscored.values())
        assert run.heldout_skipped == 0
        assert set(run.claims) == {"covariate-x", "covariate-x-flat", "school"}
        assert set(SEIR_THEORIES) <= set(run.weights_cumulative)
        assert math.isclose(sum(run.weights_cumulative.values()), 1.0)
        assert run.skipped_decisions == ()
        # no arm target resolves by issue 9 (the arms start in week 8 or 12), so there is no intervened unit
        # and no baseline unit to set against it
        assert (run.arm_coverage90, run.arm_points, run.baseline_coverage90, run.baseline_points) == (None, 0, None, 0)
    assert len(gain.fulfilled) == len(random.fulfilled) == 1
    # the all run fulfills every item feasible at issue 5: both jurisdictions, both kinds, starts 8 and 12
    assert len(every.fulfilled) == 2 * 2 * 2 and set(gain.fulfilled + random.fulfilled) <= set(every.fulfilled)
    assert gain.truth == random.truth == every.truth  # matched: the same world
    assert [c.name for c in report.criteria] == ["gain-vs-random", "concentration", "concentration-gain", "surgery", "power"]
    assert EvalReport.model_validate_json(report.model_dump_json()) == report


def test_every_run_of_a_replicate_publishes_byte_identical_baseline_vintages(
    tiny: tuple[EvalConfig, Path, EvalReport],
) -> None:
    _, scratch, report = tiny
    gain, random, every = (run_dir(scratch, r.generator, r.replicate, r.strategy, r.seed) for r in report.results)

    def published(run: Path, stream: str) -> dict[str, bytes]:
        directory = run / "public" / "archive" / "toy-r0" / stream
        return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}

    expected = {"admissions": 1 + 9, "x": 1 + 9, "term": 1 + 1}  # the spec and issues 1 to 9; the calendar at 0
    for stream, files in expected.items():
        assert len(published(gain, stream)) == files
        assert published(random, stream) == published(every, stream) == published(gain, stream)


def test_the_tiny_runs_left_their_results_and_a_second_evaluation_reuses_them(
    tiny: tuple[EvalConfig, Path, EvalReport],
) -> None:
    cfg, scratch, report = tiny
    files = [run_dir(scratch, r.generator, r.replicate, r.strategy, r.seed) / RESULT_FILE for r in report.results]
    assert [read_model(f, ReplicateResult) for f in files] == list(report.results)
    stamps = [f.stat().st_mtime_ns for f in files]
    assert evaluate(cfg, scratch) == report
    assert [f.stat().st_mtime_ns for f in files] == stamps


def test_a_run_never_reuses_a_directory(tiny: tuple[EvalConfig, Path, EvalReport]) -> None:
    cfg, scratch, _ = tiny
    with pytest.raises(FileExistsError):
        run_replicate(cfg, "toy", 0, "gain", 0, scratch)


# --- the public side of the pre-registration ---


def test_the_preregistration_parses_here() -> None:
    cfg = read_model(REPO_ROOT / "preregistrations" / "tier1.json", EvalConfig)
    assert cfg.generators == ("seir", "abm")  # private generators, not installed here, and the config still parses


def test_an_uninstalled_generator_is_refused_before_any_run(tmp_path: Path) -> None:
    cfg = config(**TINY | {"generators": ("toy", "absent")})
    for call in (lambda: stamps(cfg), lambda: evaluate(cfg, tmp_path / "scratch"), lambda: run_replicate(cfg, "absent", 0, "gain", 0, tmp_path)):
        with pytest.raises(EvaluationError, match=r"\['absent'\] are not installed.*toy"):
            call()
    assert list(tmp_path.iterdir()) == []


def test_a_heldout_target_resolving_after_the_last_week_is_still_realized(custodian: Custodian, public: Workspace) -> None:
    world = custodian.create("toy", "w", 14, 2, "admissions", NOW, rng=np.random.default_rng(3))
    mission = load_mission(public, "w")
    (arm, *_) = heldout_arms(Archive(public).spec("w/admissions"), 9)
    targets = heldout_targets(arm, mission, 9)
    assert max(t.resolution_issue for t in targets) > 14
    final = {(r.jurisdiction, r.week): r.value for r in world.vintage(arm, 14).rows}  # toy counts are final at once
    assert heldout_realized(world, arm, targets) == [int(final[t.jurisdiction, t.target_week]) for t in targets]
