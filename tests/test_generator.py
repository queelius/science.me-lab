from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scienceme.model import Intervention, StreamSpec, Vintage

import scienceme_lab
from scienceme_lab import generator as generator_module
from scienceme_lab.contract import CONTRACT, arms_match_the_baseline_before_they_start, draws_reproducibly
from scienceme_lab.generator import (
    GENERATORS_GROUP,
    GeneratorError,
    generator_of,
    generator_source,
    generators,
    source_name,
)
from scienceme_lab.toy import GENERATOR, ToyConfig, ToyGenerator, ToyWorld


def test_the_toy_generator_is_installed() -> None:
    found = generators()
    assert found["toy"].name == "toy"


@pytest.mark.parametrize("check", CONTRACT, ids=lambda c: c.__name__)
def test_the_toy_generator_keeps_the_contract(check, tmp_path: Path) -> None:
    check(GENERATOR, tmp_path)


class _LeakyWorld(ToyWorld):
    """A toy world whose arms differ from the baseline from week 0: the matched comparison is broken."""

    def vintage(self, spec: StreamSpec, issue: int) -> Vintage:
        published = super().vintage(spec, issue)
        if spec.intervention is None:
            return published
        rows = tuple(r.model_copy(update={"value": r.value + 1}) for r in published.rows)
        return published.model_copy(update={"rows": rows})


class _LeakyGenerator(ToyGenerator):
    world_class = _LeakyWorld


def test_a_broken_generator_fails_the_contract_check_it_breaks(tmp_path: Path) -> None:
    leaky = _LeakyGenerator()
    draws_reproducibly(leaky, tmp_path / "a")
    with pytest.raises(AssertionError, match="baseline"):
        arms_match_the_baseline_before_they_start(leaky, tmp_path / "b")


def test_an_entry_point_must_carry_its_generators_name(monkeypatch: pytest.MonkeyPatch) -> None:
    misnamed = EntryPoint(name="other", value="scienceme_lab.toy:GENERATOR", group=GENERATORS_GROUP)
    monkeypatch.setattr(generator_module, "entry_points", lambda group: [misnamed])
    with pytest.raises(GeneratorError, match="other.*toy"):
        generators()


@pytest.mark.parametrize("first, second", [("first-dist", "second-dist"), (None, "second-dist"), ("first-dist", None)])
def test_duplicate_entry_point_names_are_refused(monkeypatch: pytest.MonkeyPatch, first, second) -> None:
    points = [
        SimpleNamespace(name="toy", dist=SimpleNamespace(name=name), load=lambda: GENERATOR)
        for name in (first, second)
    ]
    monkeypatch.setattr(generator_module, "entry_points", lambda group: points)
    with pytest.raises(GeneratorError) as raised:
        generators()
    assert all(str(value) in str(raised.value) for value in ("toy", first, second))


@pytest.mark.parametrize("discover", [generators, lambda: source_name(GENERATOR)], ids=["generators", "source_name"])
@pytest.mark.parametrize("error", [ImportError("missing dependency"), RuntimeError("broken initialization")])
def test_entry_point_load_errors_name_the_point_and_keep_the_cause(monkeypatch: pytest.MonkeyPatch, discover, error) -> None:
    def broken():
        raise error

    point = SimpleNamespace(name="broken", value="broken.module:GENERATOR", dist=SimpleNamespace(name="broken-dist"), load=broken)
    monkeypatch.setattr(generator_module, "entry_points", lambda group: [point])
    with pytest.raises(GeneratorError) as raised:
        discover()
    assert point.name in str(raised.value) and point.value in str(raised.value)
    assert raised.value.__cause__ is error


def test_generator_of_reads_the_generator_field(tmp_path: Path) -> None:
    world = GENERATOR.draw("w", 10, 2, np.random.default_rng(1), tmp_path)
    assert generator_of(world.dump()) == "toy"


@pytest.mark.parametrize("config", [b"[]", b"{}", b"not json", b'{"generator": 3}'])
def test_generator_of_refuses_bytes_without_one(config: bytes) -> None:
    with pytest.raises(GeneratorError):
        generator_of(config)


def test_the_toy_generators_source_is_this_package() -> None:
    assert generator_source(GENERATOR) == Path(scienceme_lab.__file__).parent
    assert source_name(GENERATOR) == "scienceme-lab"  # the distribution registering it


def test_toy_interventions_are_the_two_variables() -> None:
    assert GENERATOR.interventions == {"schools_closed", "vaccinate_fraction"}


def _toy(**fields) -> ToyWorld:
    values = dict(id="w", seed=5, weeks=20, jurisdictions=("j0", "j1"), levels=(80.0, 120.0), x_effect=0.3,
                  school_effect=0.4, holidays={"j0": (2,), "j1": (3,)})
    return ToyWorld(ToyConfig(**(values | fields)))


def _spec(world: ToyWorld, role: str = "admissions", **fields) -> StreamSpec:
    values = dict(id=f"w/{role}", world="w", role=role, kind="count", jurisdictions=world.jurisdictions, support_max=10**6)
    return StreamSpec.model_validate(values | fields)


def _closed(base: StreamSpec, end: int | None) -> StreamSpec:
    closure = Intervention(variable="schools_closed", value=1.0, start_week=5, end_week=end)
    return StreamSpec.model_validate(base.model_dump() | {"id": base.id + "~c", "intervention": closure, "parent": base.id})


def _after(world: ToyWorld, spec: StreamSpec, week: int = 5) -> float:
    return sum(r.value for r in world.vintage(spec, 20).rows if r.week >= week)


@pytest.mark.parametrize("end", [9, None])
def test_a_toy_closure_lowers_the_counts_when_school_matters(end: int | None) -> None:
    world = _toy()
    base = _spec(world)
    assert _after(world, _closed(base, end)) < _after(world, base)


def test_a_toy_closure_changes_nothing_when_school_does_not_matter() -> None:
    world = _toy(school_effect=0.0)
    base = _spec(world)
    assert world.vintage(_closed(base, 9), 20).rows == world.vintage(base, 20).rows


def test_the_toy_refuses_what_it_does_not_simulate() -> None:
    world = _toy()
    base = _spec(world)
    unknown = Intervention(variable="masks", value=1.0, start_week=5)
    with pytest.raises(ValueError, match="masks"):
        world.vintage(StreamSpec.model_validate(base.model_dump() | {"id": "w/admissions~m", "intervention": unknown, "parent": base.id}), 10)
    with pytest.raises(ValueError, match="channel"):
        world.vintage(_spec(world, role="deaths"), 10)
    with pytest.raises(ValueError, match="role"):
        world.vintage(_spec(world, role="humidity", kind="covariate", support_max=None), 10)
    with pytest.raises(ValueError, match="not of world"):
        _toy(id="v").vintage(_spec(world), 10)


def test_the_toy_refuses_another_generators_configuration(tmp_path: Path) -> None:
    with pytest.raises(GeneratorError):
        GENERATOR.load(b'{"generator": "seir"}', tmp_path)
