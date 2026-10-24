"""The `lab` command line: the custodian's commands and the evaluation, against temporary roots."""

import typing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from helpers import NOW, at
from scienceme.archive import Archive
from scienceme.mission import load_mission, load_wanted
from scienceme.workspace import Workspace

from scienceme_lab import cli
from scienceme_lab.cli import REPO_ROOT, build_parser, main
from scienceme_lab.commitments import CommitmentLog, SubjectKind
from scienceme_lab.custodian import Custodian
from scienceme_lab.evaluate import EvalConfig, EvalReport, Stamps, criteria
from scienceme_lab.generator import generators
from scienceme_lab.toy import ToyConfig

REPO = Path(__file__).resolve().parent.parent
Run = tuple[int, str, str]
ORIGINAL_NOW = cli._now


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> typing.Any:
    """The CLI's wall clock, fixed at `NOW`; a test moves it by assigning `clock.now`."""

    class Clock:
        now = NOW

    monkeypatch.setattr(cli, "_now", lambda: Clock.now)
    return Clock


@pytest.fixture
def worlds(
    public: Workspace, private: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> typing.Callable[..., tuple[int, str, str]]:
    """Run `main` against the temporary roots and a log in `tmp_path/lab`; returns (exit code, stdout, stderr)."""
    (tmp_path / "lab").mkdir()

    def run(*args: str) -> tuple[int, str, str]:
        code = main([*args, "--public", str(public.root), "--private", str(private), "--log", str(tmp_path / "lab")])
        out, err = capsys.readouterr()
        return code, out, err

    return run


def create_world(worlds: typing.Callable[..., tuple[int, str, str]], world: str = "w1", generator: str = "toy") -> None:
    code, _, err = worlds(
        "create", "--generator", generator, "--world", world, "--channel", "admissions", "--weeks", "24", "--jurisdictions", "3"
    )
    assert (code, err) == (0, "")


def test_public_and_private_may_come_before_or_after_the_subcommand(public: Workspace, private: Path) -> None:
    parser = build_parser()
    tick = ["tick", "--world", "w", "--to-issue", "3"]
    after = parser.parse_args([*tick, "--public", str(public.root), "--private", str(private)])
    before = parser.parse_args(["--public", str(public.root), "--private", str(private), *tick])
    assert (after.public, after.private) == (public.root, private)
    assert (before.public, before.private) == (public.root, private)
    mixed = parser.parse_args(["--public", "a", *tick, "--public", str(public.root)])
    assert mixed.public == public.root
    assert mixed.private == REPO.parent / "science.me-worlds"


def test_the_roots_default_to_the_sibling_repositories_and_this_one() -> None:
    args = build_parser().parse_args(["tick", "--world", "w", "--to-issue", "3"])
    assert (args.log, args.public, args.private) == (REPO, REPO.parent / "science.me", REPO.parent / "science.me-worlds")


def test_create_makes_the_world(worlds: typing.Callable[..., tuple[int, str, str]], public: Workspace, private: Path) -> None:
    code, out, err = worlds(
        "create", "--generator", "toy", "--world", "w1", "--channel", "admissions", "--weeks", "24", "--jurisdictions", "3"
    )
    assert (code, err) == (0, "")
    assert "w1" in out
    assert load_mission(public, "w1").streams == ("w1/admissions", "w1/x", "w1/term")
    assert len(load_wanted(public, "w1")) == 2 * 3 * 4
    assert Archive(public).published_at("w1/term", 0) == NOW
    assert (private / "worlds/w1/config.json").is_file()


def test_create_never_prints_the_truth(worlds: typing.Callable[..., tuple[int, str, str]], private: Path) -> None:
    create_world(worlds)
    cfg = ToyConfig.model_validate_json((private / "worlds/w1/config.json").read_text())
    code, out, err = worlds("tick", "--world", "w1", "--to-issue", "3")
    assert code == 0
    printed = out + err
    for value in cfg.levels:
        assert repr(value) not in printed


def test_the_default_world_has_thirty_weeks_and_four_jurisdictions(
    worlds: typing.Callable[..., tuple[int, str, str]], private: Path
) -> None:
    code, _, _ = worlds("create", "--generator", "toy", "--world", "w1", "--channel", "admissions")
    assert code == 0
    cfg = ToyConfig.model_validate_json((private / "worlds/w1/config.json").read_text())
    assert (cfg.weeks, len(cfg.jurisdictions)) == (30, 4)


def test_create_offers_every_installed_generator() -> None:
    parser = build_parser()
    for generator in generators():
        assert parser.parse_args(["create", "--generator", generator, "--world", "w", "--channel", "c"]).generator == generator
    with pytest.raises(SystemExit):
        parser.parse_args(["create", "--generator", "sir", "--world", "w", "--channel", "c"])


def test_tick_publishes_at_the_clock(
    worlds: typing.Callable[..., tuple[int, str, str]], public: Workspace, clock: typing.Any
) -> None:
    create_world(worlds)
    clock.now = at(5)
    code, out, err = worlds("tick", "--world", "w1", "--to-issue", "4")
    assert (code, err) == (0, "")
    assert "4" in out
    archive = Archive(public)
    assert archive.issues("w1/admissions") == [1, 2, 3, 4]
    assert archive.published_at("w1/x", 4) == at(5)


def test_fulfill_creates_the_arm_at_the_clock(
    worlds: typing.Callable[..., tuple[int, str, str]], public: Workspace, clock: typing.Any
) -> None:
    create_world(worlds)
    worlds("tick", "--world", "w1", "--to-issue", "5")
    clock.now = at(7)
    code, out, err = worlds("fulfill", "--world", "w1", "--item", "close-j1-w8")
    assert (code, err) == (0, "")
    assert "w1/admissions~close-j1-w8" in out
    archive = Archive(public)
    assert archive.spec("w1/admissions~close-j1-w8").fulfills == "close-j1-w8"
    assert archive.published_at("w1/admissions~close-j1-w8", 5) == at(7)


def test_add_channel_creates_the_stream_at_the_clock(
    worlds: typing.Callable[..., tuple[int, str, str]], public: Workspace, clock: typing.Any
) -> None:
    create_world(worlds)
    worlds("tick", "--world", "w1", "--to-issue", "6")
    clock.now = at(9)
    code, out, err = worlds("add-channel", "--world", "w1", "--channel", "ed-visits")
    assert (code, err) == (0, "")
    assert "w1/ed-visits" in out
    archive = Archive(public)
    assert archive.spec("w1/ed-visits").first_issue == 7
    assert archive.published_at("w1/ed-visits", 6) == at(9)


@pytest.mark.parametrize(
    "args",
    [
        ("tick", "--world", "w1", "--to-issue", "99"),
        ("tick", "--world", "nope", "--to-issue", "3"),
        ("fulfill", "--world", "w1", "--item", "no-such-item"),
        ("add-channel", "--world", "w1", "--channel", "school-term"),
        ("create", "--generator", "toy", "--world", "w1", "--channel", "admissions"),
    ],
)
def test_a_refused_operation_is_one_line_on_stderr_and_exit_code_one(
    worlds: typing.Callable[..., tuple[int, str, str]], args: tuple[str, ...]
) -> None:
    create_world(worlds)
    code, out, err = worlds(*args)
    assert code == 1
    assert out == ""
    assert err.startswith("lab: error: ")
    assert len(err.strip().splitlines()) == 1


def test_fulfilling_a_published_week_is_refused(worlds: typing.Callable[..., tuple[int, str, str]], public: Workspace) -> None:
    create_world(worlds)
    worlds("tick", "--world", "w1", "--to-issue", "9")
    code, _, err = worlds("fulfill", "--world", "w1", "--item", "close-j1-w8")
    assert code == 1
    assert "start" in err
    assert not [s for s in Archive(public).specs() if s.fulfills]


def test_a_usage_error_exits_with_two(public: Workspace, private: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tick", "--world", "w1"]) == 2
    assert main(["tick", "--world", "w1", "--to-issue", "3", "--public", str(public.root / "missing")]) == 2
    assert main(["tick", "--world", "w1", "--to-issue", "3", "--public", str(public.root), "--private", str(private / "missing")]) == 2
    assert main([]) == 2
    capsys.readouterr()


def test_help_exits_with_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "create" in capsys.readouterr().out


def test_the_cli_is_a_thin_layer_over_the_custodian(
    worlds: typing.Callable[..., tuple[int, str, str]], public: Workspace, private: Path
) -> None:
    create_world(worlds)
    worlds("tick", "--world", "w1", "--to-issue", "4")
    assert Custodian(public, private, log=None).current_issue("w1") == 4


def test_the_wall_clock_is_an_aware_utc_datetime() -> None:
    now = ORIGINAL_NOW()
    assert now.utcoffset() == timedelta(0)
    assert abs(datetime.now(UTC) - now) < timedelta(seconds=5)


# --- evaluate ---

STAMPS = Stamps(
    programs={"lens-admissions": "sha256:" + "1" * 64},
    scorer="sha256:" + "2" * 64,
    lab="sha256:" + "3" * 64,
    generators={"seir": "sha256:" + "4" * 64, "abm": "sha256:" + "5" * 64},
)


@pytest.fixture
def eval_config(tmp_path: Path) -> tuple[EvalConfig, Path]:
    """A one-replicate evaluation config written to a file."""
    cfg = EvalConfig(programs=("lens-admissions",), public_root=tmp_path, master_seed=3, replicates=1)
    path = tmp_path / "eval.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    return cfg, path


@pytest.fixture
def evaluations(monkeypatch: pytest.MonkeyPatch) -> list[tuple[EvalConfig, Path]]:
    """Replace the evaluation itself (minutes of simulation) with one that records its call and runs nothing."""
    calls: list[tuple[EvalConfig, Path]] = []

    def run(cfg: EvalConfig, scratch: Path) -> EvalReport:
        calls.append((cfg, scratch))
        return EvalReport(config=cfg, stamps=STAMPS, results=(), criteria=criteria(cfg, ()))

    monkeypatch.setattr(cli, "evaluate", run)
    return calls


def test_evaluate_runs_the_config_and_writes_the_report(
    eval_config: tuple[EvalConfig, Path],
    evaluations: list[tuple[EvalConfig, Path]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg, path = eval_config
    out = tmp_path / "report.json"
    code = main(["evaluate", "--config", str(path), "--scratch", str(tmp_path / "scratch"), "--out", str(out)])
    stdout, err = capsys.readouterr()
    assert (code, err) == (0, "")
    assert evaluations == [(cfg, tmp_path / "scratch")]
    report = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report == EvalReport(config=cfg, stamps=STAMPS, results=(), criteria=criteria(cfg, ()))
    assert "failed" in stdout and str(out) in stdout
    assert all(c.name in stdout for c in report.criteria)


def test_evaluate_refuses_to_overwrite_a_report_before_running(
    eval_config: tuple[EvalConfig, Path],
    evaluations: list[tuple[EvalConfig, Path]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, path = eval_config
    out = tmp_path / "report.json"
    out.write_text("{}", encoding="utf-8")
    code = main(["evaluate", "--config", str(path), "--scratch", str(tmp_path / "scratch"), "--out", str(out)])
    _, err = capsys.readouterr()
    assert code == 1
    assert err.startswith("lab: error: ") and len(err.strip().splitlines()) == 1
    assert evaluations == []
    assert out.read_text(encoding="utf-8") == "{}"


def test_evaluate_with_an_invalid_config_is_refused(
    evaluations: list[tuple[EvalConfig, Path]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "eval.json"
    path.write_text('{"programs": [], "master_seed": 1}', encoding="utf-8")
    code = main(["evaluate", "--config", str(path), "--scratch", str(tmp_path), "--out", str(tmp_path / "r.json")])
    _, err = capsys.readouterr()
    assert code == 1
    assert err.startswith("lab: error: ") and len(err.strip().splitlines()) == 1
    assert evaluations == []


def test_evaluate_without_a_config_file_is_a_usage_error(
    evaluations: list[tuple[EvalConfig, Path]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["evaluate", "--config", str(tmp_path / "missing.json"), "--scratch", str(tmp_path), "--out", str(tmp_path / "r")]
    assert main(args) == 2
    assert main(["evaluate", "--scratch", str(tmp_path), "--out", str(tmp_path / "r")]) == 2
    capsys.readouterr()
    assert evaluations == []


def test_evaluate_runs_a_committed_config_on_another_machine(
    eval_config: tuple[EvalConfig, Path],
    evaluations: list[tuple[EvalConfig, Path]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg, path = eval_config
    out, elsewhere = tmp_path / "report.json", tmp_path / "checkout"
    args = ["evaluate", "--config", str(path), "--scratch", str(tmp_path / "s"), "--out", str(out)]
    assert main([*args, "--workers", "2", "--public-root", str(elsewhere)]) == 0
    capsys.readouterr()
    assert evaluations == [(cfg.model_copy(update={"workers": 2, "public_root": elsewhere}), tmp_path / "s")]
    assert path.read_text(encoding="utf-8") == cfg.model_dump_json()  # the committed file is untouched


def test_evaluate_validates_an_overridden_value(
    eval_config: tuple[EvalConfig, Path],
    evaluations: list[tuple[EvalConfig, Path]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, path = eval_config
    args = ["evaluate", "--config", str(path), "--scratch", str(tmp_path / "s"), "--out", str(tmp_path / "r.json")]
    assert main([*args, "--workers", "0"]) == 1
    _, err = capsys.readouterr()
    assert err.startswith("lab: error: ") and len(err.strip().splitlines()) == 1
    assert evaluations == []


# --- sealing from the command line ---


def test_create_seals_the_world_in_the_given_log(worlds: typing.Callable[..., tuple[int, str, str]], tmp_path: Path) -> None:
    create_world(worlds)
    kinds = [(e.subject.kind, e.subject.name) for e in CommitmentLog(tmp_path / "lab").entries()]
    assert kinds == [(SubjectKind.GENERATOR, "scienceme_lab"), (SubjectKind.WORLD, "w1")]


def test_reveal_world_opens_a_closed_world(worlds: typing.Callable[..., tuple[int, str, str]], tmp_path: Path) -> None:
    create_world(worlds)
    code, _, err = worlds("reveal-world", "--world", "w1")
    assert code == 1 and "not closed" in err
    worlds("tick", "--world", "w1", "--to-issue", "24")
    code, out, err = worlds("reveal-world", "--world", "w1")
    assert (code, err) == (0, "") and "revealed/worlds/w1/config.json" in out
    assert CommitmentLog(tmp_path / "lab").problems() == []


def test_a_misregistered_generator_is_one_line_and_exit_code_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scienceme_lab.generator import GeneratorError  # noqa: PLC0415

    def broken() -> dict:
        raise GeneratorError("the entry point 'other' holds the generator 'toy'")

    monkeypatch.setattr(cli, "generators", broken)
    assert main(["check"]) == 1
    assert capsys.readouterr().err.startswith("lab: error: the entry point")


# --- the commitment log ---


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Run `lab --log tmp_path/public ...` and return (exit code, stdout, stderr)."""
    (tmp_path / "public").mkdir()

    def run(*args: str) -> Run:
        code = main(["--log", str(tmp_path / "public"), *args])
        out, err = capsys.readouterr()
        return code, out, err

    return run


def one_line(text: str) -> bool:
    return len(text.strip().splitlines()) == 1


def test_commit_reveal_and_check_round_trip(lab, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"r0": 1.5}')
    keep = tmp_path / "nonce"
    code, out, err = lab("commit", "--kind", "world", "--name", "w-alpha", "--content", str(config), "--keep", str(keep))
    assert (code, err) == (0, "") and one_line(out) and str(keep) in out
    (commitment,) = CommitmentLog(tmp_path / "public").entries()
    published = tmp_path / "public" / "w-alpha.json"
    published.write_bytes(config.read_bytes())
    code, out, err = lab("reveal", "--seal", commitment.seal, "--content", str(published), "--nonce", str(keep))
    assert (code, err) == (0, "") and "w-alpha.json" in out
    assert lab("check") == (0, "ok\n", "")
    published.write_text("{}")
    code, out, _ = lab("check")
    assert code == 1 and "does not open" in out


def test_a_refused_reveal_is_one_line_on_stderr(lab, tmp_path: Path) -> None:
    content = tmp_path / "public" / "x.json"
    content.write_text("{}")
    nonce = tmp_path / "nonce"
    nonce.write_text("00" * 32)
    code, out, err = lab("reveal", "--seal", "sha256:" + "0" * 64, "--content", str(content), "--nonce", str(nonce))
    assert (code, out) == (1, "") and err.startswith("lab: error: no commitment") and one_line(err)


def test_a_nonce_file_that_is_not_hex_is_refused(lab, tmp_path: Path) -> None:
    content = tmp_path / "public" / "x.json"
    content.write_text("{}")
    nonce = tmp_path / "nonce"
    nonce.write_text("not hex")
    code, _, err = lab("reveal", "--seal", "sha256:" + "0" * 64, "--content", str(content), "--nonce", str(nonce))
    assert code == 1 and err.startswith("lab: error: ") and one_line(err)


def test_committing_over_a_kept_nonce_is_refused(lab, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    keep = tmp_path / "nonce"
    keep.write_text("kept")
    code, _, err = lab("commit", "--kind", "world", "--name", "w", "--content", str(config), "--keep", str(keep))
    assert code == 1 and "refusing to overwrite" in err and one_line(err)


@pytest.mark.parametrize(
    "args",
    [
        ("commit", "--kind", "planet", "--name", "w", "--content", ".", "--keep", "k"),
        ("commit", "--kind", "world", "--name", "w", "--content", "/no/such/file", "--keep", "k"),
        ("reveal", "--seal", "s"),
        (),
    ],
)
def test_usage_errors_exit_2(lab, args: tuple[str, ...]) -> None:
    assert lab(*args)[0] == 2


def test_the_default_log_is_this_repository() -> None:
    assert build_parser().parse_args(["check"]).log == REPO_ROOT
    assert (REPO_ROOT / "pyproject.toml").is_file()
