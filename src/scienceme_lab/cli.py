"""The `lab` command line: the custodian's commands, the tier 1 evaluation, and the commitment log.

The custodian commands (`create`, `tick`, `fulfill`, `add-channel`, `reveal-world`) work on a public
workspace (`--public`, default the sibling `science.me`), a private root (`--private`, default the sibling
`science.me-worlds`) and the commitment log in `--log` (default this repository), and seal into that log
as they go.  `commit`, `reveal` and `check` work on the log alone; `evaluate` takes everything from its
config file.  The root options may come before or after the subcommand.  Generators are the installed
ones (`scienceme_lab.generator.generators`).  Exit codes: 0 for success, 1 for a refused operation or a
failed check (one line on stderr, never a traceback) and 2 for a usage error.

This is the only module that reads the wall clock (`_now`): everything below it takes `now`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from scienceme.archive import ArchiveError
from scienceme.mission import MissionError
from scienceme.workspace import Workspace, WriteOnceError, read_model, write_new

from scienceme_lab.commitments import CommitmentError, CommitmentLog, Subject, SubjectKind
from scienceme_lab.custodian import Custodian, CustodianError
from scienceme_lab.evaluate import EvalConfig, EvaluationError, evaluate, machine_fields
from scienceme_lab.generator import GeneratorError, generators

Handler = Callable[[argparse.Namespace], int]
CustodianHandler = Callable[[Custodian, argparse.Namespace], int]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT
DEFAULT_PUBLIC = REPO_ROOT.parent / "science.me"
DEFAULT_PRIVATE = REPO_ROOT.parent / "science.me-worlds"
DEFAULT_WEEKS = 30
DEFAULT_JURISDICTIONS = 4

# What a command reports as a one-line error.  ValueError covers a nonce file that is not hex.
_DOMAIN_ERRORS = (
    CommitmentError,
    CustodianError,
    EvaluationError,
    GeneratorError,
    ArchiveError,
    MissionError,
    WriteOnceError,
    ValidationError,
    ValueError,
)
# The root directories, and which commands need each to exist.
_ROOTS = {"log": DEFAULT_LOG, "public": DEFAULT_PUBLIC, "private": DEFAULT_PRIVATE}
_ROOT_HELP = {
    "log": "the directory of the commitment log, where reveals are published",
    "public": "the public science.me workspace",
    "private": "the private root holding worlds/ and commitments/",
}


def _now() -> datetime:
    """The wall clock, as an aware UTC datetime: the one place the package reads it."""
    return datetime.now(UTC)


def _fail(error: Exception) -> int:
    """Print `error` on one line on stderr and return the refused-operation exit code."""
    print(f"lab: error: {' '.join(str(error).split())}", file=sys.stderr)
    return 1


# --- the custodian's commands ---


def _create(custodian: Custodian, args: argparse.Namespace) -> int:
    custodian.create(args.generator, args.world, args.weeks, args.jurisdictions, args.channel, _now())
    print(f"created world {args.world}: mission {args.world}, stream {args.world}/{args.channel}")
    return 0


def _tick(custodian: Custodian, args: argparse.Namespace) -> int:
    captures = custodian.tick(args.world, args.to_issue, _now())
    print(f"published {len(captures)} captures; {args.world} is at issue {custodian.current_issue(args.world)}")
    return 0


def _fulfill(custodian: Custodian, args: argparse.Namespace) -> int:
    arm = custodian.fulfill(args.world, args.item, _now())
    print(f"fulfilled {args.item} with stream {arm.id} (first issue {arm.first_issue})")
    return 0


def _add_channel(custodian: Custodian, args: argparse.Namespace) -> int:
    spec = custodian.add_channel(args.world, args.channel, _now())
    print(f"added stream {spec.id} (first issue {spec.first_issue})")
    return 0


def _reveal_world(custodian: Custodian, args: argparse.Namespace) -> int:
    reveal = custodian.reveal(args.world, _now())
    print(f"revealed world {args.world} as {reveal.path}")
    return 0


def _with_custodian(run: CustodianHandler) -> Handler:
    """A handler that runs `run` on the custodian of `--public` and `--private`, sealing into `--log`."""
    return lambda args: run(Custodian(Workspace(args.public), args.private, CommitmentLog(args.log)), args)


# --- the log's commands ---


def _commit(args: argparse.Namespace) -> int:
    commitment = CommitmentLog(args.log).commit(Subject(kind=args.kind, name=args.name), args.content, args.keep, _now())
    print(f"committed {commitment.subject.kind} {commitment.subject.name} as {commitment.seal}; nonce kept at {args.keep}")
    return 0


def _reveal(args: argparse.Namespace) -> int:
    nonce = bytes.fromhex(args.nonce.read_text(encoding="utf-8").strip())
    reveal = CommitmentLog(args.log).reveal(args.seal, nonce, args.content, _now())
    print(f"revealed {reveal.seal} as {reveal.path}")
    return 0


def _check(args: argparse.Namespace) -> int:
    problems = CommitmentLog(args.log).problems()
    print("\n".join(problems) if problems else "ok")
    return 1 if problems else 0


# --- the evaluation ---


def _evaluate(args: argparse.Namespace) -> int:
    """Run the evaluation `--config` describes in runs under `--scratch` and write its report to `--out`.

    Each `machine_fields` option given replaces the config's value, so a pre-registered config runs as
    committed on any machine.
    """
    if args.out.exists():  # refused before the evaluation, not after it
        raise WriteOnceError(f"refusing to overwrite {args.out}")
    cfg = read_model(args.config, EvalConfig)
    here = {name: value for name in machine_fields() if (value := getattr(args, name)) is not None}
    report = evaluate(EvalConfig.model_validate({**cfg.model_dump(), **here}), args.scratch)
    write_new(args.out, report)
    for c in report.criteria:
        note = "" if c.judged else " (reported, not judged)"
        print(f"{c.name} ({c.generator or 'pooled'}): {'passed' if c.passed else 'failed'}{note}")
    outcome = "passed" if report.passed else "failed"
    print(f"tier 1 {outcome} over {len(report.results)} runs; report written to {args.out}")
    return 0


# --- parsing ---


def _existing(value: str) -> Path:
    """`value` as a path, for argparse; a usage error unless something is there."""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return path


def _existing_file(value: str) -> Path:
    """`value` as a path, for argparse; a usage error unless it names a file."""
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{value} is not a file")
    return path


def _add_roots(parser: argparse.ArgumentParser, defaults: dict[str, object]) -> None:
    """Add `--log`, `--public` and `--private`; an option left out gives its value in `defaults`."""
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=default, metavar="DIR", help=f"{_ROOT_HELP[name]} (default: {_ROOTS[name]})")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser: one subcommand per command."""
    parser = argparse.ArgumentParser(prog="lab", description=__doc__.splitlines()[0])
    _add_roots(parser, dict(_ROOTS))
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    suppressed = dict.fromkeys(_ROOTS, argparse.SUPPRESS)  # an option given after the subcommand wins

    def command(name: str, handler: Handler, summary: str, needs: tuple[str, ...]) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.set_defaults(handler=handler, needs=needs)
        _add_roots(sub, suppressed)
        return sub

    def custodian(name: str, handler: CustodianHandler, summary: str) -> argparse.ArgumentParser:
        sub = command(name, _with_custodian(handler), summary, ("log", "public", "private"))
        sub.add_argument("--world", required=True, metavar="ID")
        return sub

    create = custodian("create", _create, "draw a world (its truth stays private), seal it, and publish its streams and mission")
    create.add_argument("--generator", required=True, choices=sorted(generators()))
    create.add_argument("--channel", required=True, metavar="ROLE", help="the base channel, e.g. admissions")
    create.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, metavar="N", help=f"default: {DEFAULT_WEEKS}")
    create.add_argument(
        "--jurisdictions", type=int, default=DEFAULT_JURISDICTIONS, metavar="N", help=f"default: {DEFAULT_JURISDICTIONS}"
    )
    tick = custodian("tick", _tick, "publish every stream's issues up to --to-issue, stamped now")
    tick.add_argument("--to-issue", type=int, required=True, metavar="N")
    fulfill = custodian("fulfill", _fulfill, "run a wanted item's arm and publish it, stamped now")
    fulfill.add_argument("--item", required=True, metavar="ID")
    add_channel = custodian("add-channel", _add_channel, "start publishing the world through another channel, stamped now")
    add_channel.add_argument("--channel", required=True, metavar="ROLE", help="the new channel, e.g. ed-visits")
    custodian("reveal-world", _reveal_world, "reveal a closed world's configuration in the log's directory")

    commit = command("commit", _commit, "seal private content and append the commitment; the nonce stays private", ("log",))
    commit.add_argument("--kind", required=True, type=SubjectKind, choices=list(SubjectKind), help="what the content is")
    commit.add_argument("--name", required=True, help="the world id or generator package the content is")
    commit.add_argument("--content", required=True, type=_existing, metavar="PATH", help="a world's file or a generator's package directory")
    commit.add_argument("--keep", required=True, type=Path, metavar="PATH", help="where to keep the nonce privately (must not exist)")
    reveal = command("reveal", _reveal, "open a commitment: its content, published inside the log's directory, and its nonce", ("log",))
    reveal.add_argument("--seal", required=True, help="the commitment's seal, sha256:<hex>")
    reveal.add_argument("--content", required=True, type=_existing, metavar="PATH", help="the content, inside the log's directory")
    reveal.add_argument("--nonce", required=True, type=_existing, metavar="PATH", help="the file the nonce was kept in")
    command("check", _check, "check that every reveal opens an earlier commitment, once", ("log",))

    evaluation = command("evaluate", _evaluate, "run the tier 1 evaluation a config file describes and write its report", ())
    evaluation.add_argument("--config", required=True, type=_existing_file, metavar="PATH", help="an EvalConfig JSON file")
    evaluation.add_argument("--scratch", required=True, type=Path, metavar="DIR", help="where each run gets its own directory")
    evaluation.add_argument("--out", required=True, type=Path, metavar="PATH", help="the report JSON to write (must not exist)")
    for name in machine_fields():  # the config's own values, unless this machine needs others
        field = EvalConfig.model_fields[name]
        evaluation.add_argument(
            f"--{name.replace('_', '-')}", dest=name, type=field.annotation, metavar=name.upper(),
            help=f"{field.description}, instead of the config's",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line `argv` (default `sys.argv[1:]`) and return its exit code."""
    try:
        parser = build_parser()
    except GeneratorError as e:  # a misregistered generator: nothing can run
        return _fail(e)
    try:
        args = parser.parse_args(argv)
        for root in args.needs:
            if not getattr(args, root).is_dir():
                parser.error(f"--{root} {getattr(args, root)} is not a directory")
    except SystemExit as e:  # argparse exits for a usage error (2) and for --help (0)
        return int(e.code or 0)
    try:
        return args.handler(args)
    except _DOMAIN_ERRORS as e:
        return _fail(e)
