"""The `lab` command line: commit to private content, reveal it, and check the commitment log.

Every command works on the log in `--log` (default this repository's root, where reveals are published).
Exit codes: 0 for success, 1 for a refused operation or a failed check (one line on stderr, never a
traceback) and 2 for a usage error.

This is the only module that reads the wall clock (`_now`): everything below it takes `now`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from scienceme.workspace import WriteOnceError

from scienceme_lab.commitments import CommitmentError, CommitmentLog, Subject, SubjectKind

Handler = Callable[[CommitmentLog, argparse.Namespace], int]

REPO_ROOT = Path(__file__).resolve().parents[2]

# What a command reports as a one-line error.  ValueError covers a nonce file that is not hex.
_DOMAIN_ERRORS = (CommitmentError, WriteOnceError, ValueError)


def _now() -> datetime:
    """The wall clock, as an aware UTC datetime: the one place the package reads it."""
    return datetime.now(UTC)


def _fail(error: Exception) -> int:
    """Print `error` on one line on stderr and return the refused-operation exit code."""
    print(f"lab: error: {' '.join(str(error).split())}", file=sys.stderr)
    return 1


# --- commands ---


def _commit(log: CommitmentLog, args: argparse.Namespace) -> int:
    commitment = log.commit(Subject(kind=args.kind, name=args.name), args.content, args.keep, _now())
    print(f"committed {commitment.subject.kind} {commitment.subject.name} as {commitment.seal}; nonce kept at {args.keep}")
    return 0


def _reveal(log: CommitmentLog, args: argparse.Namespace) -> int:
    nonce = bytes.fromhex(args.nonce.read_text(encoding="utf-8").strip())
    reveal = log.reveal(args.seal, nonce, args.content, _now())
    print(f"revealed {reveal.seal} as {reveal.path}")
    return 0


def _check(log: CommitmentLog, args: argparse.Namespace) -> int:
    problems = log.problems()
    print("\n".join(problems) if problems else "ok")
    return 1 if problems else 0


# --- parsing ---


def _existing(value: str) -> Path:
    """`value` as a path, for argparse; a usage error unless something is there."""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log", type=Path, default=REPO_ROOT, metavar="DIR", help="the directory of the commitment log (default: this repository)"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, handler: Handler, summary: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.set_defaults(handler=handler)
        return sub

    commit = command("commit", _commit, "seal private content and append the commitment; the nonce stays private")
    commit.add_argument("--kind", required=True, type=SubjectKind, choices=list(SubjectKind), help="what the content is")
    commit.add_argument("--name", required=True, help="the world id or generator package the content is")
    commit.add_argument("--content", required=True, type=_existing, metavar="PATH", help="a world's file or a generator's package directory")
    commit.add_argument("--keep", required=True, type=Path, metavar="PATH", help="where to keep the nonce privately (must not exist)")

    reveal = command("reveal", _reveal, "open a commitment: its content, published inside the log's directory, and its nonce")
    reveal.add_argument("--seal", required=True, help="the commitment's seal, sha256:<hex>")
    reveal.add_argument("--content", required=True, type=_existing, metavar="PATH", help="the content, inside the log's directory")
    reveal.add_argument("--nonce", required=True, type=_existing, metavar="PATH", help="the file the nonce was kept in")

    command("check", _check, "check that every reveal opens an earlier commitment, once")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line `argv` (default `sys.argv[1:]`) and return its exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # argparse exits for a usage error (2) and for --help (0)
        return int(e.code or 0)
    try:
        return args.handler(CommitmentLog(args.log), args)
    except _DOMAIN_ERRORS as e:
        return _fail(e)
