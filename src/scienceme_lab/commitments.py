"""Commitments: how a lab binds itself to private content now and lets anyone check it later.

A commitment publishes a seal, `sha256(nonce || content)`, and keeps the nonce and the content private.
The random nonce makes the seal hiding even when the content has little entropy, and sha256 makes it
binding: no other content opens it.  A reveal publishes the content and the nonce, so anyone can recompute
the seal.  The log (`COMMITMENTS_FILE`) is append-only JSONL of `Commitment` and `Reveal` entries;
`CommitmentLog.problems` checks that every reveal opens an earlier commitment.

What a subject's content is depends on its kind (`SubjectKind.content`): a world's drawn configuration is
its file's bytes, and a generator's source is its `package_hash`, which covers every module.  Nothing here
reads a clock: every write takes `now`.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter
from scienceme.model import Frozen
from scienceme.worker import package_hash
from scienceme.workspace import WriteOnceError, append_jsonl, read_jsonl

COMMITMENTS_FILE = "commitments.jsonl"
NONCE_BYTES = 32
_SEAL_PREFIX = "sha256:"


class SubjectKind(StrEnum):
    """What kind of private thing a commitment is to."""

    WORLD = "world"  # a world's drawn configuration: a file
    GENERATOR = "generator"  # a generator's source: a package directory

    def content(self, path: Path) -> bytes:
        """The bytes a seal of this kind covers, read from `path`: a file for a world, a directory for a generator.

        CommitmentError when `path` is not that, since a generator's hash over a file would be the empty hash.
        """
        if self is SubjectKind.WORLD:
            if not path.is_file():
                raise CommitmentError(f"a world's content is a file, and {path} is not one")
            return path.read_bytes()
        if not path.is_dir():
            raise CommitmentError(f"a generator's content is a package directory, and {path} is not one")
        return package_hash(path).encode("utf-8")


class Subject(Frozen):
    """One private thing, by kind and name (a world id, a generator package)."""

    kind: SubjectKind
    name: str = Field(min_length=1)


class Commitment(Frozen):
    """`subject` was sealed as `seal` at `committed_at`; its content and nonce stay private until revealed."""

    entry: Literal["commitment"] = "commitment"
    subject: Subject
    seal: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    committed_at: AwareDatetime


class Reveal(Frozen):
    """The commitment `seal` is opened by `nonce` (hex) over the content at `path`, relative to the log's directory."""

    entry: Literal["reveal"] = "reveal"
    seal: str
    nonce: str = Field(pattern=r"^(?:[0-9a-f]{2})+$")
    path: str
    revealed_at: AwareDatetime


class CommitmentError(Exception):
    """Raised when a commitment or a reveal is refused."""


_ENTRIES = TypeAdapter(Annotated[Commitment | Reveal, Field(discriminator="entry")])


def seal(nonce: bytes, content: bytes) -> str:
    """`sha256:` plus the hex sha256 of `nonce` followed by `content`."""
    return _SEAL_PREFIX + hashlib.sha256(nonce + content).hexdigest()


class CommitmentLog:
    """Reads and appends the commitment log in directory `root`, where revealed content is published too."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def path(self) -> Path:
        return self._root / COMMITMENTS_FILE

    def entries(self) -> list[Commitment | Reveal]:
        """Every entry, in the order it was appended."""
        return read_jsonl(self.path, _ENTRIES)

    def commit(self, subject: Subject, content: Path, keep: Path, now: datetime) -> Commitment:
        """Seal the content at `content` under a fresh nonce, kept privately at `keep`, and append the commitment.

        The nonce is written (hex, write-once) before the commitment is appended, so no commitment exists
        that its owner cannot open.  WriteOnceError when `keep` exists; nothing is appended then.
        """
        nonce = secrets.token_bytes(NONCE_BYTES)
        commitment = Commitment(subject=subject, seal=seal(nonce, subject.kind.content(content)), committed_at=now)
        keep.parent.mkdir(parents=True, exist_ok=True)
        try:
            with keep.open("x", encoding="utf-8") as f:
                f.write(nonce.hex() + "\n")
        except FileExistsError as e:
            raise WriteOnceError(f"refusing to overwrite the nonce at {keep}") from e
        append_jsonl(self.path, commitment)
        return commitment

    def adopt(self, subject: Subject, content: Path, keep: Path, now: datetime) -> Commitment:
        """The commitment to `content` under the nonce already kept at `keep`, appended unless this log has it.

        For a nonce whose commitment never reached this log: written before an append that failed, or kept
        for another log.
        """
        nonce = bytes.fromhex(keep.read_text(encoding="utf-8").strip())
        sealed = seal(nonce, subject.kind.content(content))
        found = self._commitments().get(sealed)
        if found is not None:
            return found
        commitment = Commitment(subject=subject, seal=sealed, committed_at=now)
        append_jsonl(self.path, commitment)
        return commitment

    def reveal(self, sealed: str, nonce: bytes, content: Path, now: datetime) -> Reveal:
        """Open the commitment `sealed` with `nonce` over `content`, which must lie inside the log's directory.

        CommitmentError, appending nothing, when no commitment has that seal, it is already revealed, the
        content is outside the log's directory, or nonce and content do not reproduce the seal.
        """
        commitment = self._commitments().get(sealed)
        if commitment is None:
            raise CommitmentError(f"no commitment has the seal {sealed}")
        if any(isinstance(e, Reveal) and e.seal == sealed for e in self.entries()):
            raise CommitmentError(f"the commitment {sealed} is already revealed")
        root = self._root.resolve()
        if not content.resolve().is_relative_to(root):
            raise CommitmentError(f"{content} is not inside {self._root}, where reveals are published")
        if seal(nonce, commitment.subject.kind.content(content)) != sealed:
            raise CommitmentError(f"{content} under that nonce does not open the commitment {sealed}")
        where = content.resolve().relative_to(root).as_posix()
        reveal = Reveal(seal=sealed, nonce=nonce.hex(), path=where, revealed_at=now)
        append_jsonl(self.path, reveal)
        return reveal

    def problems(self) -> list[str]:
        """One line per thing wrong with the log; empty when every reveal opens an earlier commitment, once."""
        found: list[str] = []
        committed: dict[str, Commitment] = {}
        revealed: set[str] = set()
        for n, e in enumerate(self.entries(), start=1):
            if isinstance(e, Commitment):
                if e.seal in committed:
                    found.append(f"line {n}: the seal {e.seal} is committed twice")
                committed[e.seal] = e
                continue
            commitment = committed.get(e.seal)
            if commitment is None:
                found.append(f"line {n}: reveals {e.seal}, which no earlier line commits")
                continue
            if e.seal in revealed:
                found.append(f"line {n}: reveals {e.seal} again")
            revealed.add(e.seal)
            if e.revealed_at < commitment.committed_at:
                found.append(f"line {n}: reveals {e.seal} before it was committed")
            path = self._root / e.path
            if not path.exists():
                found.append(f"line {n}: the revealed {e.path} is missing")
            elif seal(bytes.fromhex(e.nonce), commitment.subject.kind.content(path)) != e.seal:
                found.append(f"line {n}: {e.path} does not open {e.seal}")
        return found

    def _commitments(self) -> dict[str, Commitment]:
        return {e.seal: e for e in self.entries() if isinstance(e, Commitment)}
