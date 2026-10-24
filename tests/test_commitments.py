import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from scienceme.worker import package_hash
from scienceme.workspace import WriteOnceError, append_jsonl

from scienceme_lab.commitments import (
    COMMITMENTS_FILE,
    NONCE_BYTES,
    Commitment,
    CommitmentError,
    CommitmentLog,
    Reveal,
    Subject,
    SubjectKind,
    seal,
)

T0 = datetime(2026, 10, 23, tzinfo=UTC)
WORLD = Subject(kind=SubjectKind.WORLD, name="w-alpha")
GENERATOR = Subject(kind=SubjectKind.GENERATOR, name="worlds")


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def nonce_at(keep: Path) -> bytes:
    return bytes.fromhex(keep.read_text())


@pytest.fixture
def log(tmp_path: Path) -> CommitmentLog:
    """A log in a public directory `tmp_path/public`."""
    (tmp_path / "public").mkdir()
    return CommitmentLog(tmp_path / "public")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """A private world configuration."""
    path = tmp_path / "private" / "w-alpha" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"r0": 1.5}')
    return path


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """A private generator package."""
    root = tmp_path / "private" / "gen"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sim.py").write_text("RATE = 0.3\n")
    return root


def publish(log: CommitmentLog, source: Path, name: str) -> Path:
    """Copy `source` (a file or a package) into the log's directory, as a reveal publishes it."""
    target = log.path.parent / "revealed" / name
    if source.is_dir():
        target.mkdir(parents=True)
        for f in source.iterdir():
            (target / f.name).write_bytes(f.read_bytes())
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return target


# --- the seal ---


def test_a_seal_is_the_sha256_of_the_nonce_then_the_content() -> None:
    assert seal(b"n", b"content") == "sha256:" + hashlib.sha256(b"ncontent").hexdigest()
    assert seal(b"m", b"content") != seal(b"n", b"content")


def test_a_generator_seal_covers_its_package_hash(package: Path) -> None:
    assert SubjectKind.GENERATOR.content(package) == package_hash(package).encode()


@pytest.mark.parametrize("kind", list(SubjectKind))
def test_content_of_the_wrong_kind_of_path_is_refused(kind: SubjectKind, config: Path, package: Path) -> None:
    wrong = package if kind is SubjectKind.WORLD else config
    with pytest.raises(CommitmentError, match="is not one"):
        kind.content(wrong)


# --- committing ---


def test_commit_keeps_the_nonce_privately_and_appends_the_seal(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    keep = tmp_path / "private" / "w-alpha" / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    nonce = nonce_at(keep)
    assert len(nonce) == NONCE_BYTES
    assert commitment == Commitment(subject=WORLD, seal=seal(nonce, config.read_bytes()), committed_at=at(1))
    assert log.entries() == [commitment]
    assert config.read_text() not in log.path.read_text() and nonce.hex() not in log.path.read_text()


def test_each_commitment_draws_its_own_nonce(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    first = log.commit(WORLD, config, tmp_path / "n1", at(1))
    second = log.commit(WORLD, config, tmp_path / "n2", at(2))
    assert first.seal != second.seal


def test_commit_refuses_to_overwrite_a_kept_nonce_and_appends_nothing(
    log: CommitmentLog, config: Path, tmp_path: Path
) -> None:
    keep = tmp_path / "nonce"
    keep.write_text("kept")
    with pytest.raises(WriteOnceError):
        log.commit(WORLD, config, keep, at(1))
    assert keep.read_text() == "kept" and log.entries() == []


# --- revealing ---


def test_a_reveal_of_a_world_opens_its_commitment(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    published = publish(log, config, "w-alpha.json")
    reveal = log.reveal(commitment.seal, nonce_at(keep), published, at(5))
    assert reveal == Reveal(seal=commitment.seal, nonce=nonce_at(keep).hex(), path="revealed/w-alpha.json", revealed_at=at(5))
    assert log.entries() == [commitment, reveal] and log.problems() == []


def test_a_reveal_of_a_generator_opens_its_commitment(log: CommitmentLog, package: Path, tmp_path: Path) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(GENERATOR, package, keep, at(1))
    log.reveal(commitment.seal, nonce_at(keep), publish(log, package, "gen"), at(5))
    assert log.problems() == []


def test_a_reveal_is_refused_unless_nonce_and_content_reproduce_the_seal(
    log: CommitmentLog, config: Path, tmp_path: Path
) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    published = publish(log, config, "w-alpha.json")
    with pytest.raises(CommitmentError, match="does not open"):
        log.reveal(commitment.seal, b"\x00" * NONCE_BYTES, published, at(5))
    published.write_text('{"r0": 1.6}')  # a quietly adjusted world
    with pytest.raises(CommitmentError, match="does not open"):
        log.reveal(commitment.seal, nonce_at(keep), published, at(5))
    assert log.entries() == [commitment]


def test_a_reveal_is_refused_for_an_unknown_seal_or_a_second_time(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    published = publish(log, config, "w-alpha.json")
    with pytest.raises(CommitmentError, match="no commitment"):
        log.reveal("sha256:" + "0" * 64, nonce_at(keep), published, at(5))
    log.reveal(commitment.seal, nonce_at(keep), published, at(5))
    with pytest.raises(CommitmentError, match="already revealed"):
        log.reveal(commitment.seal, nonce_at(keep), published, at(6))


def test_a_reveal_must_publish_its_content_inside_the_log_directory(
    log: CommitmentLog, config: Path, tmp_path: Path
) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    with pytest.raises(CommitmentError, match="not inside"):
        log.reveal(commitment.seal, nonce_at(keep), config, at(5))  # still private
    assert log.entries() == [commitment]


# --- checking ---


def test_check_finds_a_revealed_file_changed_afterwards(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    published = publish(log, config, "w-alpha.json")
    log.reveal(commitment.seal, nonce_at(keep), published, at(5))
    published.write_text('{"r0": 1.6}')
    assert log.problems() == [f"line 2: revealed/w-alpha.json does not open {commitment.seal}"]
    published.unlink()
    assert log.problems() == ["line 2: the revealed revealed/w-alpha.json is missing"]


def test_check_finds_reveals_appended_by_hand(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    keep = tmp_path / "nonce"
    commitment = log.commit(WORLD, config, keep, at(1))
    published = publish(log, config, "w-alpha.json")
    opened = Reveal(seal=commitment.seal, nonce=nonce_at(keep).hex(), path="revealed/w-alpha.json", revealed_at=at(5))
    early = opened.model_copy(update={"revealed_at": at(0)})
    stray = opened.model_copy(update={"seal": "sha256:" + "1" * 64})
    for entry in (early, opened, stray, commitment):
        append_jsonl(log.path, entry)
    assert log.problems() == [
        f"line 2: reveals {commitment.seal} before it was committed",
        f"line 3: reveals {commitment.seal} again",
        f"line 4: reveals {stray.seal}, which no earlier line commits",
        f"line 5: the seal {commitment.seal} is committed twice",
    ]


def test_an_empty_or_absent_log_has_no_problems(log: CommitmentLog) -> None:
    assert log.entries() == [] and log.problems() == []


@pytest.mark.parametrize(
    "entry",
    [
        {"entry": "commitment", "subject": {"kind": "world", "name": "w"}, "seal": "sha256:abc", "committed_at": T0.isoformat()},
        {"entry": "commitment", "subject": {"kind": "planet", "name": "w"}, "seal": "sha256:" + "0" * 64, "committed_at": T0.isoformat()},
        {"entry": "reveal", "seal": "s", "nonce": "abc", "path": "p", "revealed_at": T0.isoformat()},
    ],
)
def test_malformed_lines_are_refused_on_read(log: CommitmentLog, entry: dict[str, object]) -> None:
    (log.path.parent / COMMITMENTS_FILE).write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValidationError):
        log.entries()


def test_adopt_appends_a_commitment_under_a_kept_nonce_once(log: CommitmentLog, config: Path, tmp_path: Path) -> None:
    keep = tmp_path / "nonce"
    first = log.commit(WORLD, config, keep, at(1))
    other = CommitmentLog(tmp_path / "other")
    adopted = other.adopt(WORLD, config, keep, at(2))
    assert adopted.seal == first.seal and other.entries() == [adopted]
    assert other.adopt(WORLD, config, keep, at(3)) == adopted  # already in this log: nothing appended
    assert other.entries() == [adopted]
