from pathlib import Path

import pytest

from scienceme_lab.cli import REPO_ROOT, build_parser, main
from scienceme_lab.commitments import CommitmentLog

Run = tuple[int, str, str]


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
