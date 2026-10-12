"""bump_version.py bumps version.py and pyproject.toml and commits, but leaves tagging to the Tag release workflow."""

import subprocess
import sys

import pytest

import bump_version

VERSION_PY = '"""Version information for TempHist API."""\n\n__version__ = "1.5.2"\n__version_info__ = (1, 5, 2)\n'
PYPROJECT = '[project]\nname = "temphist-api"\nversion = "1.5.2"\n'


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def run_script(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["bump_version.py", *args])
    bump_version.main()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo on a feature branch with the version files at 1.5.2."""
    monkeypatch.chdir(tmp_path)
    git("init", "-q", "-b", "chore/bump-version")
    for key, value in [
        ("user.email", "test@example.com"),
        ("user.name", "Test"),
        ("commit.gpgsign", "false"),
        ("core.hooksPath", "/dev/null"),
    ]:
        git("config", key, value)
    (tmp_path / "version.py").write_text(VERSION_PY)
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    git("add", ".")
    git("commit", "-qm", "init")
    return tmp_path


@pytest.mark.parametrize(
    "bump_type, expected",
    [("patch", "1.5.3"), ("minor", "1.6.0"), ("major", "2.0.0")],
)
def test_bump_version_arithmetic(bump_type, expected):
    assert bump_version.bump_version("1.5.2", bump_type) == expected


def test_bump_version_rejects_unknown_bump_type():
    with pytest.raises(SystemExit):
        bump_version.bump_version("1.5.2", "bogus")


def test_bump_version_rejects_malformed_version():
    with pytest.raises(SystemExit):
        bump_version.bump_version("1.5", "patch")


def test_get_current_version_reads_version_py(repo):
    assert bump_version.get_current_version() == "1.5.2"


def test_get_current_version_exits_when_version_py_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        bump_version.get_current_version()


def test_get_current_version_exits_when_version_unparseable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "version.py").write_text("answer = 42\n")
    with pytest.raises(SystemExit):
        bump_version.get_current_version()


def test_update_version_file_exits_when_version_py_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        bump_version.update_version_file("1.5.3")


def test_update_pyproject_toml_skips_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bump_version.update_pyproject_toml("1.5.3")
    assert "pyproject.toml not found" in capsys.readouterr().out


def test_bump_updates_both_files_and_commits(repo, monkeypatch):
    run_script(monkeypatch, "minor")

    version_py = (repo / "version.py").read_text()
    assert '__version__ = "1.6.0"' in version_py
    assert "__version_info__ = (1, 6, 0)" in version_py
    assert 'version = "1.6.0"' in (repo / "pyproject.toml").read_text()
    assert git("log", "-1", "--format=%s") == "Bump version to 1.6.0"
    assert git("status", "--porcelain") == ""


def test_bump_does_not_create_a_tag(repo, monkeypatch):
    run_script(monkeypatch, "patch")

    assert git("tag") == ""


@pytest.mark.parametrize("branch", ["main", "staging", "develop"])
def test_refuses_to_run_on_protected_branch(repo, monkeypatch, branch):
    git("switch", "-qc", branch)

    with pytest.raises(SystemExit) as exc:
        run_script(monkeypatch, "patch")

    assert exc.value.code == 1
    assert '__version__ = "1.5.2"' in (repo / "version.py").read_text()
    assert git("log", "-1", "--format=%s") == "init"


def test_wrong_argument_count_prints_usage(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_script(monkeypatch)

    assert exc.value.code == 1
    assert "Usage: python bump_version.py" in capsys.readouterr().out


def test_git_commit_exits_on_git_error(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(bump_version.subprocess, "run", fail)

    with pytest.raises(SystemExit) as exc:
        bump_version.git_commit("1.5.3")

    assert exc.value.code == 1
