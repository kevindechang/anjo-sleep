# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The example runs from the published tree alone (Task 15), in Task 1's filtered fixture.

The fixture's trunk gets a copy of ``anjo/sleep``; ``git filter-repo`` cuts it the way
``publish.py`` does (the same folders and renames); then a fresh interpreter, with only that clone
on its path and no private Anjo anywhere, imports the example and runs it with ``--fake``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import anjo.sleep
from anjo.sleep.tests._git_fixtures import (
    Repo,
    git,
    isolate_git,
    make_trunk,
    needs_publish_tools,
)
from anjo.sleep.tools import boundary, publish

SLEEP = Path(anjo.sleep.__file__).parent
RUNNER = """
import runpy, sys
clone = sys.argv[1]
import os
# drop every path that holds private Anjo (an editable install of it, in this repo's venv)
sys.path[:] = [clone] + [p for p in sys.path if not os.path.isdir(os.path.join(p, "anjo", "core"))]
import importlib.util
assert importlib.util.find_spec("anjo.core") is None, "private Anjo is importable"
import anjo.sleep.examples.synthetic.run
sys.argv = ["python", "--fake", "--workdir", sys.argv[3], "--no-chat"]
runpy.run_module("anjo.sleep.examples.synthetic", run_name="__main__")
"""


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)


def _tree_files() -> dict[str, bytes]:
    files = {
        f"anjo/sleep/{path.relative_to(SLEEP).as_posix()}": path.read_bytes()
        for path in sorted(SLEEP.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    for old, new in publish.PATH_RENAMES:  # run inside a published tree, these sit at its root
        renamed = SLEEP.parents[1] / new
        if old not in files and renamed.is_file():
            files[old] = renamed.read_bytes()
    return files


def _published_clone(tmp_path: Path) -> Path:
    trunk: Repo = make_trunk(tmp_path)
    trunk.commit("feat: the engine and the example [t15]", _tree_files())
    clone = tmp_path / "published"
    git(tmp_path, "clone", "-q", "--no-local", str(trunk.path), str(clone))
    args = ["filter-repo", "--force", "--quiet", "--replace-refs", "delete-no-add"]
    for folder in boundary.PUBLISHED_FOLDERS:
        args += ["--path", folder]
    for old, new in publish.PATH_RENAMES:
        args += ["--path-rename", f"{old}:{new}"]
    git(clone, *args)
    return clone


@needs_publish_tools
def test_published_tree_imports_on_its_own(tmp_path: Path) -> None:
    clone = _published_clone(tmp_path)
    assert (clone / "pyproject.toml").is_file() and not (clone / "anjo" / "__init__.py").exists()
    repo_root = str(SLEEP.parents[1])
    work = tmp_path / "run"

    done = subprocess.run(
        [sys.executable, "-c", RUNNER, str(clone), repo_root, str(work)],
        cwd=clone, capture_output=True, text=True, timeout=300, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "no crisis gate" in done.stdout.lower()


def test_every_rename_source_exists_in_the_trunk() -> None:
    root = SLEEP.parents[1]  # the trunk, or a published tree where the files already sit at `new`
    missing = [old for old, new in publish.PATH_RENAMES
               if not (root / old).is_file() and not (root / new).is_file()]

    assert missing == []


@needs_publish_tools
def test_published_root_carries_the_agent_and_policy_files(tmp_path: Path) -> None:
    clone = _published_clone(tmp_path)

    for name in ("AGENTS.md", "llms.txt", "SAFETY.md", "SECURITY.md", "CONTRIBUTING.md",
                 "CITATION.cff", ".github/workflows/ci.yml"):
        assert (clone / name).is_file(), name
    assert not (clone / "anjo" / "sleep" / "AGENTS.md").exists()
    assert not (clone / "anjo" / "sleep" / "github").exists()
