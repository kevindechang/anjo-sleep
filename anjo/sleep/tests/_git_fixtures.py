"""Temp git repos for the boundary and publish tests.

Everything here is synthetic: a stand-in trunk with a few fake Anjo files, and a local bare
repo as the remote. Nothing touches GitHub or the network.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

HAVE_PUBLISH_TOOLS = bool(shutil.which("git-filter-repo") and shutil.which("gitleaks"))
needs_publish_tools = pytest.mark.skipif(
    not HAVE_PUBLISH_TOOLS, reason="git-filter-repo and gitleaks are not installed"
)

TEST_NAME = "Test Worker"
TEST_EMAIL = "worker@example.invalid"
BASE = "main"
TRUNK_BRANCH = "hackathon/sleep"

# A stand-in for an existing private module the glue plugs into.
CHAT_PY = '''"""Chat routing (synthetic stand-in for an existing module)."""


class Router:
    def route(self, text):
        unchanged_body_line = text.strip()
        old_threshold = 0.25
        return unchanged_body_line, old_threshold


def helper():
    return "helper body line"
'''

ROOT_GITIGNORE = """.env
ops/sleep/publish_deny.txt
ops/sleep/replace_messages.txt
ops/sleep/publish_state.json
"""

STRIP_BLOBS_HEADER = "# Blob ids to strip from the published history, one per line.\n"


def isolate_git(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point git at a throwaway global config so the person's hooks and signing never run."""
    config = home / "gitconfig"
    config.write_text(
        f"[user]\n\tname = {TEST_NAME}\n\temail = {TEST_EMAIL}\n"
        "[init]\n\tdefaultBranch = main\n"
        "[commit]\n\tgpgsign = false\n"
        "[tag]\n\tgpgsign = false\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-10-27T12:00:00+08:00")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-10-27T12:00:00+08:00")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GITLEAKS_CONFIG", "GITLEAKS_CONFIG_TOML"):
        monkeypatch.delenv(name, raising=False)


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


class Repo:
    """A small working repo that tests write commits into."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, rel: str) -> str:
        return (self.path / rel).read_text()

    def commit(self, message: str, files: Mapping[str, str | bytes | None]) -> str:
        """Write (or delete, for None) each file, stage exactly those paths, commit."""
        for rel, content in files.items():
            target = self.path / rel
            if content is None:
                git(self.path, "rm", "-q", "--", rel)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content)
            git(self.path, "add", "-f", "--", rel)
        git(self.path, "commit", "-q", "-m", message)
        return self.head()

    def head(self) -> str:
        return git(self.path, "rev-parse", "HEAD")

    def blob_id(self, rel: str) -> str:
        return git(self.path, "rev-parse", f"HEAD:{rel}")


def make_trunk(tmp_path: Path) -> Repo:
    """A fake Anjo: `main` with existing files, then the hackathon branch with the package."""
    path = tmp_path / "trunk"
    path.mkdir()
    git(path, "init", "-q", "-b", BASE)
    repo = Repo(path)
    repo.commit(
        "chore: existing app",
        {
            "README.md": "Private app.\n",
            ".gitignore": ROOT_GITIGNORE,
            "anjo/__init__.py": "",
            "anjo/core/chat.py": CHAT_PY,
            "anjo/dashboard/auth.py": "def login():\n    return None\n",
        },
    )
    git(path, "switch", "-q", "-c", TRUNK_BRANCH)
    repo.commit(
        "chore: start the anjo/sleep package",
        {
            "anjo/sleep/__init__.py": "",
            "anjo/sleep/README.md": "# anjo-sleep\n",
            "anjo/sleep/LICENSE": "Apache License 2.0 (test copy)\n",
            "anjo/sleep/NOTICE": "Built with test fixtures.\n",
            "ops/sleep/README.md": "Private publish config.\n",
            "ops/sleep/strip_blobs.txt": STRIP_BLOBS_HEADER,
        },
    )
    return repo


def make_remote(tmp_path: Path, *, scaffold: bool = False) -> tuple[Path, str | None]:
    """A local bare repo standing in for anjo-sleep.

    With ``scaffold``, its `main` is one parentless commit holding README, LICENSE and
    .gitignore, and it also carries a `prereg` branch, as the real repo does.
    """
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    if not scaffold:
        return bare, None
    seed = Repo(tmp_path / "scaffold")
    seed.path.mkdir()
    git(seed.path, "init", "-q", "-b", "main")
    sha = seed.commit(
        "chore: scaffold",
        {"README.md": "scaffold\n", "LICENSE": "license\n", ".gitignore": "*.pyc\n"},
    )
    git(seed.path, "push", "-q", str(bare), "main:refs/heads/main")
    git(seed.path, "switch", "-q", "--orphan", "prereg")
    seed.commit("chore: probe hashes", {"hashes.txt": "night-1 0000\n"})
    git(seed.path, "push", "-q", str(bare), "prereg:refs/heads/prereg")
    return bare, sha


def remote_ref(bare: Path, ref: str) -> str | None:
    out = subprocess.run(
        ["git", "-C", str(bare), "rev-parse", "--verify", "-q", ref],
        capture_output=True, text=True,
    )
    return out.stdout.strip() or None


def paths_in_history(bare: Path) -> set[str]:
    out = git(bare, "log", "--all", "--format=", "--name-only", "--no-renames")
    return {line for line in out.splitlines() if line}
