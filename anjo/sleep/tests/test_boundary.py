"""The boundary check: only small glue lands outside the published folders (spec §12.1, §12.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.tests._git_fixtures import BASE, git, isolate_git, make_trunk
from anjo.sleep.tools import boundary


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)


def _found(repo: Path) -> set[tuple[str, str]]:
    return {(v.path, v.reason) for v in boundary.check(repo, BASE)}


def _lines(prefix: str, count: int) -> str:
    return "".join(f"{prefix}_{i} = {i}\n" for i in range(count))


def test_new_file_outside_published_folders_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: add a helper", {"anjo/core/new_helper.py": "X = 1\n"})

    assert _found(trunk.path) == {("anjo/core/new_helper.py", "new_outside_file")}


def test_glue_over_20_added_lines_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    trunk.commit("feat: a hook", {"anjo/core/chat.py": chat + _lines("HOOK", 20)})
    assert _found(trunk.path) == set()

    # The cap counts the whole branch diff, not one commit.
    trunk.commit("feat: one more", {"anjo/core/chat.py": trunk.read("anjo/core/chat.py") + "LAST = 1\n"})

    assert _found(trunk.path) == {("anjo/core/chat.py", "glue_too_big")}


def test_change_under_private_path_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    auth = trunk.read("anjo/dashboard/auth.py")
    trunk.commit("fix: a tweak", {"anjo/dashboard/auth.py": auth + "TWEAK = 1\n"})

    assert _found(trunk.path) == {("anjo/dashboard/auth.py", "private_path")}


def test_deleting_a_private_file_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("chore: drop a file", {"anjo/dashboard/auth.py": None})

    assert _found(trunk.path) == {("anjo/dashboard/auth.py", "private_path")}


@pytest.mark.parametrize(
    "path",
    [
        "anjo/dashboard/auth.py",
        "anjo/dashboard/routes/auth_routes.py",
        "anjo/admin_mcp/server.py",
        "anjo/dashboard/routes/admin_routes.py",
        "anjo/dashboard/routes/billing_routes.py",
        "anjo/core/subscription.py",
        ".github/workflows/deploy.yml",
        "deploy.sh",
        "scripts/safe_deploy_prod.sh",
        "anjo/core/crisis.py",
        "anjo/core/crisis_reply.py",
        "anjo/evals/crisis_cases.py",
        "mobile/lib/crisis/rules.json",
        "anjo/core/output_safety.py",
        "anjo/core/output_guard_rules.py",
        "anjo/core/floor.py",
        "mobile/lib/floor/output-guard.ts",
        "anjo/core/age_lock.py",
        "anjo/dashboard/routes/age_lock_routes.py",
        "tests/test_crisis_detection.py",
    ],
)
def test_private_paths_cover_each_named_area(path: str) -> None:
    assert boundary.is_private(path)


@pytest.mark.parametrize(
    "path", ["anjo/core/chat_completions.py", "anjo/core/outreach.py", "pyproject.toml"]
)
def test_ordinary_app_files_are_not_private(path: str) -> None:
    assert not boundary.is_private(path)


def test_published_folders_docs_and_ops_sleep_are_not_checked(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    big = _lines("LINE", 50)
    trunk.commit(
        "docs: plan and engine",
        {
            "docs/hackathon/plan.md": big,
            "ops/sleep/notes.txt": big,
            "anjo/sleep/engine.py": big,
            "mobile/features/sleep/Night.tsx": big,
        },
    )

    assert _found(trunk.path) == set()


def test_other_ops_folders_are_not_exempt(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("docs: a decision", {"ops/decisions/note.md": "x\n"})

    assert _found(trunk.path) == {("ops/decisions/note.md", "new_outside_file")}


def test_diff_starts_at_the_merge_base(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    # main moves on after the branch point; its new files are not the branch's changes.
    git(trunk.path, "switch", "-q", BASE)
    trunk.commit("feat: main moves on", {"anjo/core/later.py": "Y = 2\n"})
    git(trunk.path, "switch", "-q", "hackathon/sleep")

    assert _found(trunk.path) == set()


def test_cli_prints_violations_and_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trunk = make_trunk(tmp_path)
    assert boundary.main(["--repo", str(trunk.path), "--base", BASE]) == 0
    assert "boundary clean" in capsys.readouterr().out

    trunk.commit("feat: add a helper", {"anjo/core/new_helper.py": "X = 1\n"})

    assert boundary.main(["--repo", str(trunk.path), "--base", BASE]) == 1
    out = capsys.readouterr().out
    assert "new_outside_file" in out
    assert "anjo/core/new_helper.py" in out
