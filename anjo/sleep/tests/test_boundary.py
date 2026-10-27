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

    assert _found(trunk.path) == {
        ("anjo/dashboard/auth.py", "private_path"),
        ("anjo/dashboard/auth.py", "not_modified_in_place"),
    }


def test_deleting_an_outside_file_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("chore: drop chat", {"anjo/core/chat.py": None})

    assert _found(trunk.path) == {("anjo/core/chat.py", "not_modified_in_place")}


def test_symlink_swap_outside_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    git(trunk.path, "rm", "-q", "anjo/core/chat.py")
    (trunk.path / "anjo/core").mkdir(exist_ok=True)
    (trunk.path / "anjo/core/chat.py").symlink_to("../sleep/README.md")
    git(trunk.path, "add", "anjo/core/chat.py")
    git(trunk.path, "commit", "-q", "-m", "fix: point chat elsewhere")

    assert _found(trunk.path) == {("anjo/core/chat.py", "not_modified_in_place")}


def test_moving_private_code_into_a_published_folder_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    git(trunk.path, "mv", "anjo/core/chat.py", "anjo/sleep/x.py")
    git(trunk.path, "commit", "-q", "-m", "refactor: move chat")

    assert _found(trunk.path) == {
        ("anjo/sleep/x.py", "moved_private_code"),
        ("anjo/core/chat.py", "moved_private_code"),
        ("anjo/core/chat.py", "not_modified_in_place"),
    }


def test_copying_private_code_into_a_published_folder_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: a copy", {"mobile/features/sleep/chat.py": trunk.read("anjo/core/chat.py")})

    assert _found(trunk.path) == {("mobile/features/sleep/chat.py", "moved_private_code")}


def test_copying_a_file_main_added_after_the_branch_point_is_a_violation(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    later = "".join(f"PRIVATE_LATER_{i} = {i}\n" for i in range(10))
    git(trunk.path, "switch", "-q", BASE)
    trunk.commit("feat: main moves on", {"anjo/core/later.py": later})
    git(trunk.path, "switch", "-q", "hackathon/sleep")
    trunk.commit("feat: a copy", {"anjo/sleep/later.py": later})

    assert _found(trunk.path) == {("anjo/sleep/later.py", "moved_private_code")}


def test_a_stock_license_text_is_not_moved_code(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    license_text = "Apache License\nVersion 2.0, January 2004\n" + "Terms and conditions.\n" * 5
    git(trunk.path, "switch", "-q", BASE)
    trunk.commit("chore: vendored", {"third_party/lib/LICENSE": license_text})
    git(trunk.path, "switch", "-q", "hackathon/sleep")
    trunk.commit("chore: license", {"anjo/sleep/LICENSE": license_text})

    assert _found(trunk.path) == set()


def test_tiny_common_files_are_not_moved_code(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    # An empty __init__.py matches the base's empty anjo/__init__.py; it carries no code.
    trunk.commit("feat: a subpackage", {"anjo/sleep/night/__init__.py": ""})

    assert _found(trunk.path) == set()


def test_one_80k_character_line_is_glue_too_big(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    trunk.commit("fix: long", {"anjo/core/chat.py": chat + "x=1;" * 20000 + "\n"})

    assert _found(trunk.path) == {("anjo/core/chat.py", "glue_too_big")}


def test_a_line_over_300_bytes_is_glue_too_big(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    trunk.commit("fix: ok", {"anjo/core/chat.py": chat + "#" * 300 + "\n"})
    assert _found(trunk.path) == set()

    trunk.commit("fix: long", {"anjo/core/chat.py": trunk.read("anjo/core/chat.py") + "#" * 301 + "\n"})

    assert _found(trunk.path) == {("anjo/core/chat.py", "glue_too_big")}


def test_added_bytes_over_6_kb_per_file_is_glue_too_big(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 20 lines of at most 300 bytes stay under 6 KB, so lift the line cap to test the file cap.
    monkeypatch.setattr(boundary, "GLUE_LINE_BYTE_CAP", 10_000)
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    trunk.commit("fix: wide", {"anjo/core/chat.py": chat + ("#" * 700 + "\n") * 8})
    assert _found(trunk.path) == set()

    trunk.commit("fix: wider", {"anjo/core/chat.py": trunk.read("anjo/core/chat.py") + "#" * 700 + "\n"})

    assert _found(trunk.path) == {("anjo/core/chat.py", "glue_too_big")}


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
        "scripts/env_flip.sh",
        "scripts/server_setup.sh",
        "scripts/release_manager.sh",
        "scripts/restore_full_state.sh",
        "scripts/preflight.sh",
        "scripts/ios_internal_build.sh",
        "scripts/pre_submission_check.sh",
        "scripts/build_release_bundle.py",
        "mobile/ios/ci_scripts/ci_post_clone.sh",
        "mobile/ios/bump_build.sh",
        "sync-to-public.sh",
        "ops/monitoring/anjo_monitor.sh",
        "anjo/evals/local_floor.py",
        "anjo/evals/local_floor_sets.py",
        "anjo/evals/launch_safety_cases.py",
        "anjo/evals/medical_gate.py",
        "anjo/core/safety.py",
        "tests/network_guard.py",
        "mobile/app/(app)/safety.tsx",
    ],
)
def test_private_paths_cover_each_named_area(path: str) -> None:
    assert boundary.is_private(path)


@pytest.mark.parametrize(
    "path",
    [
        "anjo/core/chat_completions.py",
        "anjo/core/outreach.py",
        "pyproject.toml",
        "anjo/dashboard/session_store.py",
        "anjo/dashboard/session_admission.py",
        "tests/conftest.py",
    ],
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
