"""The publish pipeline (spec §12.4, RF4): index, filter, scan, push, all against temp repos.

The remote is a local bare repo. Pipeline tests skip when git-filter-repo or gitleaks is
missing; the integration-index tests need neither.
"""

from __future__ import annotations

import random
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from anjo.sleep.tests._git_fixtures import (
    BASE,
    Repo,
    git,
    isolate_git,
    make_remote,
    make_trunk,
    needs_publish_tools,
    paths_in_history,
    remote_ref,
)
from anjo.sleep.tools import integration, publish

NOREPLY = "196453186+kevindechang@users.noreply.github.com"
# Built at runtime so this file never holds the planted value as one literal.
PLANTED = "plantedvalue" + "zq" + "4217"
DENY_TERM = "synthetic-deny" + "-term-91"
FIVE_MB = 5 * 1024 * 1024


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)


def _publish(
    trunk: Repo,
    remote: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
    env: Mapping[str, str] | None = None,
    deny: Sequence[str] | None = (DENY_TERM,),
) -> tuple[int, str]:
    env_file = tmp_path / "fake.env"
    values = {"FAKE_SERVICE_KEY": PLANTED} if env is None else env
    env_file.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    deny_file = trunk.path / "ops" / "sleep" / "publish_deny.txt"
    if deny is None:
        deny_file.unlink(missing_ok=True)
    else:
        deny_file.write_text("# synthetic terms\n" + "".join(f"{t}\n" for t in deny))
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    rc = publish.main(
        [
            "--repo", str(trunk.path),
            "--base", BASE,
            "--remote", str(remote),
            "--env-file", str(env_file),
            "--workdir", str(workdir),
            *extra,
        ]
    )
    captured = capsys.readouterr()
    return rc, captured.out + captured.err


def _head(out: str) -> str:
    match = re.search(r"^head ([0-9a-f]{40})$", out, re.MULTILINE)
    assert match, out
    return match.group(1)


def _reachable_objects(bare: Path) -> set[str]:
    out = git(bare, "rev-list", "--objects", "--all")
    return {line.split()[0] for line in out.splitlines() if line}


# ── Step 2: the integration index ─────────────────────────────────────────────


def test_integration_index_lists_only_added_lines(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    trunk.commit(
        "feat: route chat to the night model [t1]",
        {"anjo/core/chat.py": chat.replace("old_threshold = 0.25", "new_threshold = 0.5")},
    )

    index = integration.build_index(trunk.path, BASE)

    assert "`anjo/core/chat.py`" in index
    assert "+        new_threshold = 0.5" in index
    assert "old_threshold" not in index  # the removed line's text never appears
    assert "unchanged_body_line" not in index  # nor does unchanged context
    assert "feat: route chat to the night model [t1]" in index
    assert "+1 -1" in index


def test_integration_index_shows_enclosing_signature_without_body(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    hooked = chat.replace(
        "        old_threshold = 0.25\n",
        "        old_threshold = 0.25\n        night = night_route(text)\n",
    )
    trunk.commit("feat: hook the night route [t1]", {"anjo/core/chat.py": hooked})

    index = integration.build_index(trunk.path, BASE)

    assert "existing: def route(self, text):" in index
    assert "+        night = night_route(text)" in index
    for body in ("unchanged_body_line", "old_threshold", "return", "class Router", "helper"):
        assert body not in index


def test_integration_index_signature_skips_a_newly_added_def(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    hooked = chat.replace(
        "        return unchanged_body_line, old_threshold\n",
        "        return unchanged_body_line, old_threshold\n\n"
        "    def night(self):\n        return 1\n",
    )
    trunk.commit("feat: a night method [t1]", {"anjo/core/chat.py": hooked})

    index = integration.build_index(trunk.path, BASE)

    # The new def is an added line; the existing code it sits in is the class.
    assert "existing: class Router:" in index
    assert "+    def night(self):" in index
    assert "existing: def route" not in index


def test_integration_index_without_outside_changes_says_so(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n", "docs/a.md": "plan\n"})

    index = integration.build_index(trunk.path, BASE)

    assert "No changes outside the published folders" in index
    assert "docs/a.md" not in index


# ── Steps 1, 5 and 7: boundary, filter, push ─────────────────────────────────


@needs_publish_tools
def test_publish_stops_on_boundary_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: a helper [t1]", {"anjo/core/new_helper.py": "X = 1\n"})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert "boundary" in out and "anjo/core/new_helper.py" in out


@needs_publish_tools
def test_filtered_history_has_only_published_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    chat = trunk.read("anjo/core/chat.py")
    trunk.commit(
        "feat: engine and its hook [t1]",
        {
            "anjo/sleep/engine.py": "def night():\n    return 1\n",
            "mobile/features/sleep/Night.tsx": "export const Night = 1;\n",
            "anjo/core/chat.py": chat + "HOOK = 1\n",
        },
    )
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys)

    assert rc == 0, out
    assert paths_in_history(remote) == {
        "README.md",
        "LICENSE",
        "NOTICE",
        "anjo/sleep/__init__.py",
        "anjo/sleep/engine.py",
        "anjo/sleep/INTEGRATION.md",
        "mobile/features/sleep/Night.tsx",
    }
    root = set(git(remote, "ls-tree", "--name-only", "main").splitlines())
    assert {"README.md", "LICENSE", "NOTICE", "anjo", "mobile"} == root
    assert "+HOOK = 1" in git(remote, "show", "main:anjo/sleep/INTEGRATION.md")
    subjects = git(remote, "log", "--format=%s", "main").splitlines()
    assert "feat: engine and its hook [t1]" in subjects
    assert "chore: existing app" not in subjects  # pruned: touched nothing published
    emails = set(git(remote, "log", "--all", "--format=%ae%n%ce").splitlines())
    assert emails == {NOREPLY}


@needs_publish_tools
def test_docs_and_ops_sleep_are_exempt_and_unpublished(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from anjo.sleep.tools import boundary

    trunk = make_trunk(tmp_path)
    big = "".join(f"plan line {i}\n" for i in range(40))
    trunk.commit(
        "docs: planning notes [t1]",
        {"docs/plan.md": big, "ops/sleep/notes.txt": big, "anjo/sleep/engine.py": "X = 1\n"},
    )
    assert boundary.check(trunk.path, BASE) == []
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys)

    assert rc == 0, out
    paths = paths_in_history(remote)
    assert not [p for p in paths if p.startswith(("docs/", "ops/"))]
    index = git(remote, "show", "main:anjo/sleep/INTEGRATION.md")
    assert "docs/plan.md" not in index and "ops/sleep" not in index


@needs_publish_tools
def test_dry_run_prints_head_and_scan_clean_without_pushing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    before = trunk.head()
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 0, out
    _head(out)
    assert "scan clean" in out
    assert remote_ref(remote, "refs/heads/main") is None
    assert trunk.head() == before  # a dry run commits nothing on the trunk


@needs_publish_tools
def test_publish_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, _ = make_remote(tmp_path)

    first = _head(_publish(trunk, remote, tmp_path, capsys, "--dry-run")[1])
    second = _head(_publish(trunk, remote, tmp_path, capsys, "--dry-run")[1])
    assert first == second

    rc, out = _publish(trunk, remote, tmp_path, capsys)
    assert rc == 0, out
    published = _head(out)
    rc, out = _publish(trunk, remote, tmp_path, capsys)
    assert rc == 0, out
    assert _head(out) == published
    assert "up to date" in out


@needs_publish_tools
def test_real_publish_commits_the_integration_index_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    before = trunk.head()
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys)
    assert rc == 0, out
    after_first = trunk.head()
    assert after_first != before
    assert git(trunk.path, "log", "-1", "--format=%s") == "docs: regenerate the integration index"
    assert git(trunk.path, "show", "--name-only", "--format=", "HEAD") == "anjo/sleep/INTEGRATION.md"

    rc, out = _publish(trunk, remote, tmp_path, capsys)
    assert rc == 0, out
    assert trunk.head() == after_first


@needs_publish_tools
def test_first_publish_replaces_scaffold_and_leaves_prereg(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, scaffold = make_remote(tmp_path, scaffold=True)
    assert scaffold
    monkeypatch.setattr(publish, "SCAFFOLD_COMMIT", scaffold[:8])
    prereg = remote_ref(remote, "refs/heads/prereg")

    rc, out = _publish(trunk, remote, tmp_path, capsys)

    assert rc == 0, out
    assert remote_ref(remote, "refs/heads/main") == _head(out)
    assert "replaced the scaffold" in out
    assert remote_ref(remote, "refs/heads/prereg") == prereg


@needs_publish_tools
def test_push_refuses_force_after_first_publish(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, scaffold = make_remote(tmp_path, scaffold=True)
    assert scaffold
    monkeypatch.setattr(publish, "SCAFFOLD_COMMIT", scaffold[:8])
    assert _publish(trunk, remote, tmp_path, capsys)[0] == 0

    # Someone else's commit lands on the remote; the next publish no longer fast-forwards.
    other = Repo(tmp_path / "other")
    subprocess.run(["git", "clone", "-q", str(remote), str(other.path)], check=True)
    other.commit("chore: written by hand", {"extra.txt": "x\n"})
    git(other.path, "push", "-q", "origin", "main")
    foreign = remote_ref(remote, "refs/heads/main")
    trunk.commit("feat: more engine [t1]", {"anjo/sleep/more.py": "Y = 2\n"})

    rc, out = _publish(trunk, remote, tmp_path, capsys)

    assert rc == 1
    assert "refusing to force-push" in out
    assert remote_ref(remote, "refs/heads/main") == foreign


@needs_publish_tools
def test_force_refused_unless_strip_list_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: a note [t1]", {"anjo/sleep/oops.txt": "published by mistake\n"})
    blob = trunk.blob_id("anjo/sleep/oops.txt")
    trunk.commit("fix: drop the note [t1]", {"anjo/sleep/oops.txt": None})
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, _ = make_remote(tmp_path)
    assert _publish(trunk, remote, tmp_path, capsys)[0] == 0
    published = remote_ref(remote, "refs/heads/main")

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--allow-force", "no reason")
    assert rc == 1
    assert "only when a strip list changed" in out

    strip = trunk.path / "ops" / "sleep" / "strip_blobs.txt"
    strip.write_text(strip.read_text() + blob + "\n")
    rc, out = _publish(trunk, remote, tmp_path, capsys)
    assert rc == 1
    assert "refusing to force-push" in out
    assert remote_ref(remote, "refs/heads/main") == published

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--allow-force", "drop an accidental note")
    assert rc == 0, out
    assert "drop an accidental note" in out
    assert remote_ref(remote, "refs/heads/main") == _head(out) != published
    assert blob not in _reachable_objects(remote)

    # The strip list has not changed since that publish, so force is refused again.
    rc, out = _publish(trunk, remote, tmp_path, capsys, "--allow-force", "again")
    assert rc == 1


# ── Steps 4 and 6: the commit check and the scan ─────────────────────────────


@needs_publish_tools
def test_scan_blocks_env_value_removed_three_commits_ago(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: settings [t1]", {"anjo/sleep/settings.py": f"VALUE = '{PLANTED}'\n"})
    trunk.commit("fix: settings from the environment [t1]", {"anjo/sleep/settings.py": "VALUE = ''\n"})
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    trunk.commit("feat: more engine [t1]", {"anjo/sleep/more.py": "Y = 2\n"})
    trunk.commit("test: engine [t1]", {"anjo/sleep/tests/test_engine.py": "Z = 3\n"})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert "FAKE_SERVICE_KEY" in out
    assert "anjo/sleep/settings.py" in out
    assert PLANTED not in out  # a hit names the key, never the value


@needs_publish_tools
def test_stripped_blob_absent_from_published_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: settings [t1]", {"anjo/sleep/settings.py": f"VALUE = '{PLANTED}'\n"})
    blob = trunk.blob_id("anjo/sleep/settings.py")
    trunk.commit("fix: settings from the environment [t1]", {"anjo/sleep/settings.py": "VALUE = ''\n"})
    remote, _ = make_remote(tmp_path)
    assert _publish(trunk, remote, tmp_path, capsys)[0] == 1

    strip = trunk.path / "ops" / "sleep" / "strip_blobs.txt"
    strip.write_text(strip.read_text() + blob + "\n")
    rc, out = _publish(trunk, remote, tmp_path, capsys)

    assert rc == 0, out
    assert blob not in _reachable_objects(remote)
    assert PLANTED not in git(remote, "log", "--all", "-p")
    assert "VALUE = ''" in git(remote, "show", "main:anjo/sleep/settings.py")


@needs_publish_tools
def test_scan_blocks_denylisted_commit_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit(f"feat: tune for {DENY_TERM} [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert "denylist term 1" in out
    assert DENY_TERM not in out


@needs_publish_tools
def test_replace_messages_fixes_a_leaked_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit(f"feat: tune for {DENY_TERM} [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    (trunk.path / "ops" / "sleep" / "replace_messages.txt").write_text(
        f"# leaked text\n{DENY_TERM}==>[redacted]\n"
    )
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys)

    assert rc == 0, out
    log = git(remote, "log", "--format=%B", "main")
    assert "feat: tune for [redacted] [t1]" in log
    assert DENY_TERM not in log


@needs_publish_tools
def test_commit_check_rejects_unformatted_subject(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("added the engine", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert "commit format" in out


@needs_publish_tools
@pytest.mark.parametrize(
    ("rel", "size", "reason"),
    [
        ("anjo/sleep/adapter.gguf", 6, "refused extension"),
        ("anjo/sleep/weights.pt", 2, "refused extension"),
        ("anjo/sleep/big.dat", FIVE_MB + 1, "over 5 MB"),
    ],
    ids=["gguf", "pt", "over-5-mb"],
)
def test_scan_blocks_weights_and_large_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], rel: str, size: int, reason: str
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: an artifact [t1]", {rel: b"x" * size})
    trunk.commit("fix: drop the artifact [t1]", {rel: None})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert reason in out and rel in out


@needs_publish_tools
def test_gitleaks_finding_blocks_publish(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = random.Random(7)
    token = "AKIA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16))
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: cloud config [t1]", {"anjo/sleep/cloud.py": f'aws_id = "{token}"\n'})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert "gitleaks" in out and "anjo/sleep/cloud.py" in out
    assert token not in out


# ── Private config: the denylist and the message list stay out of git ─────────


@needs_publish_tools
@pytest.mark.parametrize("name", ["publish_deny.txt", "replace_messages.txt"])
def test_publish_refuses_tracked_denylist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("chore: oops [t1]", {f"ops/sleep/{name}": "# private\n"})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run")

    assert rc == 1
    assert "tracked" in out and name in out


@needs_publish_tools
def test_real_publish_refused_without_denylist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    remote, _ = make_remote(tmp_path)

    rc, out = _publish(trunk, remote, tmp_path, capsys, deny=None)
    assert rc == 1
    assert "denylist" in out
    assert remote_ref(remote, "refs/heads/main") is None

    rc, out = _publish(trunk, remote, tmp_path, capsys, deny=[])
    assert rc == 1

    rc, out = _publish(trunk, remote, tmp_path, capsys, "--dry-run", deny=None)
    assert rc == 0, out
    assert "WARNING" in out and "scan clean" in out


def test_default_env_file_is_the_main_checkouts(tmp_path: Path) -> None:
    trunk = make_trunk(tmp_path)
    worktree = tmp_path / "slot"
    git(trunk.path, "worktree", "add", "-q", "--detach", str(worktree))

    # Resolves the path only; the file is never opened here.
    assert publish.default_env_file(worktree) == (trunk.path / ".env").resolve()
    assert publish.default_env_file(trunk.path) == (trunk.path / ".env").resolve()


# ── Step 3: the results export ───────────────────────────────────────────────


@needs_publish_tools
def test_publish_exports_synthetic_results_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trunk = make_trunk(tmp_path)
    trunk.commit("feat: engine [t1]", {"anjo/sleep/engine.py": "X = 1\n"})
    data = tmp_path / "data"
    score = data / "sleep" / "results" / "synth-donor" / "scores" / "night-02.json"
    score.parent.mkdir(parents=True)
    score.write_text('{"score": 1}\n')
    ids = tmp_path / "synthetic_ids.txt"
    ids.write_text("# synthetic people\nsynth-donor\n")
    remote, _ = make_remote(tmp_path)
    export_args = ("--export-from", str(data), "--synthetic-ids", str(ids))

    rc, out = _publish(trunk, remote, tmp_path, capsys, *export_args)
    assert rc == 0, out
    assert "anjo/sleep/results/synth-donor/scores/night-02.json" in paths_in_history(remote)

    real = data / "sleep" / "results" / "someone-real" / "scores" / "a.json"
    real.parent.mkdir(parents=True)
    real.write_text("{}\n")
    before = trunk.head()
    rc, out = _publish(trunk, remote, tmp_path, capsys, *export_args)
    assert rc == 1
    assert "not on the synthetic list" in out
    assert trunk.head() == before
