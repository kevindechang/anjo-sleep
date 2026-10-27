"""Publish the hackathon folders to ``anjo-sleep`` (spec §12.4). Only the orchestrator runs it.

1. Boundary check (``boundary.check``); any violation stops the publish.
2. Regenerate ``anjo/sleep/INTEGRATION.md`` and commit it when it changed.
3. With ``--export-from <data_dir> --synthetic-ids <file>``, export synthetic results into
   ``anjo/sleep/results/`` and commit them. New content from steps 2 and 3 is scanned
   before it is committed, so a hit never lands in the trunk's history.
4. Commit check: every commit since the branch point has a ``<type>: ...`` subject and no
   denylist term (after the replace-message rules).
5. Filter a ``git clone --no-local`` of the trunk with ``git filter-repo``: the two published
   folders only, README/LICENSE/NOTICE/pyproject/.gitignore renamed to the root, every
   author and committer email mapped to the GitHub noreply address, and the strip lists
   applied. The same input gives the same head.
6. Scan the filtered history: gitleaks, every env-file value (read here, never printed), the
   denylist, files over 5 MB and weights extensions.
7. Push ``main`` without force. The one exception without a flag is the first publish, which
   replaces the scaffold commit once with ``--force-with-lease``. ``--allow-force "<reason>"``
   is accepted only when a strip list changed since the last publish.
8. Report the head, the file count, the scan result and what changed since the last publish.

``--dry-run`` runs steps 1-6 and plans step 7 without committing on the trunk or pushing:
the index and results are committed only inside the throwaway clone, with dates pinned to
the trunk head so the reported head is stable. The ``prereg`` branch is never touched.

The private inputs live in ``ops/sleep/`` (see its README): the denylist and the message
replacements must be untracked, and the script refuses to run if either is tracked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from anjo.sleep.tools import _git, boundary, integration, scan
from anjo.sleep.tools import export_results as results
from anjo.sleep.tools import strip_lists

DEFAULT_REMOTE = "git@github.com:kevindechang/anjo-sleep.git"
NOREPLY_EMAIL = "196453186+kevindechang@users.noreply.github.com"
SCAFFOLD_COMMIT = "6c3eae89"  # anjo-sleep's three scaffold files, 2026-10-26
PUBLISHED_REF = "refs/heads/main"

OPS_DIR = "ops/sleep"
DENYLIST = "publish_deny.txt"
REPLACE_MESSAGES = "replace_messages.txt"
STRIP_BLOBS = "strip_blobs.txt"
STATE_FILE = "publish_state.json"
MUST_BE_UNTRACKED = (DENYLIST, REPLACE_MESSAGES)

RESULTS_DIR = "anjo/sleep/results"
INDEX_MESSAGE = "docs: regenerate the integration index"
RESULTS_MESSAGE = "chore: export synthetic results"
PATH_RENAMES = (
    ("anjo/sleep/README.md", "README.md"),
    ("anjo/sleep/LICENSE", "LICENSE"),
    ("anjo/sleep/NOTICE", "NOTICE"),
    ("anjo/sleep/pyproject.public.toml", "pyproject.toml"),
    ("anjo/sleep/.gitignore", ".gitignore"),
)

PushAction = Literal["create", "up-to-date", "fast-forward", "replace-scaffold", "forced"]


class PublishRefused(RuntimeError):
    """A check failed; nothing was pushed."""


@dataclass(frozen=True)
class Options:
    repo: Path
    base: str
    remote: str
    env_file: Path | None
    env_skip_keys: frozenset[str]
    dry_run: bool
    allow_force: str | None
    export_from: Path | None
    synthetic_ids: Path | None
    workdir: Path | None


@dataclass(frozen=True)
class Config:
    terms: list[str]
    env: scan.EnvValues
    strip_ids: list[str]
    rules: list[strip_lists.Replacement]
    digests: dict[str, str]
    last: dict[str, Any] | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PushPlan:
    action: PushAction
    remote_head: str | None


# ── Inputs ────────────────────────────────────────────────────────────────────


def default_env_file(repo: Path) -> Path:
    """The main checkout's ``.env``, found through the shared git dir. Only resolves the path."""
    common = Path(_git.text(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return (common.parent / ".env").resolve()


def _ops(repo: Path, name: str) -> Path:
    return repo / OPS_DIR / name


def _refuse_tracked(repo: Path) -> None:
    for name in MUST_BE_UNTRACKED:
        rel = f"{OPS_DIR}/{name}"
        tracked = _git.text(repo, "ls-files", "--", rel) or _git.text(
            repo, "ls-tree", "--name-only", "HEAD", "--", rel
        )
        if tracked:
            raise PublishRefused(
                f"{rel} is tracked by git. It must stay untracked so worker slots never hold "
                f"it: `git rm --cached {rel}`, commit, and keep the file only in the trunk."
            )


def _load_state(repo: Path, remote: str) -> dict[str, Any] | None:
    path = _ops(repo, STATE_FILE)
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    entry = state.get(remote) if isinstance(state, dict) else None
    return entry if isinstance(entry, dict) else None


def _save_state(repo: Path, remote: str, entry: dict[str, Any]) -> None:
    path = _ops(repo, STATE_FILE)
    state = json.loads(path.read_text()) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**state, remote: entry}, indent=2, sort_keys=True) + "\n")


def _load_env(repo: Path, opts: Options, warnings: list[str]) -> scan.EnvValues:
    env_file = opts.env_file or default_env_file(repo)
    if env_file.exists():
        return scan.load_env_values(env_file, opts.env_skip_keys)
    if not opts.dry_run:
        raise PublishRefused(f"the env file {env_file} is missing; a real publish scans its values")
    warnings.append(f"no env file at {env_file}; env values were not checked")
    return scan.EnvValues({}, 0)


def _load_config(repo: Path, opts: Options) -> Config:
    warnings: list[str] = []
    terms = scan.load_terms(_ops(repo, DENYLIST))
    if not terms:
        if not opts.dry_run:
            raise PublishRefused(
                f"the denylist {OPS_DIR}/{DENYLIST} is missing or empty; a real publish needs it"
            )
        warnings.append(f"the denylist {OPS_DIR}/{DENYLIST} is missing or empty; no terms checked")
    env = _load_env(repo, opts, warnings)
    try:
        strip_ids = strip_lists.load_strip_blobs(_ops(repo, STRIP_BLOBS))
        rules = strip_lists.load_replacements(_ops(repo, REPLACE_MESSAGES))
    except strip_lists.StripListError as exc:
        raise PublishRefused(str(exc)) from exc
    digests = strip_lists.digest(strip_ids, rules)
    last = _load_state(repo, opts.remote)
    if opts.allow_force is not None and (last is None or last.get("digests") == digests):
        raise PublishRefused(
            "--allow-force is accepted only when a strip list changed since the last publish"
        )
    return Config(terms, env, strip_ids, rules, digests, last, tuple(warnings))


def _require_tools() -> None:
    missing = [tool for tool in ("git-filter-repo", "gitleaks") if not shutil.which(tool)]
    if missing:
        raise PublishRefused(f"missing tools: {', '.join(missing)} (brew install them)")


def _check_boundary(repo: Path, base: str) -> None:
    violations = boundary.check(repo, base)
    if violations:
        lines = "\n".join(f"  {v.reason:<17} {v.path}" for v in violations)
        raise PublishRefused(f"boundary check failed against {base}:\n{lines}")


def _branch(repo: Path) -> str:
    branch = _git.text(repo, "branch", "--show-current")
    if not branch:
        raise PublishRefused("HEAD is detached; publish runs from the trunk branch")
    return branch


def _refuse_findings(step: str, findings: list[scan.Finding]) -> None:
    if findings:
        lines = "\n".join(f"  {finding.render()}" for finding in findings)
        raise PublishRefused(f"{step} found {len(findings)} problem(s):\n{lines}")


# ── Steps 2 and 3: new content for the trunk ─────────────────────────────────


def _committed_text(repo: Path, rel: str) -> str | None:
    if not _git.ok(repo, "cat-file", "-e", f"HEAD:{rel}"):
        return None
    return _git.run(repo, "show", f"HEAD:{rel}").decode("utf-8", "replace")


def _stage(repo: Path, opts: Options, cfg: Config, staging: Path) -> bool:
    """Write the new index and exported results under ``staging`` and scan them there.

    Returns whether the index changed.
    """
    staging.mkdir(parents=True)

    def rewrite(subject: str) -> str:
        return strip_lists.apply_replacements(subject.encode(), cfg.rules).decode("utf-8", "replace")

    index = integration.build_index(repo, opts.base, rewrite)
    index_changed = index != _committed_text(repo, integration.INDEX_PATH)
    if index_changed:
        target = staging / integration.INDEX_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(index)
    if opts.export_from is not None:
        if opts.synthetic_ids is None:
            raise PublishRefused("--export-from needs --synthetic-ids <file>")
        try:
            ids = results.read_synthetic_ids(opts.synthetic_ids)
            results.export(opts.export_from, staging / RESULTS_DIR, ids)
        except results.ExportRefused as exc:
            raise PublishRefused(f"results export refused: {exc}") from exc
    _refuse_findings("the scan of new content", scan.scan_folder(staging, cfg.env, cfg.terms))
    return index_changed


def _pinned_env(repo: Path) -> dict[str, str]:
    date = _git.text(repo, "log", "-1", "--format=%cI", "HEAD")
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "anjo-sleep publish", "GIT_AUTHOR_EMAIL": NOREPLY_EMAIL,
        "GIT_COMMITTER_NAME": "anjo-sleep publish", "GIT_COMMITTER_EMAIL": NOREPLY_EMAIL,
        "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
    }


def _commit_staged(target: Path, staging: Path, *, pinned: bool) -> list[str]:
    """Copy the staged files into ``target`` and commit each group that changed."""
    groups = ((integration.INDEX_PATH, INDEX_MESSAGE), (RESULTS_DIR, RESULTS_MESSAGE))
    made: list[str] = []
    for rel, message in groups:
        source = staging / rel
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, target / rel, dirs_exist_ok=True)
        else:
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / rel)
        _git.run(target, "add", "--", rel)
        if _git.ok(target, "diff", "--cached", "--quiet", "--", rel):
            continue
        env = _pinned_env(target) if pinned else None
        flags = ("-c", "commit.gpgsign=false", "commit", "--no-verify") if pinned else ("commit",)
        _git.run(target, *flags, "-q", "-m", message, "--", rel, env=env)
        made.append(_git.text(target, "rev-parse", "HEAD"))
    return made


# ── Step 5: filter ────────────────────────────────────────────────────────────


def _clone(repo: Path, branch: str, dest: Path) -> None:
    _git.run(
        repo, "clone", "-q", "--no-local", "--no-tags", "--single-branch",
        "--branch", branch, str(repo), str(dest),
    )


def _write_mailmap(clone: Path, dest: Path) -> None:
    emails = sorted(set(_git.text(clone, "log", "--all", "--format=%ae%n%ce").splitlines()))
    dest.write_text("".join(f"<{NOREPLY_EMAIL}> <{email}>\n" for email in emails))


def _filter(clone: Path, work: Path, cfg: Config) -> str:
    if not clone.resolve().is_relative_to(work.resolve()):
        raise PublishRefused("refusing to filter a repo outside the publish workdir")
    mailmap, blobs, messages = work / "mailmap", work / "strip_blobs", work / "replace_messages"
    _write_mailmap(clone, mailmap)
    args = ["filter-repo", "--force", "--quiet", "--replace-refs", "delete-no-add"]
    for folder in boundary.PUBLISHED_FOLDERS:
        args += ["--path", folder]
    for old, new in PATH_RENAMES:
        args += ["--path-rename", f"{old}:{new}"]
    args += ["--mailmap", str(mailmap)]
    if cfg.strip_ids:
        strip_lists.write_strip_blobs(cfg.strip_ids, blobs)
        args += ["--strip-blobs-with-ids", str(blobs)]
    if cfg.rules:
        strip_lists.write_replacements(cfg.rules, messages)
        args += ["--replace-message", str(messages)]
    _git.run(clone, *args)
    return _git.text(clone, "rev-parse", "HEAD")


# ── Step 7: push ──────────────────────────────────────────────────────────────


def _remote_head(clone: Path, remote: str) -> str | None:
    line = _git.text(clone, "ls-remote", remote, PUBLISHED_REF)
    return line.split()[0] if line else None


def _is_scaffold(clone: Path, sha: str) -> bool:
    return sha.startswith(SCAFFOLD_COMMIT) and _git.text(clone, "rev-list", "--count", sha) == "1"


def _plan_push(clone: Path, opts: Options, head: str) -> PushPlan:
    remote_head = _remote_head(clone, opts.remote)
    if remote_head is None:
        return PushPlan("create", None)
    if remote_head == head:
        return PushPlan("up-to-date", remote_head)
    _git.run(clone, "fetch", "-q", "--no-tags", opts.remote, f"{PUBLISHED_REF}:refs/publish/remote-main")
    if _git.ok(clone, "merge-base", "--is-ancestor", remote_head, head):
        return PushPlan("fast-forward", remote_head)
    if _is_scaffold(clone, remote_head):
        return PushPlan("replace-scaffold", remote_head)
    if opts.allow_force is not None:
        return PushPlan("forced", remote_head)
    raise PublishRefused(
        f"remote main {remote_head[:12]} is not an ancestor of the new head {head[:12]}; "
        "refusing to force-push. A rewrite is allowed only with --allow-force \"<reason>\" "
        "after a strip-list change."
    )


def _push(clone: Path, remote: str, head: str, plan: PushPlan) -> None:
    refspec = f"{head}:{PUBLISHED_REF}"  # main only; never prereg or any other ref
    if plan.action in ("create", "fast-forward"):
        _git.run(clone, "push", "-q", remote, refspec)
    elif plan.action in ("replace-scaffold", "forced") and plan.remote_head:
        lease = f"--force-with-lease={PUBLISHED_REF}:{plan.remote_head}"
        _git.run(clone, "push", "-q", lease, remote, refspec)


# ── Step 8: report ────────────────────────────────────────────────────────────


def _describe_push(plan: PushPlan, opts: Options) -> str:
    prefix = "would push" if opts.dry_run else "pushed"
    old = plan.remote_head[:12] if plan.remote_head else ""
    return {
        "create": f"{prefix} main to an empty remote",
        "up-to-date": f"up to date (remote main is already {old})",
        "fast-forward": f"{prefix} main, fast-forward from {old}",
        "replace-scaffold": f"{prefix} main: first publish, replaced the scaffold {old} with --force-with-lease",
        "forced": f"{prefix} main with --force-with-lease over {old} (reason: {opts.allow_force})",
    }[plan.action]


def _since_last(clone: Path, cfg: Config, head: str) -> str:
    last = (cfg.last or {}).get("head")
    if not isinstance(last, str):
        return "first publish to this remote"
    if last == head:
        return "nothing"
    if _git.ok(clone, "merge-base", "--is-ancestor", last, head):
        count = _git.text(clone, "rev-list", "--count", f"{last}..{head}")
        stat = _git.text(clone, "diff", "--shortstat", last, head) or "no file changes"
        return f"{count} commit(s); {stat}"
    return f"history rewritten since {last[:12]}"


def _report(opts: Options, cfg: Config, clone: Path, head: str, plan: PushPlan, index: str) -> str:
    files = len(_git.text(clone, "ls-tree", "-r", "--name-only", head).splitlines())
    skipped = ", ".join(sorted(opts.env_skip_keys)) or "none"
    lines = [
        f"publish report{' (dry run)' if opts.dry_run else ''}",
        f"head {head}",
        f"files {files}",
        "scan clean",
        f"push: {_describe_push(plan, opts)}",
        f"since the last publish: {_since_last(clone, cfg, head)}",
        f"integration index: {index}",
        f"env scan: {len(cfg.env.values)} value(s) checked, {cfg.env.skipped_short} too short "
        f"to be secrets, keys skipped by flag: {skipped}",
    ]
    lines += [f"WARNING: {warning}" for warning in cfg.warnings]
    return "\n".join(lines)


# ── The pipeline ──────────────────────────────────────────────────────────────


@contextmanager
def _workspace(parent: Path | None) -> Iterator[Path]:
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="anjo-sleep-publish-", dir=parent))
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _index_note(changed: bool, made: list[str], dry_run: bool) -> str:
    if not changed:
        return "unchanged"
    if dry_run:
        return "regenerated and committed in the throwaway clone only (dry run)"
    return f"committed on the trunk as {made[0][:12]}" if made else "unchanged"


def run(opts: Options) -> str:
    repo = _git.toplevel(opts.repo)
    _refuse_tracked(repo)
    cfg = _load_config(repo, opts)
    _require_tools()
    _check_boundary(repo, opts.base)
    branch = _branch(repo)
    since = _git.merge_base(repo, opts.base)
    with _workspace(opts.workdir) as work:
        staging, clone = work / "staging", work / "clone"
        index_changed = _stage(repo, opts, cfg, staging)
        made = [] if opts.dry_run else _commit_staged(repo, staging, pinned=False)
        _clone(repo, branch, clone)
        if opts.dry_run:
            _commit_staged(clone, staging, pinned=True)
        _refuse_findings("the commit check", scan.check_commits(clone, since, "HEAD", cfg.terms, cfg.rules))
        head = _filter(clone, work, cfg)
        _refuse_findings("the scan", scan.scan_history(clone, cfg.env, cfg.terms, expected_email=NOREPLY_EMAIL))
        plan = _plan_push(clone, opts, head)
        if not opts.dry_run:
            _push(clone, opts.remote, head, plan)
            _save_state(repo, opts.remote, {
                "head": head,
                "digests": cfg.digests,
                "forced_reason": opts.allow_force if plan.action == "forced" else None,
                "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        return _report(opts, cfg, clone, head, plan, _index_note(index_changed, made, opts.dry_run))


def _parse(argv: list[str] | None) -> Options:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.tools.publish")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="a path inside the trunk")
    parser.add_argument("--base", default=boundary.DEFAULT_BASE)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--env-file", type=Path, help="default: the main checkout's .env")
    parser.add_argument("--env-skip-key", action="append", default=[], metavar="KEY",
                        help="leave one env key's value out of the scan (named in the report)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-force", metavar="REASON")
    parser.add_argument("--export-from", type=Path, metavar="DATA_DIR")
    parser.add_argument("--synthetic-ids", type=Path, metavar="FILE")
    parser.add_argument("--workdir", type=Path, help="where the throwaway clone goes")
    a = parser.parse_args(argv)
    return Options(
        a.repo, a.base, a.remote, a.env_file, frozenset(a.env_skip_key), a.dry_run,
        a.allow_force, a.export_from, a.synthetic_ids, a.workdir,
    )


def main(argv: list[str] | None = None) -> int:
    opts = _parse(argv)
    try:
        print(run(opts))
    except PublishRefused as exc:
        print(f"publish refused: {exc}", file=sys.stderr)
        return 1
    except (_git.GitError, RuntimeError, OSError, ValueError) as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
