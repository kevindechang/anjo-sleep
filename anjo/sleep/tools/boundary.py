"""The boundary check: hackathon code stays in the published folders (spec §12.1, §12.4 step 1).

It reads the branch's own changes, ``merge-base(base, HEAD)..HEAD``, and reports:

- ``new_outside_file``: a file added outside the published folders
- ``glue_too_big``: an outside file with more than 20 added lines across the branch
- ``private_path``: any change, deletion included, under a private path (auth, admin,
  billing, deploy, and the crisis, output-guard and age-lock modules)

``docs/`` and ``ops/sleep/`` are private planning: exempt here and never published.

Run it with ``python -m anjo.sleep.tools.boundary [--base origin/main]``. It prints the
violations and exits 1 on any.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal

from anjo.sleep.tools import _git

Reason = Literal["new_outside_file", "glue_too_big", "private_path"]

PUBLISHED_FOLDERS: tuple[str, ...] = ("anjo/sleep/", "mobile/features/sleep/")
EXEMPT_FOLDERS: tuple[str, ...] = ("docs/", "ops/sleep/")
GLUE_LINE_CAP = 20
DEFAULT_BASE = "origin/main"

# Glob patterns (fnmatch: `*` also crosses `/`). Each area lists its modules and their tests,
# because editing a module's tests can weaken it as surely as editing the module.
PRIVATE_PATHS: tuple[str, ...] = (
    # auth and sign-in
    "anjo/dashboard/auth.py",
    "anjo/dashboard/routes/auth_routes.py",
    "anjo/dashboard/routes/mobile_auth_routes.py",
    "mobile/app/(auth)/*",
    "mobile/lib/auth-context.tsx",
    "mobile/lib/device-bound-auth*",
    "mobile/lib/apple-signin.ts",
    "mobile/lib/apple-reauth.ts",
    "mobile/lib/password-rules*",
    "tests/test_auth*",
    # admin
    "anjo/admin_mcp/*",
    "anjo/dashboard/routes/admin_*",
    "anjo/dashboard/static/admin.html",
    "tests/test_admin*",
    # billing
    "anjo/core/subscription.py",
    "anjo/dashboard/routes/billing_routes.py",
    "mobile/app/(app)/billing.tsx",
    "mobile/features/billing/*",
    "mobile/lib/purchases.ts",
    "tests/test_billing*",
    # deploy
    ".github/workflows/*",
    "deploy.sh",
    "scripts/safe_deploy_prod.sh",
    "scripts/post_deploy_smoke.py",
    "scripts/backup_db.sh",
    "anjo/dashboard/deploy_state.py",
    "tests/test_*deploy*",
    # crisis
    "anjo/core/crisis*",
    "anjo/evals/crisis*",
    "anjo/floor_data/crisis/*",
    "mobile/lib/crisis/*",
    "mobile/lib/crisis-resources*",
    "mobile/assets/crisis_resources.json",
    "scripts/*crisis*",
    "tests/test_*crisis*",
    # output guard: the server floor and the reply guards
    "anjo/core/floor.py",
    "anjo/core/output_safety.py",
    "anjo/core/output_guard_rules.py",
    "anjo/core/medical_boundary.py",
    "anjo/floor_data/*",
    "anjo/evals/output_guard*",
    "mobile/lib/floor/*",
    "mobile/lib/device-floor*",
    "scripts/export_output_guards.py",
    "scripts/floor_guard_eval.py",
    "tests/test_output_safety*",
    "tests/test_floor_data_parity*",
    "tests/test_medical_boundary*",
    # age lock
    "anjo/core/age_lock.py",
    "anjo/core/age_assurance.py",
    "anjo/dashboard/routes/age_lock_routes.py",
    "tests/test_age_lock*",
    "tests/test_age_assurance*",
)


@dataclass(frozen=True)
class Violation:
    path: str
    reason: Reason


@dataclass(frozen=True)
class Change:
    """One file the branch changed: its status letter and added-line count (None if binary)."""

    path: str
    status: str
    added: int | None
    removed: int | None


def is_published(path: str) -> bool:
    return path.startswith(PUBLISHED_FOLDERS)


def is_exempt(path: str) -> bool:
    return path.startswith(EXEMPT_FOLDERS)


def is_private(path: str) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in PRIVATE_PATHS)


def is_outside(path: str) -> bool:
    """A path the hackathon may only touch as glue: not published and not exempt."""
    return not is_published(path) and not is_exempt(path)


def _counts(field: str) -> int | None:
    return None if field == "-" else int(field)


def branch_changes(repo: Path, base: str) -> list[Change]:
    """Every file changed in ``merge-base(base, HEAD)..HEAD``, renames split into delete + add."""
    since = _git.merge_base(repo, base)
    diff = ("diff", "--no-renames", "--no-ext-diff", "--no-textconv", "-z")
    statuses: dict[str, str] = {}
    fields = _git.run(repo, *diff, "--name-status", since, "HEAD").decode().split("\0")
    for status, path in zip(fields[0::2], fields[1::2]):
        if status:
            statuses[path] = status[0]
    changes: list[Change] = []
    for record in _git.run(repo, *diff, "--numstat", since, "HEAD").decode().split("\0"):
        if not record:
            continue
        added, removed, path = record.split("\t", 2)
        changes.append(Change(path, statuses.get(path, "M"), _counts(added), _counts(removed)))
    return sorted(changes, key=lambda change: change.path)


def _violations_for(change: Change) -> list[Violation]:
    found: list[Violation] = []
    if is_private(change.path):
        found.append(Violation(change.path, "private_path"))
    if change.status == "A":
        found.append(Violation(change.path, "new_outside_file"))
    elif change.added is None or change.added > GLUE_LINE_CAP:
        found.append(Violation(change.path, "glue_too_big"))
    return found


def check(repo: Path, base: str) -> list[Violation]:
    """Return every boundary violation on the branch, sorted by path."""
    found: list[Violation] = []
    for change in branch_changes(repo, base):
        if is_outside(change.path):
            found.extend(_violations_for(change))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.tools.boundary")
    parser.add_argument("--base", default=DEFAULT_BASE, help="the ref the branch was cut from")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="a path inside the repo")
    args = parser.parse_args(argv)
    try:
        violations = check(_git.toplevel(args.repo), args.base)
    except _git.GitError as exc:
        print(f"boundary check failed to run: {exc}", file=sys.stderr)
        return 2
    if not violations:
        print(f"boundary clean (base {args.base})")
        return 0
    print(f"boundary: {len(violations)} violation(s) against {args.base}")
    for violation in violations:
        print(f"  {violation.reason:<17} {violation.path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
