"""The boundary check: hackathon code stays in the published folders (spec §12.1, §12.4 step 1).

It reads the branch's own changes, ``merge-base(base, HEAD)..HEAD``, and reports:

- ``new_outside_file``: a file added outside the published folders
- ``not_modified_in_place``: an outside file deleted, turned into a symlink or otherwise
  changed in type; outside files may only be modified in place
- ``glue_too_big``: an outside file whose branch diff adds more than 20 lines, any line over
  300 bytes, or more than 6 KB in all (or is binary)
- ``private_path``: any change, deletion included, under a private path (auth, admin,
  billing, deploy, and the crisis, output-guard and age-lock modules)
- ``moved_private_code``: a published file whose content is byte-identical to a file in
  the base (moved or copied in), or a deleted outside file whose content now sits in a
  published folder. Files under 64 bytes (empty ``__init__.py`` and the like) and stock
  license texts (base files named LICENSE, LICENCE or COPYING) are not matched.

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

from anjo.sleep.tools import _git, diffs

Reason = Literal[
    "new_outside_file", "not_modified_in_place", "glue_too_big", "private_path",
    "moved_private_code",
]

PUBLISHED_FOLDERS: tuple[str, ...] = ("anjo/sleep/", "mobile/features/sleep/")
EXEMPT_FOLDERS: tuple[str, ...] = ("docs/", "ops/sleep/")
GLUE_LINE_CAP = 20
GLUE_LINE_BYTE_CAP = 300
GLUE_FILE_BYTE_CAP = 6 * 1024
MIN_MOVED_BLOB_BYTES = 64
DEFAULT_BASE = "origin/main"
_NULL_BLOB = "0" * 40
_LICENSE_NAMES = ("LICENSE", "LICENCE", "COPYING")

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
    # deploy, release and production operations
    ".github/workflows/*",
    "deploy.sh",
    "sync-to-public.sh",
    "scripts/safe_deploy_prod.sh",
    "scripts/post_deploy_smoke.py",
    "scripts/backup_db.sh",
    "scripts/env_flip.sh",
    "scripts/server_setup.sh",
    "scripts/release_manager.sh",
    "scripts/restore_full_state.sh",
    "scripts/preflight.sh",
    "scripts/pre_submission_check.sh",
    "scripts/build_release_bundle.py",
    "scripts/ios_internal_build.sh",
    "mobile/ios/ci_scripts/*",
    "mobile/ios/bump_build.sh",
    "mobile/ios/phonon/upload_models.sh",
    "ops/monitoring/*",
    "anjo/dashboard/deploy_state.py",
    "anjo/core/release_identity.py",
    "tests/test_*deploy*",
    # crisis
    "anjo/core/crisis*",
    "anjo/evals/crisis*",
    "anjo/floor_data/crisis/*",
    "mobile/lib/crisis/*",
    "mobile/lib/crisis-resources*",
    "mobile/assets/crisis_resources.json",
    "mobile/app/(app)/safety.tsx",
    "scripts/*crisis*",
    "tests/test_*crisis*",
    # output guard, the server floor and the companion-safety checks
    "anjo/core/floor.py",
    "anjo/core/output_safety.py",
    "anjo/core/output_guard_rules.py",
    "anjo/core/medical_boundary.py",
    "anjo/core/safety.py",
    "anjo/floor_data/*",
    "anjo/evals/output_guard*",
    "anjo/evals/local_floor*",
    "anjo/evals/launch_safety*",
    "anjo/evals/medical_gate*",
    "mobile/lib/floor/*",
    "mobile/lib/device-floor*",
    "scripts/export_output_guards.py",
    "scripts/floor_guard_eval.py",
    "tests/test_output_safety*",
    "tests/test_floor_data_parity*",
    "tests/test_medical_boundary*",
    "tests/test_medical_gate*",
    "tests/test_local_floor*",
    "tests/test_launch_safety*",
    "tests/test_safety_adversarial*",
    "tests/test_companion_safety*",
    # the offline test guard
    "tests/network_guard.py",
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
    """One file the branch changed: status letter, blob ids and line counts (None if binary)."""

    path: str
    status: str
    old_blob: str
    new_blob: str
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
    raw: dict[str, tuple[str, str, str]] = {}
    fields = _git.run(repo, *diff, "--raw", "--no-abbrev", since, "HEAD").decode().split("\0")
    for meta, path in zip(fields[0::2], fields[1::2]):
        parts = meta.lstrip(":").split()
        if len(parts) == 5:
            raw[path] = (parts[4][0], parts[2], parts[3])
    changes: list[Change] = []
    for record in _git.run(repo, *diff, "--numstat", since, "HEAD").decode().split("\0"):
        if not record:
            continue
        added, removed, path = record.split("\t", 2)
        status, old, new = raw.get(path, ("M", _NULL_BLOB, _NULL_BLOB))
        changes.append(Change(path, status, old, new, _counts(added), _counts(removed)))
    return sorted(changes, key=lambda change: change.path)


def _glue_too_big(repo: Path, since: str, change: Change) -> bool:
    if change.added is None or change.added > GLUE_LINE_CAP:
        return True
    lines = [line for hunk in diffs.added_hunks(repo, since, change.path) for line in hunk.added]
    if any(len(line) > GLUE_LINE_BYTE_CAP for line in lines):
        return True
    return sum(len(line) + 1 for line in lines) > GLUE_FILE_BYTE_CAP


def _outside_violations(repo: Path, since: str, change: Change) -> list[Violation]:
    found: list[Violation] = []
    if is_private(change.path):
        found.append(Violation(change.path, "private_path"))
    if change.status == "A":
        found.append(Violation(change.path, "new_outside_file"))
    elif change.status != "M":
        found.append(Violation(change.path, "not_modified_in_place"))
    elif _glue_too_big(repo, since, change):
        found.append(Violation(change.path, "glue_too_big"))
    return found


def _tree_entries(repo: Path, ref: str, *paths: str) -> list[tuple[str, str, int]]:
    """(blob id, path, size) for every regular file in ``ref``'s tree (optionally limited)."""
    raw = _git.run(repo, "ls-tree", "-r", "-l", "-z", ref, "--", *paths).decode("utf-8", "replace")
    entries: list[tuple[str, str, int]] = []
    for record in raw.split("\0"):
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) == 4 and parts[1] == "blob" and parts[0] != "120000":
            entries.append((parts[2], path, int(parts[3])))
    return entries


def _is_license_text(path: str) -> bool:
    """Stock license texts are public by definition, so a copy of one is not private code."""
    return path.rsplit("/", 1)[-1].upper().startswith(_LICENSE_NAMES)


def _base_blobs(repo: Path, base: str, since: str) -> set[str]:
    """Blob ids of every unpublished file in the branch point's tree and in the base's tip."""
    refs = dict.fromkeys(_git.text(repo, "rev-parse", ref) for ref in (since, base))
    found: set[str] = set()
    for ref in refs:
        for blob, path, _ in _tree_entries(repo, ref):
            if not is_published(path) and not _is_license_text(path):
                found.add(blob)
    return found


def _moved_code(repo: Path, base: str, since: str, changes: list[Change]) -> list[Violation]:
    published = _tree_entries(repo, "HEAD", *PUBLISHED_FOLDERS)
    big = {blob for blob, _, size in published if size >= MIN_MOVED_BLOB_BYTES}
    if not big:
        return []
    added = [c for c in changes if is_published(c.path) and c.status != "D" and c.new_blob in big]
    gone = [c for c in changes if is_outside(c.path) and c.status not in ("A", "M")]
    found = [Violation(c.path, "moved_private_code") for c in gone if c.old_blob in big]
    if added:
        base_blobs = _base_blobs(repo, base, since)
        found += [Violation(c.path, "moved_private_code") for c in added if c.new_blob in base_blobs]
    return found


def check(repo: Path, base: str) -> list[Violation]:
    """Return every boundary violation on the branch, sorted by path and reason."""
    since = _git.merge_base(repo, base)
    changes = branch_changes(repo, since)
    found: list[Violation] = []
    for change in changes:
        if is_outside(change.path):
            found.extend(_outside_violations(repo, since, change))
    found.extend(_moved_code(repo, base, since, changes))
    return sorted(set(found), key=lambda v: (v.path, v.reason))


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
        print(f"  {violation.reason:<21} {violation.path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
