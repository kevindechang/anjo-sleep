"""The freeze manifest: every file the final test reads, hashed once at the freeze.

    python -m anjo.sleep.prereg.freeze_manifest --sealed <sealed dir>/<life>.jsonl [--sealed ...]

Run it after the final lives' probes are sealed and before their first night, from the checkout of
the freeze tag (``--tag``, default ``sleep-freeze``: HEAD must be that commit, with no uncommitted
change to a frozen file). It writes ``$ANJO_DATA_DIR/sleep/freeze_manifest.json``, never into the
repo, so HEAD stays the freeze commit: the sha256 of ``thresholds.json``, ``judge_v1.md``,
``reports/final.py`` and each sealed probe file the final reads, and that commit. A manifest already
there is never overwritten. Its sha256 goes into the prereg ledger (kind ``freeze_manifest``).
``reports.final.run`` reads it from there and refuses to score when it is missing, when its sha256
is not the ledger's one ``freeze_manifest`` entry, when its commit is not both the freeze tag's and
the HEAD the final runs at, or when any file differs. Only local files, the ledger and ``git``.

``registered_comparable`` is the one comparison of ``thresholds.json`` with its registered copy
(``tests/frozen/thresholds_registered.json``): the keys added after registration are left out of
both, and anything else that differs is drift. The thresholds test and the final both use it.
"""
from __future__ import annotations

import argparse
import copy
import os
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from anjo.sleep.prereg import ledger

SLEEP: Final = Path(__file__).resolve().parents[1]
REPO: Final = SLEEP.parents[1]
MANIFEST_NAME: Final = "freeze_manifest.json"  # under $ANJO_DATA_DIR/sleep, beside the ledger
LEDGER_KIND: Final = "freeze_manifest"
FREEZE_TAG: Final = "sleep-freeze"
THRESHOLDS: Final = SLEEP / "prereg" / "thresholds.json"
REGISTERED_THRESHOLDS: Final = SLEEP / "tests" / "frozen" / "thresholds_registered.json"
# Paths under anjo/sleep the final reads (besides the sealed probes), as the manifest names them.
FROZEN_FILES: Final = ("prereg/thresholds.json", "probes/prompts/judge_v1.md", "reports/final.py")
GIT_TIMEOUT_S: Final = 30.0


class ManifestError(ValueError):
    """The freeze manifest is missing, unreadable, or does not match the files."""


def manifest_path() -> Path:
    """``$ANJO_DATA_DIR/sleep/freeze_manifest.json``: outside the repo, so HEAD stays the freeze."""
    base = os.environ.get(ledger.DATA_DIR_ENV, "").strip()
    if not base:
        raise ManifestError(f"{ledger.DATA_DIR_ENV} is not set; the manifest lives in the data dir")
    return Path(base) / "sleep" / MANIFEST_NAME


def registered_comparable(now: Mapping[str, Any], frozen: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``now`` (``thresholds.json``) and ``frozen`` (its registered copy) without the keys allowed to
    differ: the development keys and ``hand_written`` in both, and Task 17b's dated placeholder
    ``replay_token_penalty`` in ``now``. Equal copies mean the final-test rule has not drifted. A key
    the comparison expects but cannot find raises ``KeyError``."""
    found, registered = copy.deepcopy(dict(now)), copy.deepcopy(dict(frozen))
    for obj in (registered, found):
        obj["development"].pop("pooled_lives")
        obj["donor_rule"].pop("dev_lives")
        obj.pop("hand_written")
    found.pop("replay_token_penalty")  # Task 17b's λ, added after registration as a dated placeholder
    found["placeholders"].pop("replay_token_penalty")
    return found, registered


def thresholds_drift(thresholds: Path = THRESHOLDS, registered: Path = REGISTERED_THRESHOLDS) -> str | None:
    """Why ``thresholds`` is not the registered rule, or None when only the allowed keys differ."""
    try:
        now = json.loads(thresholds.read_text(encoding="utf-8"))
        frozen = json.loads(registered.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"{thresholds.name} or its registered copy can't be read ({type(error).__name__})"
    try:
        found, expected = registered_comparable(now, frozen)
    except (KeyError, TypeError, AttributeError) as error:
        return f"{thresholds.name} lacks a key the registered comparison expects ({error})"
    if found != expected:
        changed = sorted(k for k in {*found, *expected} if found.get(k) != expected.get(k))
        return f"{thresholds.name} differs from {registered.name} beyond the allowed keys: {', '.join(changed)}"
    return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_S, check=False)
    if done.returncode != 0:
        raise ManifestError(f"git {' '.join(args)} failed in {repo}")
    return done.stdout


def head_commit(repo: Path = REPO, *, tag: str = FREEZE_TAG) -> str:
    """HEAD's sha, once it is the commit ``tag`` names; refused when a frozen file has uncommitted
    changes (its hash would name no commit)."""
    sha = _git(repo, "rev-parse", "HEAD").strip()
    if len(sha) != 40:
        raise ManifestError("this checkout's git sha can't be read")
    if _git(repo, "rev-parse", f"{tag}^{{commit}}").strip() != sha:
        raise ManifestError(f"HEAD is not the commit {tag!r} names; write the manifest from the freeze checkout")
    dirty = _git(repo, "status", "--porcelain", "--", *(str(SLEEP / rel) for rel in FROZEN_FILES)).strip()
    if dirty:
        raise ManifestError(f"uncommitted changes to a frozen file: {dirty}")
    return sha


def build(sealed: Sequence[Path], *, commit: str, root: Path = SLEEP) -> dict[str, Any]:
    """The manifest's record: each frozen file's and each sealed probe file's sha256, and ``commit``."""
    if not sealed:
        raise ManifestError("the manifest names every sealed probe file the final reads: pass --sealed")
    names = [path.name for path in sealed]
    if len(set(names)) != len(names):
        raise ManifestError("two sealed probe files share a name")
    missing = [str(p) for p in (*(root / rel for rel in FROZEN_FILES), *sealed) if not p.is_file()]
    if missing:
        raise ManifestError(f"missing: {', '.join(missing)}")
    return {
        "git_commit": commit,
        "files": {rel: sha256_file(root / rel) for rel in FROZEN_FILES},
        "sealed": {path.name: sha256_file(path) for path in sealed},
    }


def _registered(ledger_file: Path | None) -> list[str]:
    """The sha256 of every ``freeze_manifest`` entry, once the ledger's chain verifies."""
    try:
        entries = ledger.read_entries(ledger_file)
        ledger.verify_entries(entries)
    except ledger.LedgerError as error:
        raise ManifestError(f"the prereg ledger can't be used: {error}") from None
    return [entry.sha256 for entry in entries if entry.kind == LEDGER_KIND]


def write(record: Mapping[str, Any], out: Path, *, ledger_file: Path | None = None) -> Path:
    """Write the manifest once and register its sha256 in the ledger; one already there, or one
    already registered, is never replaced."""
    if out.exists():
        raise ManifestError(f"{out} already exists; the freeze's manifest stands")
    if _registered(ledger_file):
        raise ManifestError("the ledger already registers a freeze manifest; the freeze's manifest stands")
    body = (json.dumps(dict(record), sort_keys=True, indent=1) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = out.with_name(f".{out.name}.tmp")
    pending.write_bytes(body)
    pending.replace(out)
    try:
        ledger.register(LEDGER_KIND, ledger.sha256_bytes(body), f"freeze manifest at {record['git_commit']}",
                        path=ledger_file)
    except ledger.LedgerError as error:
        raise ManifestError(f"{out} was written but not registered: {error}") from None
    return out


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"no freeze manifest at {path}; write it at the freeze "
                            "(python -m anjo.sleep.prereg.freeze_manifest --sealed ...)")
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ManifestError(f"{path.name} can't be read ({type(error).__name__})") from None
    if not (isinstance(found, dict) and isinstance(found.get("files"), dict)
            and isinstance(found.get("sealed"), dict) and isinstance(found.get("git_commit"), str)):
        raise ManifestError(f"{path.name} lacks its files, sealed probes or git commit")
    return found


def check(path: Path, sealed: Sequence[Path], *, commit: str, ledger_file: Path | None = None,
          root: Path = SLEEP) -> dict[str, Any]:
    """The manifest at ``path``, once its sha256 is the ledger's one ``freeze_manifest`` entry, its
    commit is ``commit`` (the freeze tag's, which the final runs at) and every frozen file and each
    of ``sealed`` matches it."""
    found = read(path)
    registered = set(_registered(ledger_file))
    if len(registered) != 1:
        raise ManifestError(f"the ledger registers {len(registered)} freeze manifests, not one")
    if sha256_file(path) not in registered:
        raise ManifestError(f"{path.name} is not the freeze manifest the ledger registers")
    if found["git_commit"] != commit:
        raise ManifestError(f"{path.name} was written at {found['git_commit']!r}, not the freeze commit {commit!r}")
    problems: list[str] = []
    for rel in FROZEN_FILES:
        want, file = found["files"].get(rel), root / rel
        if want is None:
            problems.append(f"{rel} is not in the manifest")
        elif not file.is_file() or sha256_file(file) != want:
            problems.append(f"{rel} differs from the manifest")
    for file in sealed:
        want = found["sealed"].get(file.name)
        if want is None:
            problems.append(f"sealed probe file {file.name} is not in the manifest")
        elif not file.is_file() or sha256_file(file) != want:
            problems.append(f"sealed probe file {file.name} differs from the manifest")
    if problems:
        raise ManifestError(f"{path.name}: " + "; ".join(problems))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the freeze manifest the final test checks.")
    parser.add_argument("--sealed", action="append", default=[], type=Path, required=True,
                        help="a sealed probe file the final reads; repeat for each")
    parser.add_argument("--tag", default=FREEZE_TAG, help=f"the freeze tag HEAD must be (default: {FREEZE_TAG})")
    parser.add_argument("--out", type=Path, default=None, help=f"(default: $ANJO_DATA_DIR/sleep/{MANIFEST_NAME})")
    args = parser.parse_args(argv)
    try:
        drift = thresholds_drift()
        if drift is not None:
            raise ManifestError(drift)
        out = write(build(args.sealed, commit=head_commit(tag=args.tag)), args.out or manifest_path())
    except ManifestError as error:
        print(f"freeze manifest: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"manifest": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
