"""The publish scan and the commit check (spec §12.4 steps 4 and 6, RF4).

A finding names what matched without repeating it: an env key name (never its value), a
denylist term by its number (never the term), a gitleaks rule id (the secret redacted).
Paths that contain a denylist term are withheld too.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anjo.sleep.tools import _git
from anjo.sleep.tools.export_results import MAX_FILE_BYTES, REFUSED_EXTENSIONS
from anjo.sleep.tools.strip_lists import Replacement, apply_replacements

FindingKind = Literal[
    "gitleaks", "env_value", "deny_term", "too_large", "refused_extension",
    "commit_format", "identity",
]

MIN_ENV_VALUE_LEN = 8  # shorter values (flags, ports, modes) are not secrets
COMMIT_TYPES = ("feat", "fix", "refactor", "docs", "test", "chore", "perf", "ci", "merge")
_SUBJECT = re.compile(rf"^(?:{'|'.join(COMMIT_TYPES)})(?:\([^()\s]+\))?!?: \S")
# git's own merge subjects carry only branch names, so they pass too.
_GIT_MERGE = re.compile(r"^Merge (?:remote-tracking )?branch '[\w./-]+'(?: into [\w./-]+)?$")
_SUBMODULE_MODE = "160000"


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    detail: str
    where: str

    def render(self) -> str:
        return f"{self.kind}: {self.detail} ({self.where})"


@dataclass(frozen=True)
class EnvValues:
    """The env file's values worth searching for, keyed by name. Values are never printed.

    ``skipped_short`` and ``skipped_by_flag`` name the keys left out, for the report.
    """

    values: Mapping[str, bytes]
    skipped_short: tuple[str, ...] = ()
    skipped_by_flag: tuple[str, ...] = ()


def load_env_values(path: Path, skip_keys: frozenset[str]) -> EnvValues:
    values: dict[str, bytes] = {}
    short: list[str] = []
    flagged: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith("#"):
            continue
        if key in skip_keys:
            flagged.append(key)
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            # An unquoted value ends at an inline comment; the shorter prefix only adds hits.
            value = value.split(" #", 1)[0].rstrip()
        if len(value) < MIN_ENV_VALUE_LEN:
            short.append(key)
            continue
        values.setdefault(key, value.encode())
    return EnvValues(values, tuple(dict.fromkeys(short)), tuple(dict.fromkeys(flagged)))


def load_terms(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


class _Matcher:
    def __init__(self, env: EnvValues, terms: Sequence[str]) -> None:
        self._env = env.values
        self._terms = [t.lower().encode() for t in terms]

    def hits(self, content: bytes) -> list[str]:
        found = [f"the value of {key}" for key, value in self._env.items() if value in content]
        lowered = content.lower()
        found += [f"denylist term {n}" for n, t in enumerate(self._terms, 1) if t in lowered]
        return found

    def where(self, path: str, commit: str) -> str:
        if any(t in path.lower().encode() for t in self._terms):
            path = "(path withheld)"
        return f"{path}@{commit[:12]}"


def read_objects(repo: Path, ids: Iterable[str]) -> dict[str, bytes]:
    """Object contents by id, read in one ``git cat-file --batch`` call."""
    wanted = list(dict.fromkeys(ids))
    if not wanted:
        return {}
    raw = _git.run(repo, "cat-file", "--batch", input_bytes="".join(f"{i}\n" for i in wanted).encode())
    out: dict[str, bytes] = {}
    pos = 0
    for object_id in wanted:
        end = raw.index(b"\n", pos)
        header = raw[pos:end].decode().split()
        size = int(header[2])
        out[object_id] = raw[end + 1:end + 1 + size]
        pos = end + 1 + size + 1
    return out


def _history_entries(repo: Path) -> list[tuple[str, str, str | None]]:
    """(commit, path, blob or None) for every change in every commit of every ref."""
    raw = _git.run(
        repo, "log", "--all", "-m", "--root", "--raw", "--no-abbrev", "--no-renames", "-z",
        "--format=%x01%H",
    ).decode("utf-8", "replace")
    entries: list[tuple[str, str, str | None]] = []
    for record in raw.split("\x01"):
        fields = record.split("\0")
        commit = fields[0].strip()
        for meta, path in zip(fields[1::2], fields[2::2]):
            parts = meta.strip().lstrip(":").split()
            if len(parts) < 5:
                continue
            new_mode, blob, status = parts[1], parts[3], parts[4]
            keep = status[0] != "D" and new_mode != _SUBMODULE_MODE
            entries.append((commit, path, blob if keep else None))
    return entries


def _scan_files(repo: Path, matcher: _Matcher) -> list[Finding]:
    findings: list[Finding] = []
    first_seen: dict[str, tuple[str, str]] = {}
    path_commit: dict[str, str] = {}
    for commit, path, blob in _history_entries(repo):
        path_commit.setdefault(path, commit)
        if blob is not None:
            first_seen.setdefault(blob, (commit, path))
    for path, commit in sorted(path_commit.items()):
        where = matcher.where(path, commit)
        if path.lower().endswith(REFUSED_EXTENSIONS):
            findings.append(Finding("refused_extension", "refused extension", where))
        findings += [Finding("deny_term", f"{h} in a file path", where) for h in matcher.hits(path.encode())]
    sizes = _git.blob_sizes(repo, list(first_seen))
    small = [blob for blob in first_seen if sizes[blob] <= MAX_FILE_BYTES]
    for blob in first_seen:
        if sizes[blob] > MAX_FILE_BYTES:
            commit, path = first_seen[blob]
            findings.append(Finding("too_large", f"over 5 MB ({sizes[blob]} bytes)", matcher.where(path, commit)))
    for blob, content in read_objects(repo, small).items():
        commit, path = first_seen[blob]
        findings += [Finding(_kind(h), h, matcher.where(path, commit)) for h in matcher.hits(content)]
    return findings


def _kind(hit: str) -> FindingKind:
    return "env_value" if hit.startswith("the value of") else "deny_term"


def _identity_findings(commit: str, raw: bytes, expected_email: str) -> list[Finding]:
    headers = raw.split(b"\n\n", 1)[0].decode("utf-8", "replace")
    found: list[Finding] = []
    for line in headers.splitlines():
        role = line.split(" ", 1)[0]
        if role in ("author", "committer") and f"<{expected_email}>" not in line:
            found.append(Finding("identity", f"{role} email is not the noreply address", commit[:12]))
    return found


def _scan_commits(repo: Path, matcher: _Matcher, expected_email: str | None) -> list[Finding]:
    findings: list[Finding] = []
    commits = _git.text(repo, "rev-list", "--all").split()
    for commit, raw in read_objects(repo, commits).items():
        findings += [Finding(_kind(h), f"{h} in a commit", commit[:12]) for h in matcher.hits(raw)]
        if expected_email:
            findings += _identity_findings(commit, raw, expected_email)
    return findings


def _gitleaks_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GITLEAKS_CONFIG")}


def run_gitleaks(source: Path, *, no_git: bool = False) -> list[Finding]:
    """``gitleaks detect --redact`` over all history (or a plain folder with ``no_git``)."""
    with tempfile.TemporaryDirectory(prefix="anjo-sleep-gitleaks-") as tmp:
        report = Path(tmp) / "report.json"
        ignore_dir = Path(tmp) / "no-ignores"
        ignore_dir.mkdir()
        cmd = [
            "gitleaks", "detect", "--source", str(source), "--redact", "--no-banner",
            "--log-level", "error", "--exit-code", "1", "--report-format", "json",
            "--report-path", str(report), "--gitleaks-ignore-path", str(ignore_dir),
            "--ignore-gitleaks-allow",
        ]
        if no_git:
            cmd.append("--no-git")
        done = subprocess.run(cmd, capture_output=True, env=_gitleaks_env(), check=False)
        if done.returncode not in (0, 1) or not report.exists():
            stderr = done.stderr.decode("utf-8", "replace").strip()[-500:]
            raise RuntimeError(f"gitleaks failed to run (exit {done.returncode}): {stderr}")
        results = json.loads(report.read_text() or "[]") or []
    return [
        Finding("gitleaks", f"gitleaks rule {r.get('RuleID', '?')}",
                f"{r.get('File', '?')}@{str(r.get('Commit', ''))[:12] or 'worktree'}")
        for r in results
    ]


def scan_history(
    repo: Path, env: EnvValues, terms: Sequence[str], *, expected_email: str | None = None
) -> list[Finding]:
    """Scan every blob, path and commit reachable from any ref, then run gitleaks."""
    matcher = _Matcher(env, terms)
    return _scan_files(repo, matcher) + _scan_commits(repo, matcher, expected_email) + run_gitleaks(repo)


def scan_folder(folder: Path, env: EnvValues, terms: Sequence[str]) -> list[Finding]:
    """The same content checks on plain files, run before anything new is committed."""
    matcher = _Matcher(env, terms)
    findings: list[Finding] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        where = matcher.where(rel, "uncommitted")
        if rel.lower().endswith(REFUSED_EXTENSIONS):
            findings.append(Finding("refused_extension", "refused extension", where))
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(Finding("too_large", "over 5 MB", where))
            continue
        findings += [Finding(_kind(h), h, where) for h in matcher.hits(rel.encode() + b"\n" + path.read_bytes())]
    return findings + run_gitleaks(folder, no_git=True)


def check_commits(
    repo: Path, since: str, head: str, terms: Sequence[str], rules: list[Replacement]
) -> list[Finding]:
    """Every commit in ``since..head``: a ``<type>: ...`` subject and no denylist term.

    Messages are checked after the replace-message rules, as they will be published.
    """
    matcher = _Matcher(EnvValues({}), terms)
    findings: list[Finding] = []
    commits = _git.text(repo, "rev-list", f"{since}..{head}").split()
    for commit, raw in read_objects(repo, commits).items():
        message = apply_replacements(raw.split(b"\n\n", 1)[-1], rules)
        subject = message.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        if not (_SUBJECT.match(subject) or _GIT_MERGE.match(subject)):
            findings.append(Finding("commit_format", "commit format: the subject is not `<type>: ...`", commit[:12]))
        findings += [Finding("deny_term", f"{h} in the commit message", commit[:12]) for h in matcher.hits(message)]
    return findings
