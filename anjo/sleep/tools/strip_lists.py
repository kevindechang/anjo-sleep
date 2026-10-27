"""The two strip lists that fix a leak in branch history without rewriting the branch (§12.4 step 5).

- ``ops/sleep/strip_blobs.txt`` (tracked): blob ids to drop, one 40-hex id per line.
- ``ops/sleep/replace_messages.txt`` (untracked): commit-message text to rewrite, in
  ``git filter-repo --replace-message`` syntax: ``text``, ``text==>replacement``,
  ``literal:text`` or ``regex:pattern`` (``==>replacement`` optional; the default replacement
  is ``***REMOVED***``). ``glob:`` lines are refused so the Python copy of the rules below
  matches filter-repo's exactly.

Both files accept blank lines and ``#`` comments; the cleaned copies handed to filter-repo
carry neither. A digest of the cleaned content tells the publish script whether a list
changed since the last publish, which is the one case that allows a force-push.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPLACEMENT = b"***REMOVED***"
_BLOB_ID = re.compile(r"^[0-9a-f]{40}$")


class StripListError(ValueError):
    """A strip list holds a line the pipeline cannot apply exactly."""


@dataclass(frozen=True)
class Replacement:
    pattern: bytes
    replacement: bytes
    is_regex: bool


def _clean_lines(path: Path) -> list[bytes]:
    if not path.exists():
        return []
    lines = (line.rstrip(b"\r\n") for line in path.read_bytes().splitlines())
    return [line for line in lines if line.strip() and not line.lstrip().startswith(b"#")]


def load_strip_blobs(path: Path) -> list[str]:
    ids: list[str] = []
    for number, line in enumerate(_clean_lines(path), 1):
        blob = line.decode("ascii", "replace").strip().lower()
        if not _BLOB_ID.match(blob):
            raise StripListError(f"{path.name}: entry {number} is not a 40-character blob id")
        ids.append(blob)
    return sorted(set(ids))


def load_replacements(path: Path) -> list[Replacement]:
    rules: list[Replacement] = []
    for number, line in enumerate(_clean_lines(path), 1):
        replacement = DEFAULT_REPLACEMENT
        if b"==>" in line:
            line, replacement = line.rsplit(b"==>", 1)
        if line.startswith(b"glob:"):
            raise StripListError(f"{path.name}: entry {number} uses glob:, use regex: instead")
        if line.startswith(b"regex:"):
            rules.append(Replacement(line[len(b"regex:"):], replacement, True))
            continue
        if line.startswith(b"literal:"):
            line = line[len(b"literal:"):]
        if line:
            rules.append(Replacement(line, replacement, False))
    return rules


def apply_replacements(message: bytes, rules: list[Replacement]) -> bytes:
    """Rewrite a message the way filter-repo does: every literal first, then every regex."""
    for rule in rules:
        if not rule.is_regex:
            message = message.replace(rule.pattern, rule.replacement)
    for rule in rules:
        if rule.is_regex:
            message = re.sub(rule.pattern, rule.replacement, message)
    return message


def write_strip_blobs(ids: list[str], dest: Path) -> None:
    dest.write_text("".join(f"{blob}\n" for blob in ids))


def write_replacements(rules: list[Replacement], dest: Path) -> None:
    lines = []
    for rule in rules:
        prefix = b"regex:" if rule.is_regex else b"literal:"
        lines.append(prefix + rule.pattern + b"==>" + rule.replacement + b"\n")
    dest.write_bytes(b"".join(lines))


def digest(blob_ids: list[str], rules: list[Replacement]) -> dict[str, str]:
    """Hashes of both cleaned lists, stored with each publish to detect a later change."""
    blobs = "".join(f"{blob}\n" for blob in blob_ids).encode()
    replacements = b"".join(
        (b"r" if r.is_regex else b"l") + r.pattern + b"\0" + r.replacement + b"\n" for r in rules
    )
    return {
        "strip_blobs": hashlib.sha256(blobs).hexdigest(),
        "replace_messages": hashlib.sha256(replacements).hexdigest(),
    }
