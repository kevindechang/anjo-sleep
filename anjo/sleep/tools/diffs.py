"""Added lines of one file's branch diff, shared by the boundary check and the integration index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from anjo.sleep.tools import _git

_HUNK_HEADER = re.compile(rb"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DIFF = ("diff", "-U0", "--no-renames", "--no-ext-diff", "--no-textconv", "--no-color")


@dataclass(frozen=True)
class Hunk:
    start: int  # line number of the first added line in HEAD's version, 1-based
    added: tuple[bytes, ...]  # the added lines as raw bytes, without newlines


def added_hunks(repo: Path, since: str, path: str) -> list[Hunk]:
    """Each hunk of ``since..HEAD`` for ``path`` that adds lines; removed text is dropped."""
    raw = _git.run(repo, *_DIFF, since, "HEAD", "--", path)
    hunks: list[Hunk] = []
    lines: list[bytes] = []
    start, in_hunk = 0, False
    for line in raw.split(b"\n"):
        header = _HUNK_HEADER.match(line)
        if header:
            if lines:
                hunks.append(Hunk(start, tuple(lines)))
            start, lines, in_hunk = int(header.group(1)), [], True
        elif in_hunk and line.startswith(b"+"):
            lines.append(line[1:])
    if lines:
        hunks.append(Hunk(start, tuple(lines)))
    return hunks
