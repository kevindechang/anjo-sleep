"""The person's ways of working with them: a versioned list of short strategy lines.

Each promotion appends one record to the person's strategy log: the night, the one atomic edit
that was made, the probe's evidence counts (wins, ties and losses), the ancestry of the text (the
models it derives from, like every other model-written text) and when it was made. The current list
is folded from those records, so a version is never rewritten in place. The list lives in the
person's sleep folder beside their evidence and transcripts, never under ``results/``: it is the
person's, and erasure removes it with the rest of their sleep data.

The list is read by the morning's reported-only ``2H`` arm and by the harness loop. It never enters
practice prompts, training rows, seed rows or dreams; those are built from evidence and transcripts
alone.

At most ``STRATEGY_MAX`` lines, each at most ``STRATEGY_WORDS`` words: a list is a few ways of
working, not a second brief. An add when the list is full is refused here as well as in the loop,
so a caller cannot grow it by forgetting to prune.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from anjo.sleep.evidence.jsonl import Codec, Guard, LogError, append_records, read_records
from anjo.sleep.harness import STRATEGY_MAX, STRATEGY_WORDS

STRATEGY_HEADING: Final = "How you work best with them:"
Action = Literal["add", "replace", "remove"]


class StrategyError(ValueError):
    """An edit that would not leave a valid strategy list."""


@dataclass(frozen=True)
class StrategyEdit:
    """One atomic edit: add a line, replace one line, or remove one line.

    ``line`` is the new line (None for a remove); ``target`` is the line it replaces or removes
    (None for an add). The exact target makes a version replayable without guessing which line the
    proposer meant.
    """

    action: Action
    line: str | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if self.action == "add":
            if self.line is None or self.target is not None:
                raise StrategyError("an add carries a line and no target")
        elif self.action == "replace":
            if self.line is None or self.target is None:
                raise StrategyError("a replace carries a line and the target it replaces")
        elif self.action == "remove":
            if self.line is not None or self.target is None:
                raise StrategyError("a remove carries a target and no line")
        else:
            raise StrategyError(f"no strategy edit is {self.action!r}")
        if self.line is not None:
            object.__setattr__(self, "line", clean_line(self.line))


def clean_line(text: str) -> str:
    """The one-line form a strategy must have: whitespace normalized, no newline, within the limit."""
    if not isinstance(text, str) or "\n" in text or "\r" in text:
        raise StrategyError("a strategy is one line of text")
    line = " ".join(text.split())
    if not line:
        raise StrategyError("a strategy line is not empty")
    if len(line.split()) > STRATEGY_WORDS:
        raise StrategyError(f"a strategy line is at most {STRATEGY_WORDS} words")
    return line


@dataclass(frozen=True)
class StrategyVersion:
    """One promotion: its number, the night, the edit, and the probe's evidence counts."""

    version: int
    night: date
    edit: StrategyEdit
    wins: int
    ties: int
    losses: int
    ancestry: frozenset[str] | None
    created_at: datetime


def edit_record(edit: StrategyEdit) -> dict[str, Any]:
    return {
        "action": edit.action,
        **({} if edit.line is None else {"line": edit.line}),
        **({} if edit.target is None else {"target": edit.target}),
    }


def edit_from_record(record: Any) -> StrategyEdit:
    if not isinstance(record, dict):
        raise LogError("a strategy edit is not a record")
    try:
        action = record["action"]
        if action not in ("add", "replace", "remove"):
            raise StrategyError("not a strategy action")
        return StrategyEdit(
            action=action,
            line=None if record.get("line") is None else str(record["line"]),
            target=None if record.get("target") is None else str(record["target"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LogError(f"not a strategy edit ({type(error).__name__})") from None


def version_record(version: StrategyVersion) -> dict[str, Any]:
    return {
        "version": version.version,
        "night": version.night.isoformat(),
        "edit": edit_record(version.edit),
        "wins": version.wins,
        "ties": version.ties,
        "losses": version.losses,
        "ancestry": None if version.ancestry is None else sorted(version.ancestry),
        "created_at": version.created_at.isoformat(),
    }


def _ancestry(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise LogError("a strategy's ancestry is a list of model names or null")
    return frozenset(value)


def version_from_record(record: dict[str, Any]) -> StrategyVersion:
    try:
        number = record["version"]
        wins, ties, losses = record["wins"], record["ties"], record["losses"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (number, wins, ties, losses)):
            raise TypeError("a count is not an int")
        if min(number, wins, ties, losses) < 0:
            raise ValueError("a count is negative")
        return StrategyVersion(
            version=number,
            night=date.fromisoformat(record["night"]),
            edit=edit_from_record(record["edit"]),
            wins=wins,
            ties=ties,
            losses=losses,
            ancestry=_ancestry(record.get("ancestry")),
            created_at=datetime.fromisoformat(record["created_at"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LogError(f"not a strategy version ({type(error).__name__})") from None


def applied(lines: Sequence[str], edit: StrategyEdit) -> tuple[str, ...]:
    """``lines`` after ``edit``; refuses an add that would grow the list past its limit, a line
    already present, and a target that is not there."""
    found = tuple(lines)
    if edit.action == "add":
        assert edit.line is not None
        if len(found) >= STRATEGY_MAX:
            raise StrategyError(f"the list is full at {STRATEGY_MAX}; an add must replace a line")
        if edit.line in found:
            raise StrategyError("the line is already in the list")
        return (*found, edit.line)
    assert edit.target is not None
    if edit.target not in found:
        raise StrategyError("the target line is not in the list")
    if edit.action == "remove":
        return tuple(line for line in found if line != edit.target)
    assert edit.line is not None
    if edit.line != edit.target and edit.line in found:
        raise StrategyError("the line is already in the list")
    return tuple(edit.line if line == edit.target else line for line in found)


def folded(versions: Sequence[StrategyVersion]) -> tuple[str, ...]:
    """The current list, folded from its versions in order."""
    lines: tuple[str, ...] = ()
    for version in versions:
        lines = applied(lines, version.edit)
    return lines


def strategy_block(context: str | None, lines: Sequence[str]) -> str | None:
    """Context C followed by the list in its marked block; with no lines it is C exactly."""
    if not lines:
        return context
    block = "\n".join((STRATEGY_HEADING, *(f"- {line}" for line in lines)))
    return block if not context else f"{context}\n\n{block}"


class StrategyStore(Protocol):
    """One person's strategy list and its versions."""

    def lines(self, until: date | None = None) -> tuple[str, ...]:
        """The list as it stood after ``until`` (every version when None), in order."""

    def versions(self, until: date | None = None) -> tuple[StrategyVersion, ...]:
        """Every promotion through ``until`` (all of them when None), oldest first."""

    def promote(
        self, night: date, edit: StrategyEdit, *, wins: int, ties: int, losses: int,
        ancestry: frozenset[str] | None, at: datetime,
    ) -> StrategyVersion:
        """Append the edit as the list's next version; refuses an edit that doesn't apply."""


class JsonlStrategyStore:
    """One person's strategy log over one file, append-only JSON Lines."""

    def __init__(self, path: Path, person_id: str, *, codec: Codec, guard: Guard) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard

    def versions(self, until: date | None = None) -> tuple[StrategyVersion, ...]:
        read = tuple(version_from_record(record) for record in read_records(self.path, self._codec))
        return read if until is None else tuple(version for version in read if version.night <= until)

    def lines(self, until: date | None = None) -> tuple[str, ...]:
        return folded(self.versions(until))

    def promote(
        self, night: date, edit: StrategyEdit, *, wins: int, ties: int, losses: int,
        ancestry: frozenset[str] | None, at: datetime,
    ) -> StrategyVersion:
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
               for count in (wins, ties, losses)):
            raise ValueError("a strategy's evidence counts are non-negative ints")
        with self._guard():
            versions = self.versions()
            line = applied(folded(versions), edit)
            version = StrategyVersion(
                version=len(versions) + 1, night=night, edit=edit, wins=wins, ties=ties, losses=losses,
                ancestry=ancestry, created_at=at,
            )
            if len(line) > STRATEGY_MAX:
                raise StrategyError(f"the list is at most {STRATEGY_MAX} lines")
            append_records(self.path, self._codec, [version_record(version)])
        return version

    def export(self) -> dict[str, object]:
        return {"strategies": [version_record(version) for version in self.versions()]}
