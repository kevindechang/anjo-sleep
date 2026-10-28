"""What the live page's full step reads on the sleep host (Task 13a): the night slider's stops and
her mind panel, for the account that plays the film student (``ANJO_SLEEP_DEMO_USER_ID``).

Every read here only reads: the night logs, the checkpoint registry, her evidence, her targets
and expectations, MIND's workspace log, the results trees, the pre-registration ledger and the
demo's own files. Nothing is written, and no model or GPU is reached. A source that can't be read
makes its section ``unavailable``; it never fails the page.

- **Only a synthetic account.** The panel and the slider show nothing for an account that isn't
  registered synthetic: no real person's state ever reaches a stranger.
- **Held-out text never leaves.** ``held_out`` builds the filter from the persona file's held-out
  part and every evidence item stored ``held_out``. If either can't be read, every section that
  carries free text is unavailable: the panel fails closed. The sealed probe files are never read.
- **The stops** are ``slider.plan_stops`` over her night logs and the registry's history, checked
  against the rendering Glimmer serves now (``framing.rendering`` at the configured strength).
- **The panel is kept** (``PanelCache``) until a file it reads changes, by modification time and
  size, or ``PANEL_TTL_S`` passes: built, it costs most of a second at full size (the held-out filter
  against every shown text), while a stat of each file costs little. A panel that failed closed is
  never kept: each request tries again, and fails closed again, cheaply.
- **The demo's files** (``<data dir>/sleep/demo/``): ``night_answers.json`` (the slider's and the
  toggle's answers, written by ``demo_pairs --nights``), ``swap_pairs.json`` (the swap
  measurement's pairs, if it ran and saved any) and ``mind.md`` (MIND's report, ``reports.mind``,
  copied there once the final run writes it).
"""
from __future__ import annotations

import json
import logging
import threading
from functools import partial
from collections.abc import Callable, Iterator
from datetime import date, datetime, time, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Final, TypeVar, cast

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.demo_live import DEMO_PERSONA, demo_dir
from anjo.sleep.app.ownership import night_logs, person_results, results_trees
from anjo.sleep.app.storage import (
    SLEEP_DIR,
    checkpoint_registry,
    data_root,
    evidence_store,
    sleep_dir,
    target_store,
    workspace_log,
)
from anjo.sleep.demo.dailies import NightFiles
from anjo.sleep.demo.heldout import HeldOutText, persona_texts
from anjo.sleep.demo.night_answers import NightAnswers
from anjo.sleep.demo.night_answers import load as load_answers
from anjo.sleep.demo.panel import MindSources, NightRecord, mind_view
from anjo.sleep.demo.record import chart_view, ledger_view, task16_view
from anjo.sleep.demo.slider import NightMark, Stop, plan_stops
from anjo.sleep.demo.toggles import SwapSet, load_swaps
from anjo.sleep.mind import expect
from anjo.sleep.night.nightlog import LOG, STORY, SUMMARY
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.prereg.ledger import LedgerError, head_path, read_entries, read_pushed_head
from anjo.sleep.probes.judge_brief import PERSONAS_DIR
from anjo.sleep.reports import results
from anjo.sleep.serving import config, framing

logger = logging.getLogger(__name__)

NIGHT_ANSWERS_FILE: Final = "night_answers.json"
SWAP_FILE: Final = "swap_pairs.json"
MIND_REPORT_FILE: Final = "mind.md"
LEDGER_FILE: Final = "ledger.jsonl"  # the pre-registration ledger, beside the budget ledger
READ_ERRORS: Final = (ValueError, OSError)  # every store's read error is a ValueError (LogError, CodecError, ...)
TEXT_SECTIONS: Final = ("questions", "targets", "wants", "choices", "expectations", "gates")
UNAVAILABLE: Final = {"state": "unavailable"}
PANEL_TTL_S: Final = 60.0  # the longest a kept panel is served, even when no file it reads changed
T = TypeVar("T")
Fingerprint = tuple[tuple[str, int, int], ...]


def _folders(root: Path) -> Iterator[Path]:
    """Each night's folder in run order: ``<date>``, then its reruns ``<date>.2``, ``<date>.3``."""
    if not root.is_dir():
        return iter(())

    def key(folder: Path) -> tuple[str, int]:
        day, _, attempt = folder.name.partition(".")
        return day, int(attempt) if attempt.isdigit() else 1

    return iter(sorted((f for f in root.iterdir() if (f / SUMMARY).is_file()), key=key))


def night_files(user: str) -> tuple[NightFiles, ...]:
    """Each night's summary, log records and story (``night.story``, when it has one), in run order."""
    out = []
    for folder in _folders(night_logs(user)):
        summary = json.loads((folder / SUMMARY).read_text(encoding="utf-8"))
        log, told = folder / LOG, folder / STORY
        steps = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line] if log.exists() else []
        story = json.loads(told.read_text(encoding="utf-8")) if told.exists() else None
        if not isinstance(summary, dict):
            raise ValueError(f"{folder.name}'s summary is not a record")
        out.append(NightFiles(summary, tuple(steps), story if isinstance(story, dict) else None))
    return tuple(out)


def night_records(user: str) -> tuple[NightRecord, ...]:
    return tuple(files.record for files in night_files(user))


def serving_rendering() -> str:
    return framing.rendering(config.reasoning_strength())


def demo_stops(user: str | None) -> tuple[Stop, ...] | None:
    """The slider's stops for the demo's account, or None when they can't be read or shown."""
    if user is None or not is_synthetic(user):
        return None
    try:
        nights = [NightMark(record.night, record.status) for record in night_records(user)]
        return plan_stops(nights, checkpoint_registry().history(user), rendering=serving_rendering())
    except READ_ERRORS as error:
        logger.error("the demo's night slider can't be read: %s", type(error).__name__)
        return None


def night_answers() -> NightAnswers | None:
    return load_answers(demo_dir() / NIGHT_ANSWERS_FILE)


def swaps(user: str | None) -> SwapSet | None:
    """The swap measurement's pairs, for the demo's synthetic account only, like every other source."""
    if user is None or not is_synthetic(user):
        return None
    return load_swaps(demo_dir() / SWAP_FILE)


# ── the panel ─────────────────────────────────────────────────────────────────


def _evidence_records(user: str) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], evidence_store(user).export()["evidence"])


def held_out(user: str) -> HeldOutText | None:
    """The filter for ``user``'s panel; None when it can't be built, so nothing with text is shown."""
    try:
        person = load_person_file(PERSONAS_DIR / f"{DEMO_PERSONA}.json")
        stored = [r["text"] for r in _evidence_records(user) if r.get("held_out") is True and isinstance(r.get("text"), str)]
    except READ_ERRORS as error:
        logger.error("the demo's held-out filter can't be built: %s", type(error).__name__)
        return None
    return HeldOutText((*persona_texts(person), *stored))


def _read(what: str, read: Callable[[], T]) -> T | None:
    try:
        return read()
    except READ_ERRORS as error:
        logger.error("the demo's mind panel can't read %s: %s", what, type(error).__name__)
        return None


def _returns(user: str) -> dict[str, tuple[datetime, ...]]:
    """When the person's own evidence came back to each target: the start of its day."""
    items = evidence_store(user).stored(date.max)
    by_target: dict[str, list[datetime]] = {}
    for item in items:
        if item.target_id is not None and expect.returning([item], item.target_id):
            by_target.setdefault(item.target_id, []).append(datetime.combine(item.day, time.min, timezone.utc))
    return {target: tuple(sorted(days)) for target, days in by_target.items()}


def _sources(user: str) -> MindSources:
    returns = _read("her evidence", lambda: _returns(user))
    return MindSources(
        nights=_read("her nights", lambda: night_records(user)),
        workspace=_read("MIND's workspace", lambda: workspace_log(user).workspace()),
        targets=_read("her targets", lambda: target_store(user).targets()),
        expectations=None if returns is None else _read("her expectations", lambda: target_store(user).expectations()),
        returns=returns or {},
    )


def _lives(published: Path, mine: Path) -> list[list[results.Night]]:
    """Every other synthetic life's nights; one whose results can't be read is left out, and logged."""
    if not published.is_dir():
        return []
    found = (_read(f"a life's results ({folder.name})", partial(results.load, folder.name, published))
             for folder in sorted(published.iterdir()) if folder.is_dir() and folder != mine)
    return [life for life in found if life]


def _chart(user: str) -> dict[str, Any]:
    """The film student's development nights beside the held-out lives' final ones."""
    published, _private = results_trees()
    mine = person_results(user)
    development = _read("her results", lambda: results.load(mine.name, mine.parent))
    if development is None:
        return {**UNAVAILABLE, "final": None}
    person = load_person_file(PERSONAS_DIR / f"{DEMO_PERSONA}.json")
    return chart_view(development, _lives(published, mine), name=person.visible.name.split()[0])


def _ledger(held: HeldOutText) -> dict[str, Any]:
    path = data_root() / SLEEP_DIR / LEDGER_FILE
    try:
        return ledger_view(read_entries(path), pushed=read_pushed_head(path), held=held)
    except (LedgerError, *READ_ERRORS) as error:
        logger.error("the demo can't read the ledger: %s", type(error).__name__)
        return {**UNAVAILABLE, "entries": []}


def _task16(held: HeldOutText) -> dict[str, Any]:
    path = demo_dir() / MIND_REPORT_FILE
    text = _read("MIND's report", lambda: path.read_text(encoding="utf-8") if path.exists() else None)
    if text is None and path.exists():
        return {**UNAVAILABLE, "lines": []}
    return task16_view(text, held)


def _closed() -> dict[str, Any]:
    """The panel with every section unavailable: what a non-synthetic account or a failed filter gets."""
    view = mind_view(MindSources(), HeldOutText(()))
    return {**view, "chart": {**UNAVAILABLE, "final": None}, "task16": {**UNAVAILABLE, "lines": []},
            "ledger": {**UNAVAILABLE, "entries": []}}


def _built(user: str, held: HeldOutText) -> dict[str, Any]:
    try:
        chart = _chart(user)
    except READ_ERRORS as error:
        logger.error("the demo's results chart can't be read: %s", type(error).__name__)
        chart = {**UNAVAILABLE, "final": None}
    return {**mind_view(_sources(user), held), "chart": chart, "task16": _task16(held), "ledger": _ledger(held)}


def mind_payload(user: str | None) -> dict[str, Any]:
    """Her mind panel, filtered. Every section with free text is unavailable when the account isn't
    synthetic or the held-out filter can't be built."""
    held = None if user is None or not is_synthetic(user) else held_out(user)
    return _closed() if user is None or held is None else _built(user, held)


def _files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
    elif root.is_dir():
        yield from (path for path in root.rglob("*") if path.is_file())


def _fingerprint(user: str) -> Fingerprint:
    """Every file the panel reads, by path, modification time and size: her own folder, the
    published results (her nights and every other life's), the persona file, the ledger and its
    pushed head, and MIND's report."""
    published, _private = results_trees()
    ledger = data_root() / SLEEP_DIR / LEDGER_FILE
    roots = (sleep_dir(user), published, PERSONAS_DIR / f"{DEMO_PERSONA}.json", ledger, head_path(ledger),
             demo_dir() / MIND_REPORT_FILE)
    found = []
    for root in roots:
        for path in _files(root):
            stat = path.stat()
            found.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(found))


class PanelCache:
    """``mind_payload`` for one account, kept until a file it reads changes or ``PANEL_TTL_S``
    passes. One build at a time: requests that arrive during a build wait for it, then share it."""

    def __init__(self, user: str | None, *, clock: Callable[[], float] = monotonic) -> None:
        self._user = user
        self._clock = clock
        self._lock = threading.Lock()
        self._kept: tuple[Fingerprint, float, dict[str, Any]] | None = None

    def payload(self) -> dict[str, Any]:
        user = self._user
        if user is None or not is_synthetic(user):
            return _closed()
        with self._lock:
            key = _read("the panel's files", lambda: _fingerprint(user))
            kept = self._kept
            if kept is not None and kept[0] == key and self._clock() - kept[1] < PANEL_TTL_S:
                return kept[2]
            self._kept = None
            held = held_out(user)
            if held is None:
                return _closed()  # never kept: the next request tries the filter again
            built = _built(user, held)
            if key is not None:
                self._kept = (key, self._clock(), built)
            return built
