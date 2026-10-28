"""What Dailies, the live page's morning card, reads on the sleep host, for the account that plays
the film student (``ANJO_SLEEP_DEMO_USER_ID``).

Every read here only reads: her night logs, her evidence, targets and MIND's workspace log, the
checkpoint registry, the budget ledger, the day phase's markers and the demo's line file
(``daily_lines.json``, written by ``demo_pairs.after_night`` after each night that promoted new
weights). Nothing is written, and no model or GPU is reached. A source that can't be read makes
its section ``unavailable``; it never fails the page.

- **Only a synthetic account**, and only behind a held-out filter that could be built
  (``demo_mind.held_out``): otherwise every section is unavailable, as in the mind panel.
- **The reply check** is Anjo's own crisis check on each full reply (``delivery.passes_reply_check``,
  as for the night's notes). A pre-generated line it trips is withheld, never replaced: nobody
  asked it, so no referral is shown or counted.
- **A malformed night** fails where it is read (``NightFiles`` parses it), so the card is
  unavailable, as the mind panel is, never an error.
- **The ledger's dollars only** (``Ledger.settlements``): Colab's compute units are not dollars.
- **The day's index** comes from the day phase's marker (``night.daily``), read by its path:
  ``<data dir>/sleep/daily/<persona>-<date>.json``. The simulator's ledger keys carry it.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final, TypeVar

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.delivery import passes_reply_check
from anjo.sleep.app.demo_live import DEMO_PERSONA, demo_dir
from anjo.sleep.app.demo_mind import READ_ERRORS, held_out, night_files
from anjo.sleep.app.storage import SLEEP_DIR, checkpoint_registry, data_root, evidence_store, target_store, workspace_log
from anjo.sleep.budget import LEDGER_NAME, Ledger
from anjo.sleep.demo import daily_lines
from anjo.sleep.demo.dailies import DailiesError, DailySources, NightFiles, closed, dailies_view
from anjo.sleep.demo.daily_lines import DailyLines

logger = logging.getLogger(__name__)

DAILY_LINES_FILE: Final = "daily_lines.json"
MARKERS: Final = "daily"  # the day phase's markers, under <data dir>/sleep/ (night.daily.MARKERS)
T = TypeVar("T")


def _read(what: str, read: Callable[[], T]) -> T | None:
    try:
        return read()
    except READ_ERRORS as error:
        logger.error("Dailies can't read %s: %s", what, type(error).__name__)
        return None


def lines_path() -> Path:
    return demo_dir() / DAILY_LINES_FILE


def _lines() -> DailyLines:
    found = daily_lines.load(lines_path())
    return found if found is not None else DailyLines(person=DEMO_PERSONA, generated_at="", pairs=())


def _day_index(night: date) -> int | None:
    path = data_root() / SLEEP_DIR / MARKERS / f"{DEMO_PERSONA}-{night.isoformat()}.json"
    if not path.exists():
        return None
    index = json.loads(path.read_text(encoding="utf-8")).get("day_index")
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def _day_indexes(nights: Sequence[NightFiles] | None) -> dict[date, int]:
    found: dict[date, int] = {}
    for files in nights or ():
        raw = files.summary.get("night")
        night = _read("a night's date", lambda: date.fromisoformat(str(raw)))
        index = None if night is None else _read("a day's marker", lambda: _day_index(night))
        if night is not None and index is not None:
            found[night] = index
    return found


def sources(user: str) -> DailySources:
    nights = _read("her nights", lambda: night_files(user))
    return DailySources(
        person=user,
        nights=nights,
        evidence=_read("her evidence", lambda: evidence_store(user).stored(date.max)),
        targets=_read("her targets", lambda: target_store(user).targets()),
        workspace=_read("MIND's workspace", lambda: workspace_log(user).workspace()),
        history=_read("her checkpoints", lambda: checkpoint_registry().history(user)),
        lines=_read("the line pairs", _lines),
        settlements=_read("the budget ledger", lambda: Ledger(data_root() / SLEEP_DIR / LEDGER_NAME).settlements()),
        day_index=_day_indexes(nights),
    )


def dailies_payload(user: str | None, night: int | None) -> dict[str, Any] | None:
    """Night ``night``'s card (by its number) or the latest night's; None for a night she hasn't run.
    Every section is unavailable when the account isn't synthetic or the filter can't be built."""
    held = None if user is None or not is_synthetic(user) else held_out(user)
    if user is None or held is None:
        return closed()
    try:
        return dailies_view(sources(user), held, night=night, passes=passes_reply_check)
    except DailiesError:
        return None
