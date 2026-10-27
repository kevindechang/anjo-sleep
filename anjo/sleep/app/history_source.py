"""The ``EvidenceSource`` port over Anjo's history and stores (Task 3).

``extract_day`` reads the turns inside the window through ``history.get_history_between``, which
already leaves out forgotten turns, exchanges and sessions. For a synthetic person it reads only
their cleared sessions and imports nothing, so a filmed open never becomes evidence. For a real
person it also imports their own stored corrections, read only through ``training_eligible``,
and their explicit preferences, read through ``list_explicit_preferences``. It then records the
day: new items enter the evidence store once and their mentions update the target store. Then it
commits the day in the extraction log, so a re-run of the same day and window over the same turns
reads the stored items back without calling the model.

Times come from the window, on the night's clock: items are created at its end, and the targets
shown to the model are those open at its end. An accelerated run passes the window recorded
around each simulated day with a virtual date, so seven simulated days stay seven days.
"""
from __future__ import annotations

from datetime import date, datetime

from anjo.core.history import get_history_between
from anjo.core.logger import logger
from anjo.sleep.app.cleared_sessions import cleared_sessions, is_synthetic
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import evidence_store, extraction_log, target_store
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.extract import (
    ChatFn,
    Extraction,
    Imported,
    Turn,
    extract,
    ingest,
    own_turns,
    spark_open,
)
from anjo.sleep.types import Evidence

# A simulated person's turns are Spark Open's words (Task 2); a real person's have no ancestor.
_SIMULATED: frozenset[str] = frozenset({SPARK_OPEN})


def own_imports(user_id: str) -> list[Imported]:
    """The person's own stored corrections and preferences, in their own words."""
    from anjo.core import learning_evidence, personal_preferences
    from anjo.core.db import get_db

    rows = get_db().execute(
        "SELECT * FROM learning_evidence WHERE user_id = ? AND kind = 'explicit_preference' "
        "ORDER BY created_at, rowid",
        (user_id,),
    ).fetchall()
    imported: list[Imported] = []
    for row in rows:
        if not learning_evidence.training_eligible(row):
            continue
        patch = learning_evidence.correction_of(row)
        if patch is not None and patch["quote"].strip():
            imported.append(Imported(
                source=f"correction:{row['evidence_id']}", kind="correction",
                text=patch["quote"].strip(), ancestry=frozenset(),
            ))
    for preference in personal_preferences.list_explicit_preferences(user_id):
        quote = str(preference.get("receipt") or "").strip()
        # Without the person's quote the record is Anjo's parse of their words: unknown ancestry.
        text = quote or f"{preference['field']}: {preference['value']}"
        imported.append(Imported(
            source=f"preference:{preference['preference_id']}", kind="preference", text=text,
            ancestry=frozenset() if quote else None,
        ))
    return imported


class HistorySource:
    """Extracts and records one day of a person's evidence. ``chat`` is Spark Open by default."""

    def __init__(self, *, chat: ChatFn = spark_open, hosted_in_training: bool = False) -> None:
        self._chat = chat
        self._hosted_in_training = hosted_in_training

    def record_day(
        self, user_id: str, day: date, window: tuple[datetime, datetime],
    ) -> tuple[list[Evidence], Extraction | None]:
        """The items the day's extraction stored, and the extraction (its tokens and refusals).
        A day and window already committed over the same turns returns its stored items and
        None: no model call."""
        require_sleep_host()
        start, end = window
        synthetic = is_synthetic(user_id)
        rows = get_history_between(
            user_id, start, end, session_ids=cleared_sessions(user_id) if synthetic else None,
        )
        ancestry = _SIMULATED if synthetic else frozenset()
        turns = [
            Turn(source=f"message:{row['id']}", role=row["role"], text=row["content"], ancestry=ancestry)
            for row in rows
        ]
        read = [turn.source for turn in own_turns(turns)]
        log = extraction_log(user_id)
        evidence = evidence_store(user_id, hosted_in_training=self._hosted_in_training)
        committed = log.committed(day, window, read)
        if committed is not None:
            stored = {item.id: item for item in evidence.for_day(day)}
            if not stored.keys() >= set(committed):
                raise ValueError("a committed extraction names items the evidence store doesn't hold")
            return [stored[item_id] for item_id in committed], None
        targets = target_store(user_id)
        extraction = extract(
            user_id, day, turns, imported=() if synthetic else own_imports(user_id),
            targets=targets.open_targets(end), known_ids=evidence.ids(), now=end, chat=self._chat,
        )
        if extraction.refused:
            logger.warning("sleep extraction left out %d malformed items", len(extraction.refused))
        added = ingest(extraction, evidence, targets)
        log.commit(day, window, read, [item.id for item in added])
        return added, extraction

    def extract_day(self, user_id: str, day: date, window: tuple[datetime, datetime]) -> list[Evidence]:
        return self.record_day(user_id, day, window)[0]

