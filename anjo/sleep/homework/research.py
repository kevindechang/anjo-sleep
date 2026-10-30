"""MIND's ``research`` tool in the live night (Task 17): Wonder studies one query.

- **The query is all Wonder sees.** Her query becomes the study's topic, with none of the
  person's evidence beside it.
- **Each verified source is a homework item** that Wonder wrote, so its ancestry is hosted and
  ``provenance.may_train`` keeps it out of training while hosted outputs can't train. It is stored
  in the person's evidence store before it is returned, as the ``Research`` port promises — unless
  the caller passes ``keep=False``, as the comparison's agent does: those sources stay in that
  night's scratchpad, since stored homework would fold into the brief and reach every other arm.
- **Paid like a study.** Every Wonder call comes off the night's ``WonderMeter`` (the $3-a-month
  cap) and is held on ``meta_usd`` (``do_homework``); the night debits one paid action for an
  inquiry that researched. With nothing left, or when the study fails, it returns nothing and the
  inquiry goes on without it.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.homework.agent import WONDER_ID, Source, WonderClient, do_homework
from anjo.sleep.homework.tools import PageFetcher, WebSearch
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.meter import WonderMeter, WonderSpent
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.stops import BudgetStop
from anjo.sleep.types import Evidence, Target

logger = logging.getLogger(__name__)

# A study that fails this way ends the research call, not the inquiry (as a study's failure is not
# the night's): a Wonder error or timeout, a meta_usd hold with no room, a request that can't fit
# its bound, a page in an unknown encoding.
RESEARCH_FAILURES: Final = (ModelApiError, BudgetStop, ValueError, LookupError)


def _digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class WonderResearch:
    """The ``Research`` port for one person's night step."""

    user_id: str
    night: date
    at: datetime
    store: EvidenceStore
    meter: WonderMeter
    wonder: WonderClient
    search: WebSearch
    fetcher: PageFetcher
    ledger: Ledger | None = None
    # False: the caller keeps the sources in its own pass and the store never sees them. The
    # comparison's agent passes it; MIND, which serves what it finds, keeps the default.
    keep: bool = field(default=True, kw_only=True)

    def research(self, question_id: str, query: str) -> tuple[Evidence, ...]:
        topic = Target(id=f"mind-{question_id}", person_id=self.user_id, kind="thread", label=query, check_after=None,
                       status="open", revision=0, acted_revision=0, confidence=None, last_mentioned=self.night)
        try:
            homework = do_homework(self.user_id, self.night, topic, "inquire", (), wonder=self.wonder,
                                   search=self.search, fetcher=self.fetcher, meter=self.meter, ledger=self.ledger)
        except WonderSpent:
            return ()
        except RESEARCH_FAILURES as error:
            logger.warning("MIND's research call failed (%s); the inquiry goes on without it", type(error).__name__)
            return ()
        items = [self._item(question_id, source) for source in homework.sources]
        if self.keep:  # the agent baseline keeps its sources in its pass, so the store never holds them
            held = {item.id for item in self.store.stored(self.night)}
            fresh = [item for item in items if item.id not in held]
            if fresh:
                self.store.add(fresh)
        return tuple(items)

    def _item(self, question_id: str, source: Source) -> Evidence:
        quotes = "\n".join(f"- {quote}" for quote in source.quotes)
        text = f"{source.title}\n{quotes}".strip()
        return Evidence(
            id=f"wr-{_digest(self.user_id, self.night.isoformat(), question_id, source.url, text)}",
            person_id=self.user_id, day=self.night, kind="homework", text=text, source=source.url, held_out=False,
            created_at=self.at, model=WONDER_ID, ancestry=frozenset({WONDER_ID}),
        )

