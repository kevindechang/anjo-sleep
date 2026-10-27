"""The ports MIND's inquiry runs through (Task 17): the thinker, the screen, the tools and the log.

The kernel, controller, render and parser never call a port; only ``loop.notice`` and
``loop.run_inquiry`` do, through one ``InquiryPorts``. Each port is bound to one person (or one
APD life) when it is built, so no method takes a user id.

Who implements them:

- ``Thinker``: base Glimmer on Modal in the live night, in its own requests apart from the
  personal adapters; Glimmer on Colab vLLM at one precision for every APD arm; fakes in tests.
- ``ScreenPort``: ``anjo/sleep/app/screen.py`` on Spark Open (17-core).
- ``EvidenceSearch``: the evidence store. ``EvidenceStore.search`` (``anjo.sleep.ports``) fits it
  as it is, time-masked by ``as_of`` and never returning held-out items.
- ``WorldQuery``: the thinker's own model.
- ``Research``: Wonder, in the live night only, debited against the paid-action cap and the
  $3-a-month Wonder cap. ``InquiryPorts.research`` is ``None`` everywhere else.
- ``PredictionResolver``: Spark Open with ``resolve_prediction_v1.md``.
- ``WorkspaceLog``: the person's workspace log on Task 3's log machinery (17-core), or an
  in-memory log for an APD run.

**The thinker returns raw text.** ``Thinker.propose`` returns the model's reply as it came, and
``parse.parse_op`` turns it into an operation or ``Rejected("malformed")``. So output that isn't
an operation is logged and costs its cycle like any other rejection, every thinker (Modal, Colab,
a replay adapter, a fake) is the same thin call, and Task 17b gets the raw trajectory.

**No token meter.** Every model-backed port returns the tokens it spent, retries included, and
the loop adds them into ``Budget`` as it goes. The thinker's, ``world_query``'s and the screen's
count against ``TOKEN_CAP`` (the spec's cap is "thinker tokens with retries and screening
included"); the resolver's are reported apart in ``InquiryRun.resolver_tokens``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from anjo.sleep.mind.inquiry.types import (
    ModelReply,
    ResolverReply,
    Screened,
    ScreenPurpose,
    WorkspaceEvent,
)
from anjo.sleep.types import Evidence


@runtime_checkable
class Thinker(Protocol):
    """Proposes her next step. It sees only the rendered workspace, never the conversation."""

    def propose(self, rendered: str) -> ModelReply:
        """The model's raw reply to ``rendered``: one JSON operation in an inquiry cycle, a JSON
        array of NOTICE operations in the notice pass. Raises on a transport failure."""


@runtime_checkable
class ScreenPort(Protocol):
    """The domain rule's LLM screen, behind the kernel's denylist. It limits what she
    investigates, not what she says."""

    def screen(self, text: str, purpose: ScreenPurpose) -> Screened:
        """Whether ``text`` may stand. A ``question`` or ``hypothesis`` is refused when its
        subject is the person or a third party named in the evidence, or it diagnoses traits,
        health, feelings or relationships; a ``research_query`` when it carries the person's
        names, places or dates."""


@runtime_checkable
class EvidenceSearch(Protocol):
    """``evidence_search``: her own evidence store, searched as of a moment."""

    def search(self, query: str, as_of: datetime, k: int) -> Sequence[Evidence]:
        """The top ``k`` non-held-out items for ``query`` among those created at or before
        ``as_of``."""


@runtime_checkable
class WorldQuery(Protocol):
    """``world_query``: asks the thinker's own model. Its answers never move credence."""

    def ask(self, text: str) -> ModelReply:
        """The model's answer to ``text``."""


@runtime_checkable
class Research(Protocol):
    """``research``: Wonder, in the live night only."""

    def research(self, question_id: str, query: str) -> tuple[Evidence, ...]:
        """The verified sources Wonder found for ``query`` on the question, each as an evidence
        item (its writer in ``model``, its ancestry hosted). ``query`` was screened first, so it
        carries topic terms only. Returns ``()`` when Wonder's calls or dollars are spent."""


@runtime_checkable
class PredictionResolver(Protocol):
    """Scores a committed prediction blind: it never sees the hypotheses."""

    def resolve(self, claim: str, evidence: Sequence[Evidence]) -> ResolverReply:
        """``true``, ``false`` or ``undetermined`` for ``claim`` against ``evidence``, which is
        only what arrived after the prediction's ``committed_at``."""


@runtime_checkable
class WorkspaceLog(Protocol):
    """Her workspace log: appended one event at a time, so a kill loses nothing logged."""

    def append(self, event: WorkspaceEvent) -> None:
        """Append ``event``; refused unless its ``step.seq`` is the log's next."""

    def events(self) -> tuple[WorkspaceEvent, ...]:
        """Every event, in order."""


@dataclass(frozen=True)
class InquiryPorts:
    """Everything a run calls. ``research`` is ``None`` outside the live night, and an
    ``INVESTIGATE`` naming it is then rejected (``unavailable``)."""

    thinker: Thinker
    screen: ScreenPort
    search: EvidenceSearch
    world: WorldQuery
    resolver: PredictionResolver
    log: WorkspaceLog
    research: Research | None = None
