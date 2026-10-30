"""The ports MIND's inquiry runs through (Task 17): the thinker, the tools and the log.

The kernel, controller, render and parser never call a port; only ``loop.notice`` and
``loop.run_inquiry`` do, through one ``InquiryPorts``. Each port is bound to one person (or one
APD life) when it is built, so no method takes a user id.

Who implements them:

- ``Thinker``: base Glimmer on Modal in the live night, in its own requests apart from the
  personal adapters; Glimmer on Colab vLLM at one precision for every APD arm; fakes in tests.
- ``EvidenceSearch``: the evidence store. ``EvidenceStore.search`` (``anjo.sleep.ports``) fits
  ``search`` as it is; 17-core adds ``get``, for example by filtering ``stored(as_of.date())``.
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

**The token cap is hard.** The thinker and ``world_query`` count against ``TOKEN_CAP``; a token
is a generated token, since prompts aren't charged. Each of their calls takes ``max_tokens``, the
night's remaining allowance (``Budget.tokens_left``), and spends no more, retries included. Each
returns the tokens it spent, and the loop charges them to the step the call served. A model call
that fails after its retries doesn't raise: it returns empty text with every token it spent, so
usage on a failed call still counts. The resolver's tokens are reported apart, in
``InquiryRun.resolver_tokens``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from anjo.sleep.mind.inquiry.types import ModelReply, ResolverReply, WorkspaceEvent
from anjo.sleep.types import Evidence


@runtime_checkable
class Thinker(Protocol):
    """Proposes her next step. It sees only the rendered workspace, never the conversation."""

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        """The model's raw reply to ``rendered``, spending at most ``max_tokens``: one JSON
        operation in an inquiry cycle, a JSON array of NOTICE operations in the notice pass."""


@runtime_checkable
class EvidenceSearch(Protocol):
    """``evidence_search``: her own evidence store, searched as of a moment, and fetched again
    by id, so the render and the resolver see the same text after a resume."""

    def search(self, query: str, as_of: datetime, k: int) -> Sequence[Evidence]:
        """The top ``k`` non-held-out items for ``query`` among those created at or before
        ``as_of``."""

    def get(self, ids: Sequence[str], as_of: datetime) -> tuple[Evidence, ...]:
        """The items with these ids that were created at or before ``as_of``, in the order
        asked. An id it doesn't hold, a later item and a held-out item are left out."""


@runtime_checkable
class WorldQuery(Protocol):
    """``world_query``: asks the thinker's own model. Its answers never move credence."""

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        """The model's answer to ``text``, spending at most ``max_tokens``."""


@runtime_checkable
class Research(Protocol):
    """``research``: Wonder, in the live night only."""

    def research(self, question_id: str, query: str) -> tuple[Evidence, ...]:
        """The verified sources Wonder found for ``query`` on the question, each as an evidence
        item (its writer in ``model``, its ancestry hosted) that is already in the person's
        evidence store, so ``EvidenceSearch.get`` finds it by id later. Returns ``()`` when
        Wonder's calls or dollars are spent."""


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
    search: EvidenceSearch
    world: WorldQuery
    resolver: PredictionResolver
    log: WorkspaceLog
    research: Research | None = None
