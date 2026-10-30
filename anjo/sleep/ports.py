"""Ports: the protocols through which the engine core reaches private Anjo (spec §12.1).

The engine core imports only ``anjo.sleep`` and public libraries. Everything it needs from the
app comes through one of these protocols: the person's turns and extraction, the encrypted
stores, the context builder, delivery, the safety gate, decontamination, the GPU runner and the
intentions ledger. ``anjo.sleep.app`` implements them on private Anjo; the standalone example
(Task 15) implements them with public parts.

Stores are bound to one person when they are built (``EvidenceStore(user_id)``), so their
methods take no user id. Every time a method takes is on the night's clock, which is virtual in
accelerated runs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from anjo.sleep.types import (
    CheckpointRef,
    Choice,
    ChoiceContext,
    Commitment,
    Evidence,
    Expectation,
    ExpectationOutcome,
    GateResult,
    Opportunity,
    Probe,
    Result,
    Target,
    Want,
    WantSnapshot,
)


@runtime_checkable
class EvidenceSource(Protocol):
    """Extracts one day's evidence from the person's own turns (Task 3, ``extract_day``)."""

    def extract_day(
        self, user_id: str, day: date, window: tuple[datetime, datetime], as_of: datetime | None = None,
    ) -> list[Evidence]:
        """Read the person's turns inside ``window`` and return evidence stamped with ``day``.

        ``as_of`` is the night's clock: the targets shown are those open at it; items are created at
        the window's end (wall time) or ``as_of``, whichever is earlier, so neither clock's readers miss
        them. It defaults to the window's end (real time); an accelerated run passes the virtual one.

        Only the person's own turns are read; for a synthetic person, only cleared sessions.
        Items imported from existing stores enter once across nights. The app's source also
        records what it returns: the items enter the evidence store once and their target
        mentions update the target store, so storing them again changes nothing.
        """


@runtime_checkable
class EvidenceStore(Protocol):
    """One person's evidence, encrypted at rest, exported and erased with their account (Task 3)."""

    def add(self, items: list[Evidence]) -> None:
        """Append items; an id already stored is not added twice."""

    def for_day(self, day: date) -> list[Evidence]:
        """Every item stamped with ``day``, held-out ones included, but never an agent note."""

    def trainable(self, until: date) -> list[Evidence]:
        """Items through ``until`` that may train: never held-out items, never items with unknown
        ancestry, never items with hosted ancestry while hosted outputs can't train, and never the
        comparison's agent notes (``agent_note``: ``agent_notes``, arm 2A's read)."""

    def stored(self, until: date) -> list[Evidence]:
        """Every non-held-out item through ``until``, for served context and arm 2F; the agent
        notes are left out (only arm 2A reads them, through ``agent_notes``)."""

    def held_out(self, day: date) -> list[Evidence]:
        """The held-out items stamped with ``day``."""

    def agent_notes(self, until: date) -> list[Evidence]:
        """The baseline arm's notes through ``until``, oldest first: what the morning arm 2A reads,
        and the only way they leave this store."""

    def search(self, query: str, as_of: datetime, k: int) -> list[Evidence]:
        """``evidence_search`` (Task 17): the top ``k`` non-held-out items for ``query`` among
        those created at or before ``as_of``, never an agent note."""

    def export(self) -> dict[str, object]:
        """The person's evidence for the account export."""


@runtime_checkable
class TargetStore(Protocol):
    """One person's targets, results and expectations: an encrypted event log and its fold
    (Tasks 3 and 16). Every mutation appends an event."""

    def upsert(self, target: Target) -> None:
        """Create or replace a target by id."""

    def open_targets(self, as_of: datetime) -> list[Target]:
        """Targets whose status is ``open`` at ``as_of``."""

    def mark_acted(self, target_id: str, revision: int) -> None:
        """Record the revision a study consumed. Only the night pipeline calls it."""

    def add_result(self, result: Result) -> None:
        """Commit a study result on its target."""

    def results(self) -> list[Result]:
        """Every result, consumed ones included."""

    def settle(self, target_id: str) -> None:
        """A held or declined note settles its target: consume every unconsumed result on it,
        and set ``due_noted`` on a commitment."""

    def record_expectation(self, expectation: Expectation) -> None:
        """Record her expectation for a study she chose (Task 16)."""

    def resolve_expectation(self, expectation_id: str, outcome: ExpectationOutcome) -> None:
        """Resolve an expectation from the person's side only (Task 16)."""

    def expectations(self) -> list[Expectation]:
        """Every expectation, resolved or not."""

    def digest(self) -> str:
        """Hash the live derived state; ``replay_digest`` must equal it (Task 9's chaos run)."""

    def replay_digest(self) -> str:
        """Fold the log again from scratch and hash the derived state."""

    def export(self) -> dict[str, object]:
        """Targets, results and expectations for the account export."""


@runtime_checkable
class ContextBuilder(Protocol):
    """Context C for a reply (Task 2): one builder, so arms 2C and 4 get byte-identical text."""

    def build(self, user_id: str, probe: Probe) -> str:
        """The served context for ``probe``: memory, the served brief, preferences and
        corrections since the brief, and ``evidence_recall``."""

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int,
    ) -> tuple[Evidence, ...]:
        """``evidence_recall``: the top ``k`` items for ``query`` inside ``budget_tokens``.
        Never returns held-out items."""


@runtime_checkable
class Notifier(Protocol):
    """Delivers a held note through the app's existing outreach (Task 5)."""

    def queue(self, user_id: str, message: str, source: str) -> None:
        """Hold the note for the person's next open (for a synthetic person, the next uncleared
        one)."""

    def push(self, user_id: str, message: str) -> bool:
        """Announce a queued note by push only for a person who opted in, under the existing
        gates. Returns whether a push was sent."""


@runtime_checkable
class SafetyGate(Protocol):
    """The nightly safety gate, which runs Anjo's crisis checks against a checkpoint (Task 5).
    The case lists stay in the private app. Switched off (the hackathon), it runs its floor only."""

    def check(self, ckpt: CheckpointRef) -> GateResult:
        """Pass only with zero regressions against the base model on the same cases; switched off,
        ``skipped`` (never passed) while the floor holds."""


@runtime_checkable
class Decontaminator(Protocol):
    """Checks training text against the sealed probes (Task 5, RF5)."""

    def check(self, texts: Sequence[str], sealed: Sequence[Probe]) -> tuple[bool, ...]:
        """One flag per text, in order: ``True`` where it overlaps any sealed probe."""


@runtime_checkable
class GpuRunner(Protocol):
    """Runs one GPU job (training, practice serving, gates) on Modal, or on Colab (Task 5)."""

    def run(
        self, job: str, inputs: Mapping[str, object], *, timeout_s: float,
    ) -> Mapping[str, object]:
        """Run the named job to completion and return its outputs, or raise. A job never runs
        past ``timeout_s``."""


@runtime_checkable
class IntentionSource(Protocol):
    """Kevin's intentions ledger, for the chooser's snapshot (Task 7a). Synthetic people have
    none."""

    def intentions(self, user_id: str, as_of: datetime) -> tuple[Commitment, ...]:
        """The person's open intentions at ``as_of``."""


@runtime_checkable
class Chooser(Protocol):
    """Her chooser (Task 7a's ``StateChooser``), with ``RuleChooser`` kept as the labeled fallback.
    Both methods are pure: no clock, database, model call or environment read."""

    def choose(self, opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
        """Pick at most one want for ``opportunity`` and record a reason for every other."""

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        """Turn her state into typed wants with strengths."""
