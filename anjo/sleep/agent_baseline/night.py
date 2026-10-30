"""The baseline's one night step (arm 2A): a stateless agent over the person's evidence and brief.

``AgentNight.night`` is the night's whole step. It reads the store and the served brief as of last
night, renders them, and asks Spark Open for one operation at a time: ``recall`` (the inquiry's own
``evidence_search`` over the store), ``research`` (the inquiry's web tool, when the night wired it),
``note`` (one item to leave) or ``done``. Tool answers and notes join tonight's scratchpad; nothing
else is kept. The model turns through ``spark_inquiry.spark_open``, which is ``baseten_api.chat``,
so ``ANJO_SLEEP_SPARK=fireworks`` sends it to Fireworks with that provider's id and pricing.

**The cap is MIND's.** Each call gets what is left of ``TOKEN_CAP`` (the inquiry's per-night
generated-token cap), a call that fails is charged its cap because its usage is unknown, and the
pass ends when nothing is left. It also ends after ``MAX_STEPS`` calls (the inquiry's cycle count)
or ``MAX_NOTES`` notes. Past either count nothing else is called.

**No state of its own.** The prompt is built from this call's store and brief alone, and the notes
it writes are not among the items the store shows any reader (``AGENT_NOTE``): its own earlier
notes are invisible even to it, which is what "no persistent state" has to mean. A note's id is a
digest of what it says, so a rerun of the same night writes no second copy and crashes on nothing.

**Paid, and never the night's failure.** The pass holds its Spark Open worst case on
``baseten_sleep_usd``, the pool Sleep's Spark Open opens on, and settles what it metered into the
``agent`` cost part. ``Ledger.reserve`` moves that hold onto Sleep's Fireworks pool when
``ANJO_SLEEP_SPARK=fireworks``, so the hold and its settlement never straddle the switch. A hold
with no room ends the pass as ``budget`` and a failure inside it as ``failed``; neither fails the
night, as MIND's own holds don't.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from anjo.sleep import spark_provider
from anjo.sleep.agent_baseline import parse, render
from anjo.sleep.agent_baseline.types import AgentCall, AgentOutcome, ResearchFor
from anjo.sleep.budget import Ledger
from anjo.sleep.evidence.store import AGENT_NOTE
from anjo.sleep.mind.inquiry.types import CYCLE_CAP, TOKEN_CAP, Budget, ModelReply
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
from anjo.sleep.night.mind import SEARCH_POOL  # MIND's own over-fetch before practice scenes are dropped
from anjo.sleep.spark_inquiry import ChatFn, spark_open
from anjo.sleep.stops import BudgetStop, Cancelled
from anjo.sleep.types import Evidence

logger = logging.getLogger(__name__)

FLAG: Final = "ANJO_SLEEP_AGENT_BASELINE"  # 1: the night runs the comparison's agent (arm 2A)
MAX_STEPS: Final = CYCLE_CAP  # the same count of calls MIND's inquiry may spend a night
SEARCH_K: Final = 5  # items one evidence recall returns, as the inquiry's search does
MAX_NOTES: Final = 3  # notes one night may leave
MAX_NOTE_CHARS: Final = 1_200  # one note's text, before it is stored
# Sleep's Spark Open pool, as ``night.mind`` holds it: ``Ledger.reserve`` remaps it onto the
# Fireworks pool when ``ANJO_SLEEP_SPARK=fireworks``, so the arm holds the credit it really spends.
POOL: Final = "baseten_sleep_usd"
PREFIX: Final = "agent-"  # every note's id starts here, so arm 2A can read them by id too
SOURCE: Final = "agent-baseline"  # the note's source, for arm 2A
FAILURES: Final = (ModelApiError, ValueError)  # a failure inside the pass, never the night's


def on(environ: Mapping[str, str]) -> bool:
    """Whether the night runs the comparison's agent: off unless ``ANJO_SLEEP_AGENT_BASELINE=1``."""
    return environ.get(FLAG, "").strip() == "1"


def _digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


@dataclass
class _Run:
    """One pass's state so far: its budget, what it wrote and what its calls metered. Mutable
    because a step that ends the pass early (a failed hold) still reports what it had spent."""

    prices: Prices
    budget: Budget = field(default_factory=lambda: Budget(token_cap=TOKEN_CAP))
    notes: list[str] = field(default_factory=list)
    calls: int = 0
    steps: int = 0
    recalls: int = 0
    research_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def usd(self) -> float:
        return spark_usd(self.prices, self.tokens_in, self.tokens_out)


@dataclass(frozen=True)
class AgentNight:
    """The night's baseline step: one stateless pass, on Spark Open, within MIND's token cap."""

    prices: Prices
    chat: ChatFn = spark_open
    research: ResearchFor | None = None  # the inquiry's web tool; None: recall and notes only
    ledger: Ledger | None = None  # the pool the pass holds its worst case on; None holds nothing

    def night(self, call: AgentCall) -> AgentOutcome:
        """One pass over ``call``: hold its worst case, run it, settle what it spent, never raise
        the night's own failure."""
        run = _Run(self.prices)
        stopped = "failed"
        try:
            with held(self.ledger, f"{call.job}:spark", (POOL,), usd(spark_worst_usd(self.prices, MAX_STEPS)),
                      lambda: run.usd):
                stopped = self._pass(call, run)
        except BudgetStop:
            stopped = "budget"
        except FAILURES as error:  # the pass failed, not the night: it keeps what it wrote
            logger.warning("the baseline agent's pass failed (%s); the night goes on", type(error).__name__)
        finally:
            call.spend.add("agent", run.usd)
        return AgentOutcome(
            notes=tuple(run.notes), tokens_used=run.budget.tokens_used, steps=run.steps, calls=run.calls,
            usd=run.usd, stopped=stopped,
            receipt={"recalls": run.recalls, "research_calls": run.research_calls,
                     "token_cap": run.budget.token_cap, "step_cap": MAX_STEPS},
        )

    # ── the loop ─────────────────────────────────────────────────────────────

    def _pass(self, call: AgentCall, run: _Run) -> str:
        """Ask, act, repeat, until the agent stops, a cap holds or a reply can't be read."""
        evidence = self._evidence(call)
        research = None if self.research is None else self.research(call)
        scratch: list[str] = []
        while True:
            if call.cancelled():
                raise Cancelled("the night was cancelled")
            if len(run.notes) >= MAX_NOTES:
                return "done"
            if run.steps >= MAX_STEPS:
                return "step_cap"
            left = run.budget.tokens_left
            if left <= 0:
                return "token_cap"
            reply = self._ask(run, render.messages(call.brief, evidence, scratch), left)
            run.steps += 1
            step = parse.parse_step(reply.text)
            if step is None:
                return "malformed"
            if step.op == "done":
                return "done"
            if step.op == "note":
                if step.text is not None:
                    run.notes.append(self._write(call, run, step.text, reply.model))
                    scratch.append(render.wrote(run.notes[-1]))
                continue
            assert step.query is not None  # the parser refuses a tool call without its query
            if step.op == "recall":
                run.recalls += 1
                # Over-fetch first, then cut: a practice scene is her rehearsal, not evidence, and
                # dropping it after the store's cut would return fewer than ``SEARCH_K`` items.
                found = [item for item in call.store.search(step.query, call.as_of, SEARCH_POOL)
                         if item.created_at <= call.as_of and item.kind not in ("practice", AGENT_NOTE)][:SEARCH_K]
                scratch.append(render.answer("evidence_search", step.query, found))
                continue
            if research is None:
                scratch.append(render.unoffered("research", step.query))
                continue
            run.research_calls += 1
            asked = f"{PREFIX}{call.night.isoformat()}"  # Wonder's study key for tonight's pass
            scratch.append(render.answer("research", step.query, research.research(asked, step.query)))

    # ── the tools and the note ───────────────────────────────────────────────

    def _evidence(self, call: AgentCall) -> tuple[Evidence, ...]:
        """What the agent holds tonight: the store's items as of the night, newest last, bounded to
        ``render.SHOWN``. Her practice scenes are her rehearsal, not her life, and the arm's own
        notes are not its memory, so neither is shown."""
        items = [
            item for item in call.store.stored(call.night)
            if not item.held_out and item.kind not in ("practice", AGENT_NOTE) and item.created_at <= call.as_of
        ]
        return tuple(sorted(items, key=lambda item: (item.created_at, item.id))[-render.SHOWN:])

    def _write(self, call: AgentCall, run: _Run, said: str, model: str) -> str:
        """Store one note as ``agent_note`` evidence on Spark Open's ancestry, once. Its id is a
        digest of what it says, so a rerun writes no second copy and no stored note ever differs."""
        text = said.strip()[:MAX_NOTE_CHARS]
        item = Evidence(
            id=f"{PREFIX}{_digest(call.user_id, call.night.isoformat(), text)}", person_id=call.user_id,
            day=call.night, kind=AGENT_NOTE, text=text, source=SOURCE, held_out=False, created_at=call.at,
            model=model, ancestry=frozenset({model}),
        )
        if item.id in run.notes:
            return item.id
        if item.id not in {held_item.id for held_item in call.store.agent_notes(call.night)}:
            call.store.add([item])
        return item.id

    def _ask(self, run: _Run, messages: list[dict[str, Any]], cap: int) -> ModelReply:
        """One capped call, charged what it wrote. A failed call is empty text charged its cap: its
        usage is unknown, and a hard cap never counts a failure as free (``spark_inquiry.call``)."""
        run.calls += 1
        model = spark_provider.current().model
        try:
            result: ChatResult = self.chat(messages, cap)
        except ModelApiError as error:
            logger.warning("the baseline agent's Spark Open call failed, charged its cap of %d tokens: %s", cap, error)
            run.tokens_out += cap
            run.budget = run.budget.spend(tokens=cap)
            return ModelReply(text="", tokens=cap, model=model)
        run.tokens_in += result.input_tokens
        run.tokens_out += result.output_tokens
        run.budget = run.budget.spend(tokens=result.output_tokens)
        return ModelReply(text=result.text, tokens=result.output_tokens, model=result.model or model)
