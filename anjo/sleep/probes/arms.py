"""The six arms of the proof (Task 2, spec 5.1): who answers, with which weights and context.

| Arm | Weights | Context |
|---|---|---|
| 1 | base | none |
| 3 | the person's adapter (base if no night was promoted) | none |
| 3x | the registered donor's adapter | none |
| 2C | base | C |
| 2F | base | C, plus every stored evidence item raw |
| 4 | the person's adapter | C, byte-identical to 2C |

``reply`` only reads: it writes no history, no evidence and no memory (RF5). It never touches the
judge brief or the scorer, which hold held-out preferences no arm may see. It is ``ask`` after
``prepare``: ``prepare`` reads the person's ports (the adapter, context C, the stored evidence) and
calls no model; ``ask`` calls the model and reads nothing else, so the morning run can prepare every
question on one thread and ask them side by side.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Final, Literal, get_args

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.ports import ContextBuilder, EvidenceStore
from anjo.sleep.probes.clients import AdapterRegistry, ModelClient, TokenLogprobs
from anjo.sleep.probes.logprob import TasteChoice, forced_choice, present
from anjo.sleep.types import CheckpointRef, Evidence, Probe

Arm = Literal["1", "2C", "2F", "3", "3x", "4"]
ARMS: Final = frozenset(get_args(Arm))
WITH_CONTEXT: Final = frozenset({"2C", "2F", "4"})
PERSONAL: Final = frozenset({"3", "4"})
REPLY_TEMPERATURE: Final = 0.7  # replies are sampled (spec 5.1); only taste_ab is deterministic
GLIMMER_WINDOW_TOKENS: Final = 131_072
REPLY_RESERVE_TOKENS: Final = 2_048  # room left for the reply itself
EVIDENCE_HEADING: Final = "## Every stored evidence item"


class DonorError(RuntimeError):
    """Arm 3x was asked for a donor other than the registered one, or the donor can't serve."""


@dataclass(frozen=True)
class ArmReply:
    """One arm's answer. ``dropped_ids`` lists the evidence 2F could not fit, oldest first.
    ``choice`` is set for a ``taste_ab`` probe, which is answered by forced choice."""

    text: str
    prompt_tokens: int
    dropped_ids: tuple[str, ...] = ()
    choice: TasteChoice | None = None


@dataclass(frozen=True)
class Question:
    """What one arm will ask about one probe: read from the ports, before any model call."""

    arm: Arm
    probe: Probe
    adapter: CheckpointRef | None
    context: str | None
    dropped_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmDeps:
    """Everything ``reply`` reaches outside through. ``donors`` maps a person to the donor
    registered for them (the persona registration, or the draw log). ``until`` is the night's
    date: 2F holds the evidence stored through it. ``context_cache`` keeps one built context
    per (person, probe) so 2C and 4 get the very same text."""

    model: ModelClient
    adapters: AdapterRegistry
    contexts: ContextBuilder
    evidence: Callable[[str], EvidenceStore]
    logprobs: TokenLogprobs
    donors: Mapping[str, str]
    until: date
    window_tokens: int = GLIMMER_WINDOW_TOKENS
    temperature: float = REPLY_TEMPERATURE
    run_seed: int = 0
    context_cache: dict[tuple[str, str], str] = field(default_factory=dict, compare=False, repr=False)


def _adapter(arm: Arm, user_id: str, deps: ArmDeps, donor: str | None) -> CheckpointRef | None:
    if arm in PERSONAL:
        return deps.adapters.last_promoted(user_id)
    if arm != "3x":
        return None
    registered = deps.donors.get(user_id)
    if registered is None:
        raise DonorError(f"{user_id} has no registered donor")
    if donor is not None and donor != registered:
        raise DonorError(f"{donor} is not the registered donor of {user_id}")
    ckpt = deps.adapters.last_promoted(registered)
    if ckpt is None:
        raise DonorError(f"the registered donor {registered} never promoted a night")
    if ckpt.person_id != registered:
        raise DonorError(f"the adapter returned is not the registered donor {registered}'s")
    return ckpt


def _context(user_id: str, probe: Probe, deps: ArmDeps) -> str:
    key = (user_id, probe.id)
    if key not in deps.context_cache:
        deps.context_cache[key] = deps.contexts.build(user_id, probe)
    return deps.context_cache[key]


def _raw_evidence(items: list[Evidence], room_tokens: int) -> tuple[str, tuple[str, ...]]:
    """Every item as a line, newest first until ``room_tokens`` is used, shown oldest first. The
    ids that didn't fit come back oldest first."""
    newest_first = sorted(items, key=lambda item: (item.created_at, item.id), reverse=True)
    kept: list[Evidence] = []
    dropped: list[Evidence] = []
    used = 0
    for item in newest_first:
        cost = estimate_tokens(_line(item))
        if not dropped and used + cost <= room_tokens:
            kept.append(item)
            used += cost
        else:
            dropped.append(item)  # once one is dropped, every older item goes too: keep the newest
    kept.reverse()
    dropped.reverse()
    return "\n".join(_line(item) for item in kept), tuple(item.id for item in dropped)


def _line(item: Evidence) -> str:
    return f"- [{item.id}] ({item.kind}, {item.day.isoformat()}) {item.text}"


def _with_all_evidence(context: str, probe: Probe, user_id: str, deps: ArmDeps) -> tuple[str, tuple[str, ...]]:
    """Context C plus every stored evidence item, truncated to the window."""
    items = [item for item in deps.evidence(user_id).stored(deps.until) if not item.held_out]
    used = estimate_tokens(context) + estimate_tokens(_question_text(probe)) + estimate_tokens(
        f"\n\n{EVIDENCE_HEADING}\n")
    room = deps.window_tokens - REPLY_RESERVE_TOKENS - used
    lines, dropped = _raw_evidence(items, max(room, 0))
    return f"{context}\n\n{EVIDENCE_HEADING}\n{lines or 'none'}", dropped


def _question_text(probe: Probe) -> str:
    if probe.options is None:
        return probe.prompt
    return "\n".join((probe.prompt, *probe.options))


def reply_seed(run_seed: int, probe_id: str, arm: Arm) -> int:
    """A per-probe, per-arm sampling seed derived from the run seed. Saved replies stay the
    record of a run; the seed only makes a re-draw repeatable where the server honours it."""
    digest = hashlib.sha256(f"{run_seed}:{probe_id}:{arm}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def prepare(
    arm: Arm, user_id: str, probe: Probe, *, deps: ArmDeps, donor: str | None = None,
) -> Question:
    """What ``arm`` will ask about ``probe`` for ``user_id``: its adapter and context, no model call.

    ``donor`` is for callers that name one (arm 3x): any donor other than the registered one is
    refused before a single adapter is asked for.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if probe.person_id != user_id:
        raise ValueError(f"probe {probe.id} belongs to another person than {user_id}")
    adapter = _adapter(arm, user_id, deps, donor)
    dropped: tuple[str, ...] = ()
    context: str | None = None
    if arm in WITH_CONTEXT:
        context = _context(user_id, probe, deps)
        if arm == "2F":
            context, dropped = _with_all_evidence(context, probe, user_id, deps)
    return Question(arm, probe, adapter, context, dropped)


def ask(question: Question, *, deps: ArmDeps) -> ArmReply:
    """The model's answer to a prepared question: its only port is the model (or its log-probabilities)."""
    probe, adapter, context, dropped = question.probe, question.adapter, question.context, question.dropped_ids
    if probe.kind == "taste_ab":
        choice = forced_choice(deps.logprobs, adapter, context, probe)
        tokens = sum(estimate_tokens(m["content"]) for m in present(probe, (0, 1), context))
        return ArmReply(choice.picked or "", tokens, dropped, choice)
    messages = [{"role": "user", "content": probe.prompt}]
    if context is not None:
        messages.insert(0, {"role": "system", "content": context})
    result = deps.model.complete(
        adapter, messages, temperature=deps.temperature,
        seed=reply_seed(deps.run_seed, probe.id, question.arm),
    )
    return ArmReply(result.text, result.input_tokens, dropped)


def reply(
    arm: Arm, user_id: str, probe: Probe, *, deps: ArmDeps, donor: str | None = None,
) -> ArmReply:
    """``arm``'s answer to ``probe`` for ``user_id``: ``prepare``, then ``ask``."""
    return ask(prepare(arm, user_id, probe, deps=deps, donor=donor), deps=deps)
