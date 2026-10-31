"""The nightly harness loop (Google's Harness^2, with RRSI's guards): propose, vet, probe, promote.

One night may change one line of the person's strategy list. The loop reads tonight's practice
transcripts where her reply lost or was flagged, asks Spark Open for exactly one atomic edit,
has a critic veto edits outside a strategy's scope, and replays ``HARNESS_SCENES`` of tonight's
scene messages through her serving Glimmer with C + the current list against C + the edited list.
The night's critic compares each pair in both orders; a flip is a tie. Only a decisive win on
``HARNESS_MIN_WINS`` scenes with a margin of ``HARNESS_MIN_MARGIN`` promotes the edit.

Why each guard:

- one atomic edit, schema-checked: an unregularized loop drifts; one edit a night is the budget.
- the critic and the code's keyword guard: some lines are never a strategy (safety, crisis, age,
  privacy, disclosure), and a prompt hack or a single restated fact is not one either.
- both orders and a noise bar: a judge that prefers the longer reply, or one scene's luck, does not
  promote a line; a flip is a tie.
- the prune: the list is full at ``STRATEGY_MAX``; an add then must replace a line.

Every Spark Open call goes through the night's metered client (the same one practice uses), so the
night's SpendMeter prices the loop like practice. Glimmer's probe replies are served on the GPU
the night already holds; the night adds that wall time to the same part.

This module reads no sealed probe, no morning score, and imports nothing under ``anjo.sleep.probes``
or ``anjo.sleep.reports``: the evaluation is never an input to the thing being learned.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.template import render
from anjo.sleep.harness.store import (
    STRATEGY_MAX,
    STRATEGY_WORDS,
    StrategyEdit,
    StrategyStore,
    StrategyVersion,
    applied,
    edit_record,
    strategy_block,
)
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.run import REPLY_TEMPERATURE, SCENE_TIMEOUT_S
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.provenance import GLIMMER, union
from anjo.sleep.types import CheckpointRef

PROPOSE_PATH: Final = Path(__file__).parent / "prompts" / "propose_v1.md"
VET_PATH: Final = Path(__file__).parent / "prompts" / "vet_v1.md"
COMPARE_PATH: Final = Path(__file__).parent / "prompts" / "compare_v1.md"
HARNESS_SCENES: Final = 8
HARNESS_MIN_WINS: Final = 5
HARNESS_MIN_MARGIN: Final = 3
# The most Spark Open calls one loop sends: the proposal, the vet, and each probed scene in both
# orders. The night holds their worst case before the first call.
HARNESS_MAX_CALLS: Final = 2 + 2 * HARNESS_SCENES
PROPOSE_MAX_TOKENS: Final = 512
VET_MAX_TOKENS: Final = 512
COMPARE_MAX_TOKENS: Final = 16
# A line that names one of these is outside the strategy list entirely, whatever the critic says.
SAFETY_TERMS: Final = re.compile(
    r"\b(safety|safe|crisis|suicid\w*|self[-\s]?harm|minor|minors?|age|aged|privacy|private|"
    r"disclos\w*|consent)\b",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


class Glimmer(Protocol):
    """Her serving Glimmer, as the loop replays a scene: ``probes.clients.ModelClient``'s reply."""

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class HarnessCall:
    """What one night's harness step reads and writes: the person's strategy log, tonight's
    practice, her serving Glimmer and the night's metered Spark Open, the run seed and the clock."""

    user_id: str
    night: date
    seed: int
    at: datetime
    store: StrategyStore
    transcripts: Sequence[PracticeTranscript]
    glimmer: Glimmer
    chat: ChatFn
    adapter: CheckpointRef | None
    cancelled: Callable[[], bool] = lambda: False


@dataclass
class _Counted:
    """The loop's Spark Open calls, counted for the receipt."""

    calls: int = 0

    def ask(self, call: HarnessCall, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        result = call.chat(
            SPARK_OPEN, [{"role": "user", "content": prompt}], None, SCENE_TIMEOUT_S,
            temperature=temperature, max_tokens=max_tokens,
        )
        return result.text


def _seed(base: int, *parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, (base, *parts))).encode()).hexdigest()[:8], 16)


def _her_reply(transcript: PracticeTranscript) -> str:
    """The reply of hers a teacher-target scene kept as the critic's rejected one; her own reply
    otherwise. A teacher-target scene whose rejected reply is the teacher's shows nothing of hers."""
    if transcript.target == "teacher":
        assert transcript.rejected is not None and transcript.rejected_from == "glimmer"
        return transcript.rejected
    return transcript.reply


def _why(transcript: PracticeTranscript) -> str:
    """Why this scene's reply lost or was flagged, from the transcript's own record. Practice keeps
    no free-text critic reason, so the loop names the record that lost."""
    if transcript.target == "teacher":
        return "the critic kept the teacher's reply over hers"
    if transcript.flagged:
        return "the critic listed her reply as missing a constraint"
    return "the critic scored the teacher's reply higher"


def _lost(transcripts: Sequence[PracticeTranscript]) -> list[PracticeTranscript]:
    """Tonight's replies of hers that lost or that the critic flagged, in scene order. A scene the
    teacher won carries her reply only as the critic's rejected candidate, and only when it is hers."""
    found: list[PracticeTranscript] = []
    for transcript in transcripts:
        if transcript.target == "teacher":
            if transcript.rejected is not None and transcript.rejected_from == "glimmer":
                found.append(transcript)
        elif transcript.flagged:
            found.append(transcript)
    return found


def _scene_text(scenes: Sequence[PracticeTranscript]) -> str:
    return "\n\n".join(
        f"Scene {scene.scene + 1}:\nThey asked: {scene.message}\n"
        f"{'Her reply (the critic rejected it)' if scene.target == 'teacher' else 'Her reply'}: "
        f"{_her_reply(scene)}\nWhy it lost: {_why(scene)}"
        for scene in scenes
    )


def _current_text(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line}" for line in lines) if lines else "- (none yet)"


def _edit_text(edit: StrategyEdit) -> str:
    if edit.action == "add":
        return f"add: {edit.line}"
    if edit.action == "replace":
        return f"replace: {edit.target}\nwith: {edit.line}"
    return f"remove: {edit.target}"


def _json(text: str) -> Any:
    return json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))


def _proposal(text: str, lines: Sequence[str]) -> tuple[StrategyEdit | None, str | None]:
    """The one edit Spark Open proposed, or None and why its answer isn't one.

    The schema is strict: a reply with another action, extra fields, a line over the limit or a
    target that isn't in the list is no edit, never a partial one."""
    try:
        data = _json(text)
    except (ValueError, TypeError):
        return None, "the proposal wasn't JSON"
    if not isinstance(data, dict):
        return None, "the proposal wasn't one edit"
    action = data.get("action")
    if action == "add":
        if set(data) - {"action", "line"} or not isinstance(data.get("line"), str):
            return None, "the add didn't carry exactly one line"
        try:
            edit = StrategyEdit("add", line=data["line"])
        except ValueError as error:
            return None, str(error)
        if len(lines) >= STRATEGY_MAX:
            return None, "the list is full: an add must replace a line"
        if edit.line in lines:
            return None, "the line is already in the list"
        return edit, None
    if action in ("replace", "remove"):
        wanted = {"action", "line", "target"} if action == "replace" else {"action", "target"}
        if set(data) != wanted:
            return None, f"the {action} didn't carry exactly its fields"
        target, line = data.get("target"), data.get("line")
        if not isinstance(target, str) or (action == "replace" and not isinstance(line, str)):
            return None, f"the {action} wasn't text"
        if target not in lines:
            return None, "the target line is not in the list"
        try:
            edit = StrategyEdit(action, line=line, target=target)
        except ValueError as error:
            return None, str(error)
        if action == "replace" and edit.line == target:
            return None, "the replacement is the line itself"
        if action == "replace" and edit.line in lines:
            return None, "the line is already in the list"
        return edit, None
    return None, "the proposal was not an add, replace or remove"


def _guard(edit: StrategyEdit) -> str | None:
    """The code's own veto: safety, crisis, age, privacy and disclosure are never strategies. Only
    the new text is scanned, so a line that already names one can still be removed or replaced."""
    if edit.line is not None and (found := SAFETY_TERMS.search(edit.line)):
        return f"{found.group(0)!r} is outside the strategy list"
    return None


def _allow(text: str) -> str | None:
    """The critic's verdict: None when it allowed the edit, else its reason (or why unreadable)."""
    try:
        data = _json(text)
    except (ValueError, TypeError):
        return "the vet's answer wasn't JSON"
    if not isinstance(data, dict) or not isinstance(data.get("allow"), bool):
        return "the vet's answer wasn't a verdict"
    if data["allow"]:
        return None
    reason = data.get("reason")
    return reason.strip() if isinstance(reason, str) and reason.strip() else "the vet rejected it without a reason"


def _winner(text: str) -> str:
    try:
        data = _json(text)
    except (ValueError, TypeError):
        return "tie"
    if not isinstance(data, dict) or data.get("winner") not in ("A", "B", "tie"):
        return "tie"
    return str(data["winner"])


def _probe_scenes(transcripts: Sequence[PracticeTranscript], seed: int, night: date) -> list[PracticeTranscript]:
    """``HARNESS_SCENES`` of tonight's scenes, picked by the night's seed and returned in scene order."""
    ranked = sorted(
        transcripts,
        key=lambda scene: hashlib.sha256(f"{seed}:{night.isoformat()}:{scene.scene}".encode()).hexdigest(),
    )
    return sorted(ranked[:HARNESS_SCENES], key=lambda scene: scene.scene)


def _messages(context: str | None, message: str) -> list[dict[str, str]]:
    system = [] if context is None else [{"role": "system", "content": context}]
    return [*system, {"role": "user", "content": message}]


def _compare(
    call: HarnessCall, counted: _Counted, scene: PracticeTranscript, first: str, second: str,
) -> str:
    """One order's verdict, as ``A``, ``B`` or ``tie`` (an unreadable answer is a tie)."""
    prompt = render(COMPARE_PATH.read_text(encoding="utf-8"), {
        "MESSAGE": scene.message, "A": first, "B": second,
    })
    return _winner(counted.ask(call, prompt, temperature=0.0, max_tokens=COMPARE_MAX_TOKENS))


def _probe(
    call: HarnessCall, counted: _Counted, current: Sequence[str], edited: Sequence[str],
) -> tuple[int, int, int, int]:
    """Replay the night's scenes on C + the current list against C + the edited list.

    Both replies use the same seed, and each pair is judged in both orders; a flip is a tie."""
    wins = ties = losses = compared = 0
    for scene in _probe_scenes(call.transcripts, call.seed, call.night):
        if call.cancelled():
            break
        seed = _seed(call.seed, call.user_id, call.night.isoformat(), scene.scene, "harness")
        first = call.glimmer.complete(
            call.adapter, _messages(strategy_block(scene.context, current), scene.message),
            temperature=REPLY_TEMPERATURE, seed=seed,
        ).text
        second = call.glimmer.complete(
            call.adapter, _messages(strategy_block(scene.context, edited), scene.message),
            temperature=REPLY_TEMPERATURE, seed=seed,
        ).text
        # A: the current list, then the edited list; the second order swaps them.
        first_order = _compare(call, counted, scene, first, second)
        second_order = _compare(call, counted, scene, second, first)
        current_wins = (first_order == "A") + (second_order == "B")
        edited_wins = (first_order == "B") + (second_order == "A")
        compared += 1
        if edited_wins == 2:
            wins += 1
        elif current_wins == 2:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses, compared


def _ancestry(call: HarnessCall, versions: Sequence[StrategyVersion]) -> frozenset[str] | None:
    """The new list's ancestry: Spark Open and Glimmer, everything the scenes derive from, and the
    list it replaces. Unknown when any part is, so the text may never train."""
    from anjo.sleep import spark_provider  # imported here: spark_provider imports the brief package

    parts = [frozenset({spark_provider.current().model, GLIMMER, SPARK_OPEN})]
    parts.extend(scene.ancestry for scene in call.transcripts)
    parts.extend(version.ancestry for version in versions)
    return union(parts)


def _receipt(
    *, current: Sequence[str], proposed: Mapping[str, Any] | None, vetted: bool | None,
    vetoed: str | None, why: str | None, wins: int = 0, ties: int = 0, losses: int = 0,
    compared: int = 0, promoted: bool = False, learned: str | None = None, calls: int = 0,
) -> dict[str, Any]:
    return {
        "proposed": None if proposed is None else dict(proposed),
        "vetted": vetted,
        **({"vetoed": vetoed} if vetoed else {}),
        **({"why": why} if why else {}),
        "wins": wins, "ties": ties, "losses": losses, "compared": compared,
        "promoted": promoted, "list_size": len(current),
        **({"learned": learned} if learned else {}),
        "calls": calls,
    }


def run_harness(call: HarnessCall) -> dict[str, Any]:
    """One night's propose, vet, probe and promotion; a receipt even when it changes nothing. The
    list and its versions are read through ``call.night``, so a rerun or a backfill sees the list as
    that night left it, never a later night's; one edit a night is the budget."""
    all_versions = call.store.versions()
    current = call.store.lines(until=call.night)
    if any(version.night == call.night for version in all_versions):
        return _receipt(current=current, proposed=None, vetted=None, vetoed=None,
                        why="the list was already edited tonight")
    newest = max((version.night for version in all_versions), default=None)
    if newest is not None and call.night < newest:
        return _receipt(current=current, proposed=None, vetted=None, vetoed=None,
                        why=f"a later night ({newest.isoformat()}) already edited the list")
    versions = tuple(version for version in all_versions if version.night <= call.night)
    if call.cancelled():
        return _receipt(current=current, proposed=None, vetted=None, vetoed=None, why="the night was cancelled")
    lost = _lost(call.transcripts)
    if not lost:
        return _receipt(current=current, proposed=None, vetted=None, vetoed=None,
                        why="no reply lost or was flagged tonight")
    counted = _Counted()
    prompt = render(PROPOSE_PATH.read_text(encoding="utf-8"), {
        "CURRENT": _current_text(current), "SCENES": _scene_text(lost),
        "MAX": str(STRATEGY_MAX), "WORDS": str(STRATEGY_WORDS),
    })
    raw = counted.ask(call, prompt, temperature=0.0, max_tokens=PROPOSE_MAX_TOKENS)
    edit, why = _proposal(raw, current)
    if edit is None:
        return _receipt(current=current, proposed=None, vetted=None, vetoed=None, why=why, calls=counted.calls)
    proposed = edit_record(edit)
    guard = _guard(edit)
    if guard is not None:
        return _receipt(current=current, proposed=proposed, vetted=False, vetoed=guard, why=None,
                        calls=counted.calls)
    vet_prompt = render(VET_PATH.read_text(encoding="utf-8"), {
        "CURRENT": _current_text(current), "EDIT": _edit_text(edit), "SCENES": _scene_text(lost),
    })
    vetoed = _allow(counted.ask(call, vet_prompt, temperature=0.0, max_tokens=VET_MAX_TOKENS))
    if vetoed is not None:
        return _receipt(current=current, proposed=proposed, vetted=False, vetoed=vetoed, why=None,
                        calls=counted.calls)
    try:
        edited = applied(current, edit)
    except ValueError as error:
        return _receipt(current=current, proposed=proposed, vetted=True, vetoed=None, why=str(error),
                        calls=counted.calls)
    wins, ties, losses, compared = _probe(call, counted, current, edited)
    promote = compared == HARNESS_SCENES and wins >= HARNESS_MIN_WINS and wins - losses >= HARNESS_MIN_MARGIN
    learned: str | None = None
    if promote and not call.cancelled():
        line = call.store.promote(
            call.night, edit, wins=wins, ties=ties, losses=losses,
            ancestry=_ancestry(call, versions), at=call.at,
        )
        current = call.store.lines(until=call.night)
        learned = edit.line if edit.line is not None else f"removed: {edit.target}"
        logger.info("harness: strategy v%d promoted for %s (%d-%d-%d)", line.version, call.user_id,
                    wins, ties, losses)
    elif promote:
        promote = False
    return _receipt(current=current, proposed=proposed, vetted=True, vetoed=None, why=None,
                    wins=wins, ties=ties, losses=losses, compared=compared, promoted=promote,
                    learned=learned, calls=counted.calls)


__all__ = [
    "HARNESS_MAX_CALLS", "HARNESS_MIN_MARGIN", "HARNESS_MIN_WINS", "HARNESS_SCENES",
    "HarnessCall", "SAFETY_TERMS", "run_harness",
]
