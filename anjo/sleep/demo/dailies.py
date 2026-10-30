"""Dailies, the live page's morning card: one night as a film crew reviews it the morning after.

``dailies_view`` builds one night's card from what the app read. Its slate says how the night slept
(``sleep``: a nap or a deep sleep, its pressure, threshold and reason; None before naps were logged).
It has three sections, each ``ok``,
``none`` (nothing to show) or ``unavailable`` (its source couldn't be read), as in ``panel``:

- ``learned``: what she (Anjo, his model) took from his day. These are the targets the night
  opened (first mentioned that day) or moved (mentioned again), from the person's own evidence of
  that day; what the night chose and what it beat; and any question MIND raised from that day's
  evidence (its newest origin item is the night's).
- ``line``: one line she'd now say differently. ``words.DAILY_SCENE`` is answered notes off by the
  checkpoint that served before the night and by the one the night promoted, pre-generated as one
  pair (``daily_lines``). ``one_line`` cuts the line: the first sentence, in order, where the two
  replies differ; a side with no sentence there is ``null``. A night that promoted nothing has no
  pair, and the card says so (``reason``).
- ``cost``: what the night cost, in dollars at list price and GPU-minutes, split into the simulated
  day, practice, training, the morning run and the rest (extraction, gates, the brief, MIND, the
  note's GPU and Wonder's calls).
  - Dollars come from the night's summary when it carries its cost parts (``NightResult.cost_parts``).
    Otherwise they come from the budget ledger's dollar settlements under the night's own keys
    (``ledger_part``), and ``source`` says which. The summary never holds the day, the morning run,
    the note's GPU (``night:note:``), Wonder's calls (``night:homework:``, the summary's
    ``wonder_usd``) or the probes (``night:probes:``), so those always come from the ledger, and
    both sources count the same things.
  - The ledger can't split the batch's shared GPU session (``night_gpu:``) per person, so the
    fallback leaves out practice's and the gates' serving time, which the summary carries. Colab's
    training is in compute units, not dollars, so the fallback never counts it (``colab_train:``);
    a hold is never counted.
  - GPU-minutes come only from durations a step measured: the night log's ``practice``, ``train``,
    ``gates``, ``morning`` and ``dailies`` records. Nothing times the day's serving.
  - ``projection`` is a person-month at the pace of the nights so far against the $10 ceiling, as
    spec §5.5 counts it (every cost part but MIND), labeled development.

Every free text passes the held-out filter and is sent as ``{"text": ..., "withheld": ...}``. The
free texts are a target's label, a question's subject and the line. The line is also withheld when
the reply check (``passes``, required) trips on either full reply. Time is the night's clock: a night's card
reads nothing dated after it, and its ``morning`` is the day after its date. Nothing here reads a
clock, a file or the environment.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final

from anjo.sleep.demo.daily_lines import LABEL, DailyLines
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.panel import NightRecord, candidate_view, night_record, slept
from anjo.sleep.demo.words import DAILY_SCENE
from anjo.sleep.evidence.targets import PERSON_KINDS
from anjo.sleep.mind.inquiry.types import Question, Workspace
from anjo.sleep.types import CheckpointRef, Evidence, Target

DEVELOPMENT: Final = "development"
MONTH_NIGHTS: Final = 30
CEILING_USD: Final = 10.0  # a person's nightly upkeep, a month (spec §5.5)
CEILING_PARTS: Final = ("extract", "practice", "training", "gates", "brief")  # every cost part but MIND's and the agent arm's
TRAINING_STEPS: Final = ("practice", "dataset", "train", "gates", "promote")  # a night failed here had started training
CARD_PARTS: Final = ("day", "practice", "training", "morning", "rest")
SUMMARY_PARTS: Final[Mapping[str, tuple[str, ...]]] = {
    "practice": ("practice",), "training": ("training",), "rest": ("extract", "gates", "brief", "mind", "agent")}
TIMED_STEPS: Final[Mapping[str, tuple[str, ...]]] = {
    "practice": ("practice",), "training": ("train",), "morning": ("morning", "dailies"), "rest": ("gates",)}
TRAIN_HOLDS: Final = ("modal_train",)  # Colab's (``colab_train``) are compute units, never dollars
UNSUMMARIZED: Final = ("note", "homework", "probes")  # night steps whose spend the summary never carries
OUTSIDE_SUMMARY: Final = "outside_summary"  # their ledger bucket: part of ``rest`` from either source
TARGETS_SHOWN: Final = 5
LINE_WORDS: Final = 40
_MARKUP = re.compile(r"^\s*(?:[-*•>]+|#+|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…][\"'”’)])\s+")


class DailiesError(LookupError):
    """The night asked for is not one she ran."""


@dataclass(frozen=True)
class NightFiles:
    """One night's ``summary.json``, its ``log.jsonl`` records and its ``story.json`` when it has one
    (``night.nightlog``), parsed as it is made: a malformed night raises ``PanelError`` where the app
    reads it, so its read fails and the card is ``unavailable``, never an error."""

    summary: Mapping[str, Any]
    steps: Sequence[Mapping[str, Any]] = ()
    story: Mapping[str, Any] | None = None
    record: NightRecord = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", night_record(self.summary, self.steps, self.story))


@dataclass(frozen=True)
class DailySources:
    """What the app read. ``None``: that source couldn't be read. ``person`` is the account the
    ledger's keys name, never sent. ``day_index``: each night's simulated-day index (the day
    phase's marker), which the simulator's ledger key carries."""

    person: str
    nights: Sequence[NightFiles] | None = None
    evidence: Sequence[Evidence] | None = None
    targets: Sequence[Target] | None = None
    workspace: Workspace | None = None
    history: Sequence[CheckpointRef] | None = None
    lines: DailyLines | None = None
    settlements: Mapping[str, float] | None = None
    day_index: Mapping[date, int] = field(default_factory=dict)


# ── the line ─────────────────────────────────────────────────────────────────


def sentences(text: str) -> list[str]:
    """``text``'s sentences in order, list and heading markers and bold marks dropped."""
    out: list[str] = []
    for raw in text.splitlines():
        line = _MARKUP.sub("", raw).replace("**", "").replace("__", "").strip()
        out.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return out


def _same(first: str | None, second: str | None) -> bool:
    return (first or "").casefold().split() == (second or "").casefold().split()


def _clip(text: str | None) -> str | None:
    if text is None:
        return None
    words = text.split()
    return text if len(words) <= LINE_WORDS else " ".join(words[:LINE_WORDS]) + "…"


def one_line(before: str, after: str) -> tuple[str | None, str | None] | None:
    """The first sentence, in order, where ``before`` and ``after`` differ, each cut to
    ``LINE_WORDS`` words; None when he'd say every sentence the same way."""
    first, second = sentences(before), sentences(after)
    for index in range(max(len(first), len(second))):
        mine = first[index] if index < len(first) else None
        theirs = second[index] if index < len(second) else None
        if not _same(mine, theirs):
            return _clip(mine), _clip(theirs)
    return None


# ── reading a night ───────────────────────────────────────────────────────────


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _numbered(nights: Sequence[NightFiles]) -> list[tuple[int, NightRecord, NightFiles]]:
    """Each night with its number, oldest first; a night run again counts once, its last run."""
    last: dict[date, tuple[NightRecord, NightFiles]] = {}
    for files in nights:
        last[files.record.night] = (files.record, files)
    return [(n, *last[day]) for n, day in enumerate(sorted(last), start=1)]


def _no_verdict(files: NightFiles) -> int:
    """How many of the morning's judged probes the judge gave no readable verdict on (its receipt's count)."""
    morning = next((s for s in reversed(files.steps) if s.get("step") == "morning"), {})
    count = morning.get("no_verdict")
    return count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0


def _section(state: str, **body: Any) -> dict[str, Any]:
    return {"state": state, **body}


def _learned_empty(state: str) -> dict[str, Any]:
    return _section(state, targets=[], more=0, chose=None, beat=[], questions=[])


def _moved(on: date, evidence: Sequence[Evidence], targets: Mapping[str, Target], held: HeldOutText) -> list[dict[str, Any]]:
    own = [e for e in evidence if e.target_id in targets and e.kind in PERSON_KINDS and not e.held_out and e.day <= on]
    tonight = {e.target_id for e in own if e.day == on and e.target_id is not None}
    earlier = {e.target_id for e in own if e.day < on}
    order = sorted(tonight, key=lambda tid: (tid in earlier, targets[tid].kind != "commitment", tid))
    return [{"kind": targets[tid].kind, "label": held.item(targets[tid].label),
             "change": "moved" if tid in earlier else "opened"} for tid in order]


def _raised_on(question: Question, days: Mapping[str, date]) -> date | None:
    found = [days[item] for item in question.origin_evidence if item in days]
    return max(found) if found else None


def _learned(on: date, record: NightRecord, sources: DailySources, held: HeldOutText) -> dict[str, Any]:
    if sources.evidence is None or sources.targets is None or sources.workspace is None:
        return _learned_empty("unavailable")
    by_id = {t.id: t for t in sources.targets}
    labels = {t.id: t.label for t in sources.targets}
    moved = _moved(on, sources.evidence, by_id, held)
    days = {e.id: e.day for e in sources.evidence}
    questions = [{"kind": q.kind, "subject": held.item(q.subject)}
                 for q in sources.workspace.questions if _raised_on(q, days) == on]
    beat = [candidate_view(c, labels, held) for c in record.candidates if c.status != "won"] if record.action else []
    chose = {"action": record.action, "target": held.item(labels.get(record.target_id or "")),
             "outcome": record.outcome, "note": record.note}
    state = "ok" if moved or questions or beat or record.action else "none"
    return _section(state, targets=moved[:TARGETS_SHOWN], more=max(0, len(moved) - TARGETS_SHOWN), chose=chose,
                    beat=beat, questions=questions)


def _line_empty(state: str, reason: str | None = None) -> dict[str, Any]:
    scene = {"id": DAILY_SCENE.id, "title": DAILY_SCENE.title, "prompt": DAILY_SCENE.prompt}
    return _section(state, reason=reason, label=LABEL, scene=scene, before=None, after=None, same=False,
                    before_night=None, after_night=None, generated_at=None)


def line_checkpoints(
    history: Sequence[CheckpointRef], night: date,
) -> tuple[CheckpointRef | None, CheckpointRef] | None:
    """The line's two checkpoints for ``night``: the one serving before it (None: the base model)
    and the one it promoted; None when it promoted none. ``history`` is the promoted ones, oldest
    first."""
    after = next((c for c in reversed(history) if c.night == night), None)
    if after is None:
        return None
    return next((c for c in reversed(history) if c.night < night), None), after


def _line(
    on: date, status: str, numbers: Mapping[date, int], sources: DailySources, held: HeldOutText,
    passes: Callable[[str], bool],
) -> dict[str, Any]:
    if sources.history is None or sources.lines is None:
        return _line_empty("unavailable")
    found = line_checkpoints(sources.history, on)
    if status != "promoted" or found is None:
        return _line_empty("none", "no_new_checkpoint")
    before, after = found
    pair = sources.lines.find(None if before is None else before.id, after.id, DAILY_SCENE)
    if pair is None:
        return _line_empty("none", "not_generated")
    shown = {**_line_empty("ok"), "before_night": 0 if before is None else numbers.get(before.night),
             "after_night": numbers.get(on), "generated_at": pair.generated_at}
    replies = (pair.before_side.text, pair.after_side.text)
    if any(held.carries(reply) or not passes(reply) for reply in replies):
        return {**shown, "before": {"text": None, "withheld": True}, "after": {"text": None, "withheld": True}}
    cut = one_line(pair.before_side.text, pair.after_side.text)
    if cut is None:
        return {**shown, "same": True}
    return {**shown, "before": held.item(cut[0]), "after": held.item(cut[1])}


# ── what the night cost ───────────────────────────────────────────────────────


def ledger_part(job: str, person: str, night: date, day_index: int | None) -> str | None:
    """The card part a settled ledger key belongs to, for ``person``'s ``night`` (``OUTSIDE_SUMMARY``
    for a night step the summary never carries); None for another person's, another night's, the
    batch's shared GPU session or Colab's compute units."""
    head, _, rest = job.partition(":")
    on = night.isoformat()
    if head == "sim":
        return "day" if day_index is not None and rest.startswith(f"{person}:d{day_index}:") else None
    if head in ("morning", "dailies"):
        return "morning" if rest.startswith(f"{on}:{person}:") else None
    if head in TRAIN_HOLDS:
        return "training" if rest.startswith(f"{on}:{person}:") else None
    step, _, key = rest.partition(":")
    if head == "night" and key.startswith(f"{person}:{on}:"):
        return "practice" if step == "practice" else OUTSIDE_SUMMARY if step in UNSUMMARIZED else "rest"
    return None


def _settled(sources: DailySources, on: date) -> dict[str, float] | None:
    if sources.settlements is None:
        return None
    found: dict[str, list[float]] = {}
    for job, amount in sources.settlements.items():
        part = ledger_part(job, sources.person, on, sources.day_index.get(on))
        if part is not None:
            found.setdefault(part, []).append(amount)
    return {part: math.fsum(amounts) for part, amounts in found.items()}


def _minutes(step: Mapping[str, Any]) -> float | None:
    if step.get("step") == "train":
        receipt = step.get("receipt")
        return _number(receipt.get("gpu_minutes")) if isinstance(receipt, Mapping) else None
    seconds = _number(step.get("seconds"))
    return None if seconds is None else seconds / 60.0


def _timed(part: str, steps: Sequence[Mapping[str, Any]]) -> float | None:
    found = [m for step in steps if step.get("step") in TIMED_STEPS.get(part, ()) if (m := _minutes(step)) is not None]
    return math.fsum(found) if found else None


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _dollars(name: str, files: NightFiles, settled: Mapping[str, float]) -> tuple[float | None, str | None]:
    """One part's dollars and where they came from. ``rest`` adds, from the ledger, the night steps
    the summary never carries, so both sources count the same things."""
    parts = files.summary.get("cost_parts")
    extra = settled.get(OUTSIDE_SUMMARY) if name == "rest" else None
    if name in SUMMARY_PARTS and isinstance(parts, Mapping):
        return math.fsum([_number(parts.get(p)) or 0.0 for p in SUMMARY_PARTS[name]] + [extra or 0.0]), "summary"
    if name in settled or extra is not None:
        return math.fsum([settled.get(name, 0.0), extra or 0.0]), "ledger"
    return None, None


def _part(name: str, files: NightFiles, settled: Mapping[str, float] | None) -> dict[str, Any]:
    usd, source = _dollars(name, files, {} if settled is None else settled)
    return {"part": name, "usd": _rounded(usd, 6), "gpu_minutes": _rounded(_timed(name, files.steps), 3),
            "source": source}


def _total(values: Sequence[float | None], digits: int) -> float | None:
    known = [v for v in values if v is not None]
    return round(math.fsum(known), digits) if known else None


def _upkeep(summary: Mapping[str, Any]) -> float | None:
    parts = summary.get("cost_parts")
    if not isinstance(parts, Mapping):
        return None
    return math.fsum(_number(parts.get(p)) or 0.0 for p in CEILING_PARTS)


def _trained(summary: Mapping[str, Any]) -> bool:
    return (summary.get("trained") is True or summary.get("status") in ("promoted", "rejected")
            or summary.get("failed_at") in TRAINING_STEPS)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def projection(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """A person-month at the pace of ``summaries`` (spec §5.5): 30 × (trained share × a trained
    night's mean upkeep + the rest × a quiet night's), and the worst case, every night trained.
    Nights whose summary carries no cost parts are left out; None when none carries them."""
    known = [(_trained(s), upkeep) for s in summaries if (upkeep := _upkeep(s)) is not None]
    if not known:
        return None
    trained = [u for was, u in known if was]
    quiet = [u for was, u in known if not was]
    share = len(trained) / len(known)
    month = MONTH_NIGHTS * (share * _mean(trained) + (1.0 - share) * _mean(quiet))
    return {"label": DEVELOPMENT, "month_usd": round(month, 6), "ceiling_usd": CEILING_USD,
            "worst_month_usd": round(MONTH_NIGHTS * _mean(trained), 6) if trained else None,
            "trained_share": share, "nights": len(known), "under_ceiling": month <= CEILING_USD}


def _cost(files: NightFiles, upto: Sequence[NightFiles], sources: DailySources, on: date) -> dict[str, Any]:
    parts = [_part(name, files, _settled(sources, on)) for name in CARD_PARTS]
    usd = _total([p["usd"] for p in parts], 6)
    minutes = _total([p["gpu_minutes"] for p in parts], 3)
    source = "summary" if isinstance(files.summary.get("cost_parts"), Mapping) else "ledger"
    return _section("ok" if usd is not None or minutes is not None else "none", source=source, parts=parts, usd=usd,
                    gpu_minutes=minutes, projection=projection([f.summary for f in upto]))


# ── the card ──────────────────────────────────────────────────────────────────


def _empty(state: str) -> dict[str, Any]:
    return {"state": state, "night": None, "date": None, "morning": None, "status": None, "as_of": None, "nights": [],
            "skipped_gates": [], "no_verdict": 0,
            "learned": _learned_empty(state), "line": _line_empty(state),
            "cost": _section(state, source=None, parts=[], usd=None, gpu_minutes=None, projection=None)}


def closed() -> dict[str, Any]:
    """The card with every section unavailable: what an account that isn't synthetic, or a failed
    held-out filter, gets."""
    return _empty("unavailable")


def dailies_view(
    sources: DailySources, held: HeldOutText, *, passes: Callable[[str], bool], night: int | None = None,
) -> dict[str, Any]:
    """Night ``night``'s card (by its number), or her latest night's; ready to send, nothing in it
    carries held-out text. ``passes`` is Anjo's reply check, which every line must pass (there is
    no default: it never fails open). Raises ``DailiesError`` for a night she hasn't run."""
    if sources.nights is None:
        return closed()
    numbered = _numbered(sources.nights)
    if not numbered:
        if night is not None:
            raise DailiesError(f"night {night} hasn't run")
        return _empty("none")
    found = numbered[-1] if night is None else next((each for each in numbered if each[0] == night), None)
    if found is None:
        raise DailiesError(f"night {night} hasn't run")
    number, record, files = found
    on = record.night
    numbers = {each[1].night: each[0] for each in numbered}
    upto = [each[2] for each in numbered if each[1].night <= on]
    return {
        "state": "ok", "night": number, "date": on.isoformat(), "morning": (on + timedelta(days=1)).isoformat(),
        "status": record.status, "sleep": slept(record),  # a nap or a deep sleep, its pressure and why
        "as_of": datetime.combine(on + timedelta(days=1), time.min, timezone.utc).isoformat(),
        "skipped_gates": [g.gate for g in record.gates if g.skipped],  # switched off: shown skipped, never passed
        "no_verdict": _no_verdict(files),
        "nights": [{"night": n, "date": r.night.isoformat(), "status": r.status,
                    "sleep": None if r.sleep is None else r.sleep.kind} for n, r, _ in reversed(numbered)],
        "learned": _learned(on, record, sources, held),
        "line": _line(on, record.status, numbers, sources, held, passes),
        "cost": _cost(files, upto, sources, on),
    }
