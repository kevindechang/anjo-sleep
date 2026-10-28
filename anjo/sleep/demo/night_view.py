"""The night view (one system, 2026-10-29): her latest night as one loop, from its story.

``story_view`` turns the night's story (``night/story.py``) into what the mind band draws left to
right, after how she slept: dream → rehearse → learn → check → wake. There is no separate MIND page
and Sleep page: each stage is one step of the same night. Each stage is:

- ``ran``, with its one key number (``key``), its exact numbers (``numbers``: each ``value`` or
  ``not_run`` with why; a record that didn't run is one ``not_run`` under its own name, never each of
  its numbers) and a link to the evidence the page already shows (``LINKS``): her questions
  for the dream, the morning card for practice, her gates for the check, the morning's scores for
  the wake. A dream that cites events sends their ids (``cites``, ``e-`` and a hash, never an account);
- ``not_run``, greyed on the page, with the reason the night wrote (``reason``). A nap stops after
  the dream: every later stage says "napped: learning tomorrow", and ``napped`` is true;
- ``reused``: practice a rerun didn't run again, with its scenes (``key``) and the night whose earlier
  run made them (``reused_from``);
- ``failed``, naming the step that failed (``step``), never its error, which can carry a message.

``brief_kept`` is the night whose brief was kept (or ``empty``) when the brief couldn't be rewritten
tonight and the night went on with the one it had, else None; its error is never sent.

``rehearsed_the_dream`` says whether the page may claim the whole loop ("dreams at night about what
she's unsure of, rehearses exactly that, and wakes up a better mind of you"): only when the dream
picked threads, practice was seeded on them and the night promoted. Otherwise the page says only
what happened.

The check also sends each gate (passed, failed or switched off, its value and threshold), how many
gates ran and whether the night promoted. A gate's reason is never sent: it can name a safety case.
Nothing here reads a clock, a file or the environment.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

from anjo.sleep.night.story import (
    STAGES,
    Check,
    Dream,
    Dreams,
    Failed,
    Imitation,
    Kept,
    Learn,
    NotRun,
    Protect,
    Rehearse,
    Reused,
    Seeded,
    Seeding,
    Sleep,
    Story,
    Surprise,
    Wake,
)

LINKS: Final[dict[str, str | None]] = {  # each stage's evidence: an id on the page (``index.html``, ``mind.js``)
    "dream": "mind-questions", "rehearse": "dailies", "learn": None, "check": "mind-gates", "wake": "mind-chart",
}


def _number(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, NotRun):
        return {"name": name, "not_run": value.reason}
    return {"name": name, "value": len(value) if isinstance(value, tuple) else value}


def _record(name: str, value: Any, parts: Callable[[Any], list[tuple[str, Any]]]) -> list[dict[str, Any]]:
    """A record's numbers, or one ``not_run`` under its own name when it didn't run."""
    return [_number(name, value)] if isinstance(value, NotRun) else [_number(n, v) for n, v in parts(value)]


def _dreams(dreams: Dreams) -> list[tuple[str, Any]]:
    return [("dreams", dreams.kept), ("dream_candidates", dreams.candidates), ("dream_keep_rate", dreams.keep_rate)]


def _seeded(seeded: Seeded) -> list[tuple[str, Any]]:
    return [("seeded", seeded.picked), ("seeded_threads", seeded.threads)]


def _surprise(surprise: Surprise) -> list[tuple[str, Any]]:
    return [("surprise", surprise.applied), ("surprise_rows", surprise.rows)]


def _seeding(seeding: Seeding) -> list[tuple[str, Any]]:
    return [("seed_rows", seeding.rows), ("seeding_kl_before", seeding.kl_before), ("seeding_kl_after", seeding.kl_after)]


def _imitation(imitation: Imitation) -> list[tuple[str, Any]]:
    return [("imitation_rows", imitation.rows), ("imitation_reward", imitation.reward)]


def _protect(protect: Protect) -> list[tuple[str, Any]]:
    return [("protect", protect.basis), ("protect_minutes", protect.minutes)]


def _cites(dream: Dream) -> list[str]:
    if isinstance(dream.ignitions, NotRun):
        return []
    return list(dict.fromkeys(cite for ignition in dream.ignitions for cite in ignition.cites))


def _dream(dream: Dream) -> dict[str, Any]:
    first = next((v for v in (dream.questions, dream.ignitions) if not isinstance(v, NotRun)), dream.questions)
    numbers = [_number(name, getattr(dream, name)) for name in ("questions", "ignitions", "threads", "boundaries")]
    return {**_keyed(first), "numbers": [*numbers, *_record("dreams", dream.dreams, _dreams)], "cites": _cites(dream)}


def _keyed(value: Any) -> dict[str, Any]:
    if isinstance(value, NotRun):
        return {"state": "not_run", "key": None, "reason": value.reason}
    return {"state": "ran", "key": len(value) if isinstance(value, tuple) else value}


def _rehearse(rehearse: Rehearse) -> dict[str, Any]:
    numbers = [_number(name, getattr(rehearse, name)) for name in ("flagged", "corrected")]
    return {**_keyed(rehearse.scenes), "numbers": [*numbers, *_record("seeded", rehearse.seeded, _seeded)]}


def _reused(reused: Reused) -> dict[str, Any]:
    return {"state": "reused", "key": reused.scenes, "reused_from": reused.night.isoformat(), "numbers": []}


def _learn(learn: Learn) -> dict[str, Any]:
    numbers = [*_record("surprise", learn.surprise, _surprise), _number("channel_loss", learn.channel_loss),
               _number("stamp", learn.stamp), *_record("seeding", learn.seeding, _seeding),
               *_record("imitation", learn.imitation, _imitation), _number("fold", learn.fold),
               *_record("protect", learn.protect, _protect), _number("core_adaptation", learn.core_adaptation)]
    return {**_keyed(learn.rows), "numbers": numbers}


def _check(check: Check) -> dict[str, Any]:
    ran = [gate for gate in check.gates if not gate.skipped]
    gates = [{"gate": g.name, "kind": "skip" if g.skipped else "pass" if g.passed else "fail", "value": g.value,
              "threshold": g.threshold} for g in check.gates]
    return {**_keyed(sum(g.passed for g in ran)), "of": len(ran), "promoted": check.promoted, "gates": gates,
            "numbers": []}


def _wake(wake: Wake) -> dict[str, Any]:
    self_model = ([_number("self_model", wake.self_model)] if isinstance(wake.self_model, NotRun) else
                  [_number("self_model_predicted", wake.self_model.predicted),
                   _number("self_model_actual", wake.self_model.actual)])
    morning = wake.morning
    if isinstance(morning, Failed):
        return {"state": "failed", "key": None, "step": morning.step, "numbers": self_model}
    if isinstance(morning, NotRun):
        return {**_keyed(morning), "numbers": self_model}
    return {"state": "ran", "key": morning.replies if not isinstance(morning.replies, NotRun) else None,
            "numbers": [_number("no_verdict", morning.no_verdict), *self_model]}


BUILDERS: Final[tuple[tuple[type, Callable[[Any], dict[str, Any]]], ...]] = (
    (Dream, _dream), (Rehearse, _rehearse), (Reused, _reused), (Learn, _learn), (Check, _check), (Wake, _wake))


def _drawn(value: Any) -> dict[str, Any]:
    if isinstance(value, NotRun):
        return {"state": "not_run", "reason": value.reason}
    if isinstance(value, Failed):
        return {"state": "failed", "step": value.step}
    for kind, build in BUILDERS:
        if isinstance(value, kind):
            return build(value)
    raise TypeError(f"no stage draws a {type(value).__name__}")


def _stage(name: str, value: Any) -> dict[str, Any]:
    shown = {"key": None, "reason": None, "step": None, "numbers": [], "cites": [], **_drawn(value)}
    return {"stage": name, **shown, "link": LINKS[name]}


def _slept(sleep: Sleep | Failed | NotRun) -> dict[str, Any]:
    if isinstance(sleep, NotRun):
        return {"not_run": sleep.reason}
    if isinstance(sleep, Failed):
        return {"failed": sleep.step}
    return {"kind": sleep.kind, "pressure": sleep.pressure, "threshold": sleep.threshold, "reason": sleep.reason}


def _rehearsed_the_dream(story: Story) -> bool:
    """The dream picked threads, practice was seeded on them, and the night promoted what it learned."""
    dream, rehearse, check = story.dream, story.rehearse, story.check
    return (isinstance(dream, Dream) and isinstance(dream.threads, tuple) and bool(dream.threads)
            and isinstance(rehearse, Rehearse) and isinstance(rehearse.seeded, Seeded) and rehearse.seeded.picked > 0
            and isinstance(check, Check) and check.promoted)


def story_view(stories: Sequence[tuple[int, Story | None]] | None) -> dict[str, Any]:
    """The newest night's loop from ``stories`` (each with its number, newest first); ``unavailable``
    when they couldn't be read or the newest couldn't be told, ``none`` before her first night."""
    if stories is None:
        return {"state": "unavailable"}
    if not stories:
        return {"state": "none"}
    number, story = stories[0]
    if story is None:
        return {"state": "unavailable", "night": number}
    return {"state": "ok", "night": number, "date": story.night.isoformat(), "status": story.status,
            "napped": story.status == "napped", "rehearsed_the_dream": _rehearsed_the_dream(story),
            "sleep": _slept(story.sleep), "brief_kept": story.brief.kept_from if isinstance(story.brief, Kept) else None,
            "stages": [_stage(name, getattr(story, name)) for name in STAGES]}
