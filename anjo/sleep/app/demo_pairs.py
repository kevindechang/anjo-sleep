"""Pre-generate the live page's starter pairs (Task 13a):

    python -m anjo.sleep.app.demo_pairs [--out PATH] [--wake] [--nights | --dailies]

Each starter scene is answered once by arm 4 and once by the demo's baseline
(``ANJO_SLEEP_DEMO_BASELINE``), through ``arms.reply`` on the checkpoint serving now, for the account
that plays the film student (``ANJO_SLEEP_DEMO_USER_ID``, registered synthetic). The file,
``<data dir>/sleep/demo/starter_pairs.json`` unless ``--out`` says otherwise, is labeled
pre-generated with that checkpoint and its night, and the page serves it with no GPU and no model
call. Run it again whenever the serving checkpoint changes.

``--nights`` writes the night slider's answers instead (the full step): every scene answered notes
off at each stop the page shows, arm 3 on the stop's checkpoint and arm 1 at night 0, into
``night_answers.json``. A checkpoint already answered keeps its answers, so after a new night only
its checkpoint is asked; a stop whose checkpoint was trained under another rendering is skipped.

``--dailies`` writes Dailies' lines (the morning card): for each night that promoted a checkpoint,
``words.DAILY_SCENE`` answered notes off by the checkpoint serving before it and by its own, into
``daily_lines.json``. A pair on file is kept and asks nothing. After each night the night phase does
the same for that night alone (``after_night``), while Glimmer is still up; its GPU time is held
and settled in the budget ledger under ``dailies:<night>:<account>``.

Glimmer must already be up, or ``--wake`` wakes her for this run, which then ends asleep. Every
keep-warm window needs Kevin's OK, so nothing here wakes her unasked. Each side's cost is its
measured seconds at the serving GPU's list price (``ANJO_SLEEP_PRICES``), or unmeasured without it.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import date, datetime, timezone
from pathlib import Path
from time import monotonic

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.demo_live import (
    DEMO_PERSONA, STARTERS_FILE, USER_ENV, GlimmerArms, Warmth, baseline_from, demo_dir, glimmer_deps, glimmer_status,
    serve_rate,
)
from anjo.sleep.app.demo_dailies import lines_path
from anjo.sleep.app.demo_mind import NIGHT_ANSWERS_FILE, demo_stops, serving_rendering
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import SLEEP_DIR, checkpoint_registry, data_root
from anjo.sleep.budget import LEDGER_NAME, Bound, Ledger
from anjo.sleep.demo import daily_lines, night_answers
from anjo.sleep.demo.dailies import line_checkpoints
from anjo.sleep.demo.daily_lines import DailyLines
from anjo.sleep.demo.night_answers import NightAnswers
from anjo.sleep.demo.service import Answer, LiveArms, StopArms
from anjo.sleep.demo.slider import Stop
from anjo.sleep.demo.starters import Side, StarterSet, generate, to_json
from anjo.sleep.demo.words import DAILY_SCENE, SCENES, Scene
from anjo.sleep.night.holds import held
from anjo.sleep.probes.morning import ARM_POOLS
from anjo.sleep.serving import backend
from anjo.sleep.serving import client as vllm
from anjo.sleep.types import CheckpointRef, NightResult

logger = logging.getLogger(__name__)


def _side(answer: Answer, serve_usd_per_hour: float | None) -> Side:
    cost = None if serve_usd_per_hour is None else round(answer.seconds * serve_usd_per_hour / 3600.0, 6)
    return Side(answer.text, answer.prompt_tokens, cost)


def write_starters(
    out: Path, arms: LiveArms, *, checkpoint: CheckpointRef | None, baseline: str,
    serve_usd_per_hour: float | None, now: datetime,
) -> StarterSet:
    """Answer every scene with ``arms`` and write the labeled file, whole or not at all."""

    def answer(scene: Scene) -> tuple[Side, Side]:
        product, other = arms.answer_pair(scene.prompt, baseline)
        return _side(product, serve_usd_per_hour), _side(other, serve_usd_per_hour)

    starters = generate(SCENES, answer, person=DEMO_PERSONA, checkpoint=checkpoint, baseline=baseline, now=now)
    _write(out, to_json(starters))
    return starters


def _write(out: Path, text: str) -> None:
    """``text`` at ``out``, whole or not at all."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(f"{out.name}.part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, out)


def write_night_answers(
    out: Path, arms: StopArms, stops: Sequence[Stop], *, serve_usd_per_hour: float | None, now: datetime,
) -> NightAnswers:
    """Every scene answered notes off at every stop that can serve, keeping what ``out`` holds."""
    checkpoints = list(dict.fromkeys(stop.checkpoint for stop in stops if stop.available))

    def answer(checkpoint: CheckpointRef | None, scene: Scene) -> Side:
        return _side(arms.answer_at(scene.prompt, checkpoint), serve_usd_per_hour)

    answers = night_answers.generate(SCENES, checkpoints, answer, keep=night_answers.load(out), person=DEMO_PERSONA,
                                     now=now)
    _write(out, night_answers.to_json(answers))
    return answers


# ── Dailies' line ─────────────────────────────────────────────────────────────


def write_daily_line(
    out: Path, arms: StopArms, before: CheckpointRef | None, after: CheckpointRef, *, keep: DailyLines | None,
    serve_usd_per_hour: float | None, now: datetime,
) -> DailyLines:
    """The scene answered at ``before``, then at ``after``, added to ``keep`` and written whole."""

    def answer(checkpoint: CheckpointRef | None, scene: Scene) -> Side:
        return _side(arms.answer_at(scene.prompt, checkpoint), serve_usd_per_hour)

    lines = daily_lines.generate(DAILY_SCENE, before, after, answer, keep=keep, person=DEMO_PERSONA, now=now)
    _write(out, daily_lines.to_json(lines))
    return lines


def _gpu_hold(user: str, night: date, rate: float | None, timer: Callable[[], float]) -> AbstractContextManager[None]:
    """The two answers' GPU time, held at the worst case and settled as measured; none unpriced."""
    if rate is None:
        return nullcontext()
    started = timer()
    return held(Ledger(data_root() / SLEEP_DIR / LEDGER_NAME), f"dailies:{night.isoformat()}:{user}", ARM_POOLS,
                Bound(rate, 2 * vllm.REQUEST_TIMEOUT_S), lambda: rate * max(0.0, timer() - started) / 3600.0)


def line_for(user: str, night: date, arms: StopArms, *, timer: Callable[[], float] = monotonic) -> DailyLines | None:
    """Night ``night``'s line, unless it promoted nothing, a checkpoint was trained under another
    rendering than the one serving, or the pair is already on file (then nothing is asked)."""
    found = line_checkpoints(checkpoint_registry().history(user), night)
    if found is None:
        return None
    before, after = found
    rendering = serving_rendering()
    if any(c is not None and c.rendering != rendering for c in found):
        logger.warning("Dailies' line for %s: a checkpoint was trained under another rendering", night.isoformat())
        return None
    out = lines_path()
    keep = daily_lines.load(out)
    if keep is not None and keep.find(None if before is None else before.id, after.id, DAILY_SCENE) is not None:
        return keep
    rate = serve_rate()
    with _gpu_hold(user, night, rate, timer):
        return write_daily_line(out, arms, before, after, keep=keep, serve_usd_per_hour=rate,
                                now=datetime.now(timezone.utc))


def after_night(
    user: str, persona: str, night: date, result: NightResult, *, arms: StopArms | None = None,
    timer: Callable[[], float] = monotonic,
) -> DailyLines | None:
    """The night phase's step (``night.daily``): after a night that promoted new weights, its line,
    on the Glimmer the night left up. Only the demo's persona, on a synthetic account."""
    require_sleep_host()
    if persona != DEMO_PERSONA or not is_synthetic(user):
        raise ValueError("Dailies' line is the demo persona's, on an account registered synthetic")
    if result.status != "promoted":
        return None
    if arms is None:
        arms = GlimmerArms(user, deps=glimmer_deps(user, backend.from_env()), status=lambda: True)
    return line_for(user, night, arms, timer=timer)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.app.demo_pairs", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="where to write (default: the demo's folder)")
    parser.add_argument("--wake", action="store_true", help="wake Glimmer for this run; it ends asleep")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--nights", action="store_true", help="the night slider's answers, notes off, per stop")
    which.add_argument("--dailies", action="store_true", help="Dailies' lines: each promoted night, before and after")
    args = parser.parse_args(argv)
    require_sleep_host()
    user = os.environ.get(USER_ENV, "").strip()
    if not user or not is_synthetic(user):
        print(f"{USER_ENV} must name an account registered as synthetic", file=sys.stderr)
        return 2
    serving = backend.from_env()
    warmth = Warmth()
    with backend.phase(serving.sleep_now if args.wake else None):
        if args.wake:
            if not serving.wake().ok:
                print("glimmer_not_ready", file=sys.stderr)
                return 1
            warmth.mark()
        elif not glimmer_status(serving, warmth):
            print("Glimmer is asleep; pass --wake to wake her for this run", file=sys.stderr)
            return 1
        arms = GlimmerArms(user, deps=glimmer_deps(user, serving), status=lambda: True)
        if args.nights:
            return _nights(args.out, arms, user)
        if args.dailies:
            return _dailies(arms, user)
        starters = write_starters(
            args.out or demo_dir() / STARTERS_FILE, arms, checkpoint=checkpoint_registry().last_promoted(user),
            baseline=baseline_from(os.environ), serve_usd_per_hour=serve_rate(),
            now=datetime.now(timezone.utc))
    print(f"{len(starters.pairs)} pairs, {starters.label}, checkpoint {starters.checkpoint or 'base'}")
    return 0


def _nights(out: Path | None, arms: GlimmerArms, user: str) -> int:
    stops = demo_stops(user)
    if stops is None:
        print("her nights or checkpoints can't be read", file=sys.stderr)
        return 1
    answers = write_night_answers(out or demo_dir() / NIGHT_ANSWERS_FILE, arms, stops, serve_usd_per_hour=serve_rate(),
                                  now=datetime.now(timezone.utc))
    print(f"{len(answers.answers)} answers over {len(stops)} stops, {answers.label}")
    return 0


def _dailies(arms: GlimmerArms, user: str) -> int:
    nights = list(dict.fromkeys(c.night for c in checkpoint_registry().history(user)))
    lines = None
    for night in nights:
        lines = line_for(user, night, arms) or lines
    print(f"{0 if lines is None else len(lines.pairs)} line pairs over {len(nights)} promoted nights, {daily_lines.LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
