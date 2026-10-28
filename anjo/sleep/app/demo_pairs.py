"""Pre-generate the live page's starter pairs (Task 13a):

    python -m anjo.sleep.app.demo_pairs [--out PATH] [--wake] [--nights]

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

Glimmer must already be up, or ``--wake`` wakes her for this run, which then ends asleep. Every
keep-warm window needs Kevin's OK, so nothing here wakes her unasked. Each side's cost is its
measured seconds at the serving GPU's list price (``ANJO_SLEEP_PRICES``), or unmeasured without it.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.demo_live import (
    DEMO_PERSONA, STARTERS_FILE, USER_ENV, GlimmerArms, Warmth, baseline_from, demo_dir, glimmer_deps, glimmer_status,
    serve_rate,
)
from anjo.sleep.app.demo_mind import NIGHT_ANSWERS_FILE, demo_stops
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import checkpoint_registry
from anjo.sleep.demo import night_answers
from anjo.sleep.demo.night_answers import NightAnswers
from anjo.sleep.demo.service import Answer, LiveArms, StopArms
from anjo.sleep.demo.slider import Stop
from anjo.sleep.demo.starters import Side, StarterSet, generate, to_json
from anjo.sleep.demo.words import SCENES, Scene
from anjo.sleep.serving import backend
from anjo.sleep.types import CheckpointRef


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.app.demo_pairs", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="where to write (default: the demo's folder)")
    parser.add_argument("--wake", action="store_true", help="wake Glimmer for this run; it ends asleep")
    parser.add_argument("--nights", action="store_true", help="the night slider's answers, notes off, per stop")
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


if __name__ == "__main__":
    raise SystemExit(main())
