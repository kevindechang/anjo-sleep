"""``python -m anjo.sleep.examples.synthetic``: one invented person, one fixed day, one night.

From the repo root, with your own keys::

    python -m anjo.sleep.examples.synthetic --persona persona.json --day day1.jsonl

or with no keys at all, on fakes::

    python -m anjo.sleep.examples.synthetic --fake

The steps are the spec's seven (Task 15): load into plain files; write and hash the probes;
extract the day on Spark Open and fold it into the brief; let her chooser pick the night's target
under the public default policy and print the receipt; practise, build the dataset and train a
LoRA on Modal; run the arms on the probes and print the score table; serve base plus adapter on
vLLM and open a chat.

**No crisis gate.** The example has none, because Anjo's case lists stay private. It says so on
every run, runs no gate of any kind, and never marks a checkpoint as promoted.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.examples.synthetic import persona as people
from anjo.sleep.examples.synthetic import stages
from anjo.sleep.examples.synthetic.report import (
    FAKE_NOTICE,
    LIVE_NOTICE,
    NO_CRISIS_GATE,
    ExampleReport,
    Say,
    heading,
)
from anjo.sleep.examples.synthetic.services import ExampleError, Services
from anjo.sleep.examples.synthetic.world import open_world

__all__ = ["ExampleError", "ExampleReport", "execute", "main"]

EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2


@dataclass(frozen=True)
class Options:
    persona: Path
    day: Path
    workdir: Path
    fake: bool
    chat: bool
    donor: str | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anjo.sleep.examples.synthetic",
        description="One invented person, one fixed day, one night of Anjo Sleep. No crisis gate.",
    )
    parser.add_argument("--persona", default="persona.json", help="the persona file (default: the bundled one)")
    parser.add_argument("--day", default="day1.jsonl", help="the day's conversation (default: the bundled one)")
    parser.add_argument("--fake", action="store_true", help="run offline on fakes: no keys, no cost")
    parser.add_argument("--workdir", help="a new folder for the run's files (default: a temp folder)")
    parser.add_argument("--donor", help="arm 3x: an adapter another run trained (p-<hash>-n<night>)")
    parser.add_argument("--no-chat", action="store_true", help="skip the chat at the end")
    return parser


def parse(argv: Sequence[str] | None) -> Options:
    args = _parser().parse_args(argv)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="anjo-sleep-example-"))
    return Options(
        persona=people.resolve(args.persona), day=people.resolve(args.day), workdir=workdir,
        fake=args.fake, chat=not args.no_chat, donor=args.donor,
    )


def _fresh(workdir: Path) -> None:
    if workdir.exists() and any(workdir.iterdir()):
        raise ExampleError(f"{workdir} already holds files; give --workdir a new folder")
    workdir.mkdir(parents=True, exist_ok=True)


def execute(
    argv: Sequence[str] | None, services: Services, *, say: Say = print,
    input_fn: stages.Input = input,
) -> ExampleReport:
    """Run the night with ``services`` and return what it did."""
    options = parse(argv)
    say(NO_CRISIS_GATE)
    say(FAKE_NOTICE if services.fake else LIVE_NOTICE)
    try:
        person = people.load_persona(options.persona)
        day = people.load_day(options.day, person.day)
        _fresh(options.workdir)
        return _night(options, services, person, day, say, input_fn)
    except people.ExampleInputError as error:
        raise ExampleError(str(error)) from None
    finally:
        services.glimmer.close()


def _night(
    options: Options, services: Services, person: people.ExamplePersona,
    day: tuple[people.DayTurn, ...], say: Say, input_fn: stages.Input,
) -> ExampleReport:
    night = stages.Night(
        person=person, turns=day, world=open_world(options.workdir, person.person_id, person.day),
        services=services, workdir=options.workdir, say=say,
    )
    donor = stages.donor_ref(options.donor, night)  # a bad name stops the run before it costs anything
    done: list[str] = []

    def step(name: str) -> None:
        done.append(name)
        heading(say, name)

    step("load")
    say(f"{person.name}, {person.age}, {person.field}; {len(day)} turns on {person.day} (invented)")
    step("probes")
    probes, digest = stages.seal(night)
    step("extract")
    _, fresh = stages.extract_day(night)
    done.append("fold")
    stages.fold_brief(night)
    step("choose")
    receipt = stages.choose(night)
    step("practice")
    stages.practise(night)
    done.append("dataset")
    data = stages.build_dataset(night, probes)
    done.append("train")
    ckpt = stages.train(night, data)
    step("arms")
    arms, table = stages.run_arms(night, probes, ckpt, donor)
    step("chat")
    stages.chat(night, ckpt, options.chat, input_fn)
    say("")
    say(f"Done. Adapter {ckpt.id} is trained, not promoted: no crisis gate or other gate ran.")
    return ExampleReport(
        stages=tuple(done), probe_sha256=digest, extracted_items=len(fresh), receipt=receipt,
        checkpoint=ckpt, dataset_examples=len(data.examples), arms=arms, table=table,
    )


def main(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
    input_fn: stages.Input = input, say: Say = print,
) -> int:
    """The command line. Exit 0 on a finished run, 2 when a key is missing, 1 on any other stop."""
    from anjo.sleep.examples.synthetic import fakes, live

    try:
        fake = _parser().parse_args(argv).fake
        services = fakes.fake_services() if fake else live.build_services(environ)
        execute(argv, services, say=say, input_fn=input_fn)
    except live.MissingKeys as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except ExampleError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED
    return 0
