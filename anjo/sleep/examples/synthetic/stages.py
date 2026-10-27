"""The seven steps of the example, each a function over one ``Night`` (spec Task 15).

The order is the order of a live night (Task 5): evidence is extracted and folded into the brief
before her chooser looks at her state, and the probes are sealed before any training input exists.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.extract import Extraction, extract, ingest
from anjo.sleep.examples.synthetic import persona as people
from anjo.sleep.examples.synthetic.report import (
    ScoreRow,
    Say,
    format_receipt,
    format_table,
    score_row,
)
from anjo.sleep.examples.synthetic.services import ExampleError, Services
from anjo.sleep.examples.synthetic.world import RECALL, LatestTrained, World
from anjo.sleep.mind import receipt, snapshot
from anjo.sleep.mind.policy import PUBLIC_DEFAULT
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.run import MinimalPractice
from anjo.sleep.probes.arms import Arm, ArmDeps, ArmReply, reply
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.probes.generate import generate_day_probes, seal_probes
from anjo.sleep.probes.score import MuseJudge, score
from anjo.sleep.context.training import TrainingContext
from anjo.sleep.serving import config
from anjo.sleep.serving.client import ServingError
from anjo.sleep.train.dataset import Dataset, build_night_dataset
from anjo.sleep.train.lora import LoraCfg, fresh_adapter_name, train_lora_with_receipt
from anjo.sleep.types import (
    CheckpointRef,
    ChoiceContext,
    Evidence,
    NightAllowance,
    Probe,
)

SEED: Final = 20261028
TOKEN_CAP: Final = 60_000
EXTRACT_TIMEOUT_S: Final = 120.0
NIGHT_CLOCK: Final = time(22, 0)
CHOOSE_CLOCK: Final = time(23, 0)
STUDY_ACTIONS: Final = frozenset({"prepare_commitment", "revisit_thread"})
CHAT_TEMPERATURE: Final = 0.7
COMPARISONS: Final = (("3", "1"), ("3", "3x"), ("4", "2C"), ("4", "2F"), ("4", "3"))
ARMS: Final[tuple[Arm, ...]] = ("1", "2C", "2F", "3", "4")
DONOR_PERSON: Final = "donor"
Input = Callable[[str], str]


@dataclass(frozen=True)
class Night:
    """One run's inputs: the person, her day, her stores, the services and where files go."""

    person: people.ExamplePersona
    turns: tuple[people.DayTurn, ...]
    world: World
    services: Services
    workdir: Path
    say: Say
    run_id: str

    @property
    def now(self) -> datetime:
        return datetime.combine(self.person.day, NIGHT_CLOCK, tzinfo=timezone.utc)

    @property
    def as_of(self) -> datetime:
        return datetime.combine(self.person.day, CHOOSE_CLOCK, tzinfo=timezone.utc)


def seal(night: Night) -> tuple[list[Probe], str]:
    """Step 2: write the probes, seal them and hash the file."""
    person = night.person
    written = generate_day_probes(
        people.visible_evidence(person), people.held_out_evidence(person), SEED,
        chat=night.services.spark, model=SPARK_OPEN,
    )
    probes = [*written, *people.new_work_probes(person)]
    digest = seal_probes(probes, night.workdir / "probes")
    kinds = sorted({probe.kind for probe in probes})
    night.say(f"sealed {len(probes)} probes ({', '.join(kinds)}); sha256 {digest}")
    return probes, digest


def _extract_chat(services: Services) -> Callable[[list[dict[str, Any]]], ChatResult]:
    def call(messages: list[dict[str, Any]]) -> ChatResult:
        return services.spark(SPARK_OPEN, messages, None, EXTRACT_TIMEOUT_S, temperature=0.0)

    return call


def extract_day(night: Night) -> tuple[Extraction, list[Evidence]]:
    """Step 3a: Spark Open reads the person's own turns; the items and their targets are stored."""
    world, person = night.world, night.person
    extraction = extract(
        person.person_id, person.day, people.turns_of(night.turns, person.person_id),
        imported=[], targets=world.targets.open_targets(night.now), known_ids=world.evidence.ids(),
        now=night.now, chat=_extract_chat(night.services),
    )
    fresh = ingest(extraction, world.evidence, world.targets)
    held = sum(1 for item in fresh if item.held_out)
    night.say(
        f"extracted {len(fresh)} items ({held} held out for the sealed tests); "
        f"{len(world.targets.open_targets(night.now))} open targets"
    )
    for refusal in extraction.refused:
        night.say(f"  left out: {refusal}")
    return extraction, fresh


def fold_brief(night: Night) -> None:
    """Step 3b: fold the day into the brief, the way a live night does (Task 5, step 6)."""
    world = night.world
    new = world.briefs.unfolded(world.evidence.stored(night.person.day))
    record = world.briefs.fold(
        night.person.day, new, at=night.now, hosted_in_training=False, chat=night.services.spark,
    )
    first = next((line for line in record.served.splitlines() if line.strip()), "")
    night.say(f"folded {len(new)} items into the brief ({len(record.served)} characters; {first})")


def choose(night: Night) -> dict[str, Any]:
    """Step 4: build her snapshot, derive wants, and let her chooser pick under the public policy."""
    world, services = night.world, night.services
    last = max((turn.at for turn in night.turns if turn.role == "user"), default=None)
    state = snapshot.build(
        world.targets.open_targets(night.as_of), world.targets.results(), night.as_of, last,
    )
    ctx = ChoiceContext(
        policy=PUBLIC_DEFAULT, allowance=NightAllowance(actions_left=1, calls_left=1, usd_left=1.0),
        allowed_actions=STUDY_ACTIONS,
    )
    choice = services.chooser.choose("away", services.chooser.derive(state), ctx)
    recorded = receipt.record(choice, None)
    for line in format_receipt(recorded, world.targets.targets()):
        night.say(line)
    night.say(
        "The example has no homework agent (that is Muse Wonder in the private app), so it runs "
        "no study on the target: the practice below draws on the whole day."
    )
    return recorded


def practise(night: Night) -> int:
    """Step 5a: wake Glimmer, run minimal practice and store its transcripts."""
    world, services, person = night.world, night.services, night.person
    if not services.serving.wake().ok:
        raise ExampleError("Glimmer's server didn't wake in time; check ANJO_SLEEP_MODAL_URL and try again")
    context = TrainingContext(world.training_briefs, world.recall, RECALL)
    practice = MinimalPractice(
        chat=services.spark, client=services.glimmer, snapshot=context.snapshot, seed=SEED,
    )
    done = practice.run(
        person.person_id, person.day, world.evidence.stored(person.day), None,
        at=night.now + timedelta(hours=1),
    )
    if not done.transcripts:
        raise ExampleError("practice wrote no usable scene; run again")
    world.transcripts.add(done.transcripts)
    world.evidence.add([item.evidence() for item in done.transcripts])
    night.say(f"practice: {len(done.transcripts)} scenes ({done.dropped_cued} dropped for a cue)")
    return len(done.transcripts)


def build_dataset(night: Night, probes: list[Probe]) -> Dataset:
    """Step 5b: the night's SFT dataset, decontaminated against the sealed probes."""
    person = night.person
    data, dropped = build_night_dataset(
        night.world.evidence, None, person.day, probes, OverlapDecontaminator(), TOKEN_CAP,
        transcripts=night.world.transcripts, seed=SEED,
    )
    if not data.examples:
        raise ExampleError("the dataset is empty; nothing to train on")
    night.say(f"dataset: {len(data.examples)} examples, {data.tokens} tokens, {dropped} dropped as probe matches")
    return data


def train(night: Night, data: Dataset) -> CheckpointRef:
    """Step 5c: train a LoRA on Modal and put it where the server loads adapters from."""
    person = night.person
    name = fresh_adapter_name(f"{person.person_id}\x1f{night.run_id}", 1, frozenset())  # new on every run
    try:
        ckpt, done = train_lora_with_receipt(
            None, data, LoraCfg(), night.services.gpu, person_id=person.person_id, night=person.day,
            name=name, workdir=night.workdir / "train", adapter_cache=night.workdir / "adapters",
            publish=night.services.serving.upload_adapter, revision=night.services.revision,
        )
    except config.ServeConfigError as error:
        raise ExampleError(f"the server's volume refused adapter {name}: {error}") from None
    night.say(
        f"trained adapter {ckpt.id}: {done.steps} steps, {done.gpu_minutes:.1f} GPU-minutes; "
        "not promoted (no gates ran)"
    )
    return ckpt


def ready(night: Night, ckpt: CheckpointRef) -> None:
    """Restart the server and wake it, so it loads the new adapter, then check that it serves it.
    A running container sees a new adapter only after its next start (the engine wakes Glimmer
    again before its gates for the same reason)."""
    serving = night.services.serving
    serving.sleep_now()
    if not serving.wake().ok:
        raise ExampleError("Glimmer's server didn't wake in time after the adapter upload; run again")
    try:
        night.services.glimmer.complete(ckpt, [{"role": "user", "content": "Reply with one word."}], temperature=0.0)
    except ServingError:
        raise ExampleError(
            f"the server doesn't serve adapter {ckpt.id} yet: Modal loads a new adapter only when the "
            "server container starts. Run `modal app stop anjo-sleep-serve`, deploy it again "
            f"(modal deploy anjo/sleep/serving/modal_app.py), then re-run with --adapter {ckpt.id}"
        ) from None
    night.say(f"serving {ckpt.id}")


def reused_ref(name: str | None, night: Night) -> CheckpointRef | None:
    """``--adapter``: an adapter already on the volume, checked before anything costs money."""
    if name is None:
        return None
    try:
        config.check_adapter_name(name)
    except config.ServeConfigError as error:
        raise ExampleError(f"--adapter: {error}") from None
    if not night.services.serving.has_adapter(name):
        raise ExampleError(f"--adapter: no adapter {name} is on the server's volume")
    person = night.person
    return CheckpointRef(
        id=name, person_id=person.person_id, night=person.day,
        uri=f"{config.ADAPTERS_VOLUME}/{name}", parent_id=None,
        rendering=night.services.glimmer.rendering,  # trained by this recipe, so under this rendering
    )


def donor_ref(name: str | None, night: Night) -> CheckpointRef | None:
    """Arm 3x's donor adapter, checked by name before anything runs; ``None`` without one."""
    if name is None:
        return None
    try:
        config.check_adapter_name(name)
    except config.ServeConfigError as error:
        raise ExampleError(f"--donor: {error}") from None
    return CheckpointRef(
        id=name, person_id=DONOR_PERSON, night=night.person.day,
        uri=f"{config.ADAPTERS_VOLUME}/{name}", parent_id=None,
        rendering=night.services.glimmer.rendering,  # trained by this recipe, so under this rendering
    )


def run_arms(
    night: Night, probes: list[Probe], ckpt: CheckpointRef, donor: CheckpointRef | None,
) -> tuple[tuple[str, ...], tuple[ScoreRow, ...]]:
    """Step 6: arms 1, 2C, 2F, 3 and 4 (and 3x with a donor) on the probes, then pairwise scores."""
    person, world = night.person, night.world
    trained: Mapping[str, CheckpointRef] = {person.person_id: ckpt}
    if donor is not None:
        trained = {**trained, DONOR_PERSON: donor}
    deps = ArmDeps(
        model=night.services.glimmer, adapters=LatestTrained(trained), contexts=world.contexts,
        evidence=lambda _user: world.evidence, logprobs=night.services.glimmer,
        donors={person.person_id: DONOR_PERSON} if donor is not None else {},
        until=person.day, run_seed=SEED,
    )
    arms: tuple[Arm, ...] = (*ARMS, "3x") if donor is not None else ARMS
    replies: dict[str, list[ArmReply]] = {
        arm: [reply(arm, person.person_id, probe, deps=deps) for probe in probes]
        for arm in arms
    }
    judge = MuseJudge(chat=night.services.judge, brief_for=lambda _person: people.judge_brief(person))
    rows = tuple(
        score_row(a, b, score(probes, replies[a], replies[b], judge))
        for a, b in COMPARISONS if a in replies and b in replies
    )
    for line in format_table(rows):
        night.say(line)
    if donor is None:
        night.say("arm 3x needs a donor: pass --donor with an adapter another run trained.")
    return arms, rows


def chat(night: Night, ckpt: CheckpointRef, enabled: bool, input_fn: Input) -> None:
    """Step 7: a chat with base and base plus adapter, side by side, until the input ends."""
    if not enabled:
        night.say("chat skipped (--no-chat).")
        return
    night.say("Type a message; an empty line or /quit ends. Each reply is shown from base, then base + adapter.")
    while True:
        try:
            line = input_fn("you> ").strip()
        except EOFError:
            night.say("(input closed)")
            return
        if not line or line == "/quit":
            return
        messages = [{"role": "user", "content": line}]
        for label, adapter in (("base", None), ("base + adapter", ckpt)):
            answer = night.services.glimmer.complete(adapter, messages, temperature=CHAT_TEMPERATURE)
            night.say(f"[{label}] {answer.text.strip()}")
