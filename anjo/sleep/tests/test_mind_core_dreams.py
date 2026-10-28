# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The sleep objective's dreams from MIND's core (``night/core_dreams.py``): with the dreams term on,
the dataset step asks her Glimmer to speak from the states tonight's dream kept, each thread written
from the real evidence it rests on, and records how many candidates the dream filter kept. With the
term off the dataset is today's; with the core off the dreams are not wired. Through the thought
channel (``ANJO_SLEEP_DREAMS_CHANNEL=1``, the channel on), each reply is made with its imagined
state as soft tokens, and its row carries that state's snapshot. A fake core and a fake Glimmer: no
model, no network."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.channel_night import MindReading
from anjo.sleep.night.core_dreams import DREAMS_CHANNEL_ENV, core_dreams, dreams_channel
from anjo.sleep.night.core_snapshot import CoreSnapshot, state_sha
from anjo.sleep.night.deps import Dreamt, MindCall, NoCore
from anjo.sleep.night.dreams import HYPOTHETICAL, SPEAK, GlimmerDreams, NoDreams, Terms
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.night_fakes import NIGHT, World, own, step, window
from anjo.sleep.tests.test_channel_night import SoftGlimmer, _projector, _setup
from anjo.sleep.tests.test_thought_channel import _predictions, _state
from anjo.sleep.train import terms
from anjo.sleep.types import Evidence, NightResult

UID = "p1"
STATE = "a" * 16
PROBE = "How would you price the film-plastics pilot?"  # the world's sealed probe (night_fakes.Registrar)
DREAMS_ON = Terms(seed=True, dreams=True)


@dataclass
class DreamingCore:
    """Dreams one boundary on her first two items, a picked state on her first, and one on an item
    she never had; keeps each night's dream for ``dreamt``."""

    people: frozenset[str] = frozenset({UID})
    tonight: dict[tuple[str, date], Dreamt] = field(default_factory=dict)
    states: tuple[str, str, str] = ("b1", "d1", "d2")  # the boundary's and the two picked states' ids

    def serves(self, user_id: str) -> bool:
        return user_id in self.people

    def dream(self, call: MindCall, evidence: Sequence[Evidence]) -> Dreamt:
        ids = [item.id for item in evidence]
        b1, d1, d2 = self.states
        receipt = {
            "state": STATE,
            "boundaries": [{"state": b1, "threads": [1], "rests_on": ids[:2], "steps": 2, "speak": True, "slot": 1}],
            "dreams": [{"state": d1, "threads": [2], "rests_on": ids[:1]},
                       {"state": d2, "threads": [3], "rests_on": ["not-hers"]}],
        }
        done = Dreamt(state_id=STATE, receipt=receipt)
        self.tonight[(call.user_id, call.night)] = done
        return done

    def dreamt(self, user_id: str, night: date) -> Dreamt | None:
        return self.tonight[(user_id, night)] if self.serves(user_id) else None

    def reading(self, user_id: str, night: date) -> MindReading | None:
        return None

    def imagined(self, user_id: str, night: date, state_ref: str) -> MindReading | None:
        return None

    def keep(self, call: MindCall, dreamt: Dreamt, *, lora: str | None, projector: str | None) -> dict[str, object]:
        return {"state": dreamt.state_id, "lora": lora}


IMAGINED = {state_sha(state): state for state in (_state(11, person=UID), _state(12, person=UID), _state(13, person=UID))}


@dataclass
class ImaginingCore(DreamingCore):
    """Keeps real states under its dream's ids, and reads each one back for the channel."""

    states: tuple[str, str, str] = tuple(IMAGINED)  # type: ignore[assignment]
    asked: list[str] = field(default_factory=list)

    def imagined(self, user_id: str, night: date, state_ref: str) -> MindReading | None:
        if not self.serves(user_id):
            return None
        self.asked.append(state_ref)
        state = IMAGINED[state_ref]
        return MindReading(state, state_ref, _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0)), None)


class SoftReplies(SoftGlimmer):
    """The channel's client, keeping each reply's messages too."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None, soft: Any = None) -> Any:
        self.messages.append(messages)
        return super().complete(adapter, messages, temperature=temperature, seed=seed, max_tokens=max_tokens, soft=soft)


class Glimmer:
    """Her Glimmer for the dreams: ``text n`` for the n-th call; the ``echo``-th call repeats the sealed probe."""

    def __init__(self, echo: int | None = None) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.echo = echo

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None) -> ChatResult:
        self.calls.append(messages)
        n = len(self.calls)
        text = f"Fine. {PROBE}" if n == self.echo else f"text {n}"
        return ChatResult(text=text, tool_calls=(), model=GLIMMER, input_tokens=10, output_tokens=5, finish_reason="stop")


@dataclass
class Rows:
    """The trainer, keeping the dream rows each night's dataset carried."""

    world: World
    dreams: list[int] = field(default_factory=list)
    rows: list[Any] = field(default_factory=list)

    def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
        self.dreams.append(len(data.dreams))
        self.rows.extend(data.dreams)
        return self.world.trainer.train(user_id, night, prev, data, name)


def _night(world: World, **deps: Any) -> NightResult:
    return run_night_batch_with(world.deps(**deps), {UID: NIGHT}, windows={UID: window(NIGHT)})[UID]


def test_dreams_wired_from_core_seeds(tmp_path: Path) -> None:
    # Arrange: her day, the core's dream over it, her Glimmer echoing a sealed probe once
    world, core, glimmer = World(tmp_path), DreamingCore(), Glimmer(echo=4)
    world.plan(UID, NIGHT)
    rows = Rows(world)
    dreams = core_dreams(core, world.evidence, glimmer, seed=7)

    # Act
    result = _night(world, core=core, trainer=rows, terms=DREAMS_ON, dreams=dreams)

    # Assert: a boundary on two items and a picked state on one, three candidates each; the state
    # on an item she never had seeds nothing
    assert result.status == "promoted" and isinstance(dreams, GlimmerDreams)
    first, second = own(UID, NIGHT, 0).text, own(UID, NIGHT, 1).text
    asked = [messages[0]["content"] for messages in glimmer.calls[::2]]
    assert len(glimmer.calls) == 12 and all(f"- {first}" in text for text in asked)
    assert sum(f"- {second}\n{HYPOTHETICAL}: {SPEAK}" in text for text in asked) == 3  # the boundary's imagined branch
    assert glimmer.calls[1][0] == {"role": "system", "content": f"- {first}\n- {second}\n{HYPOTHETICAL}: {SPEAK}"}
    # The dataset step records the dream filter's keep rate: the reply echoing the probe never trains
    receipt = step(result.log_path, "dataset")["terms"]
    assert (receipt["dream_candidates"], receipt["dreams"], receipt["dreams_dropped"]["probe"]) == (6, 5, 1)
    assert receipt["dreams_keep_rate"] == round(5 / 6, 4) and receipt["anchors"] > 0
    assert receipt["dreams_by_seed"] == {"boundary": 2, "picked": 3, "random": 0}
    assert rows.dreams == [5]


def test_dreams_off_is_todays_dataset(tmp_path: Path) -> None:
    # The dreams wired but the term off: no call to her Glimmer, no terms, the dataset as without them
    plain_world, wired_world = World(tmp_path / "plain"), World(tmp_path / "wired")
    for world in (plain_world, wired_world):
        world.plan(UID, NIGHT)
    glimmer = Glimmer()
    core = DreamingCore()
    plain = _night(plain_world, core=DreamingCore())
    wired = _night(wired_world, core=core, dreams=core_dreams(core, wired_world.evidence, glimmer, seed=7))
    assert plain.status == wired.status == "promoted" and glimmer.calls == []
    assert step(wired.log_path, "dataset") == step(plain.log_path, "dataset")
    assert "terms" not in step(wired.log_path, "dataset")


def test_without_the_core_the_dreams_are_not_wired(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    dreams = core_dreams(NoCore(), world.evidence, Glimmer(), seed=7)
    assert isinstance(dreams, NoDreams)
    result = _night(world, terms=DREAMS_ON, dreams=dreams)
    receipt = step(result.log_path, "dataset")["terms"]
    assert receipt["dreams"] == terms.DREAMS_NOT_WIRED and "dreams_keep_rate" not in receipt


def test_a_dream_never_reads_an_item_that_may_not_train_and_none_for_anyone_else(tmp_path: Path) -> None:
    world, core = World(tmp_path), DreamingCore()
    world.planned[(UID, NIGHT)] = [replace(own(UID, NIGHT, 0), held_out=True), own(UID, NIGHT, 1),
                                   replace(own(UID, NIGHT, 2), ancestry=None)]
    glimmer = Glimmer()
    result = _night(world, core=core, terms=DREAMS_ON, dreams=core_dreams(core, world.evidence, glimmer, seed=7))
    told = json.dumps([messages for messages in glimmer.calls])
    assert own(UID, NIGHT, 0).text not in told and own(UID, NIGHT, 2).text not in told
    assert step(result.log_path, "dataset")["terms"]["dream_candidates"] == len(glimmer.calls) // 2
    other = core_dreams(core, world.evidence, glimmer, seed=7)
    assert other.dream("someone-else", NIGHT, None) == ()


@pytest.mark.parametrize("arm", ["state", "zero"])
def test_dreams_through_the_channel_reply_from_each_imagined_state_and_carry_its_snapshot(
    tmp_path: Path, arm: str,
) -> None:
    # Arrange: the channel on, and the dreams through it
    world, core, glimmer, soft = World(tmp_path), ImaginingCore(), Glimmer(), SoftReplies()
    world.plan(UID, NIGHT)
    rows = Rows(world)
    channel = _setup(world, soft, _projector(tmp_path / "shared.projector.npz"), arm)
    dreams = core_dreams(core, world.evidence, glimmer, seed=7, channel=channel)

    # Act
    result = _night(world, core=core, trainer=rows, terms=DREAMS_ON, dreams=dreams)

    # Assert: her Glimmer wrote each message as before, plain; every reply came through the channel,
    # read from its own imagined state, and its row carries that state's snapshot in the night's arm
    assert result.status == "promoted"
    b1, d1, _ = core.states
    assert len(glimmer.calls) == 6 and all(len(messages) == 1 for messages in glimmer.calls)
    assert len(soft.soft) == 6 and all(soft.soft) and core.asked == [b1, d1]
    assert soft.messages[0][0]["content"].endswith(f"{HYPOTHETICAL}: {SPEAK}")  # the seed stays her system turn
    snaps = [CoreSnapshot.from_json(json.loads(row.core)) for row in rows.rows]
    assert len(snaps) == 6 and [s.state_sha for s in snaps] == [b1] * 3 + [d1] * 3
    assert {s.arm for s in snaps} == {arm} and all(s.core_model == "core-abc" for s in snaps)
    assert step(result.log_path, "dataset")["terms"]["dreams"] == step(result.log_path, "dataset")["terms"]["dreams_in_channel"] == 6


def test_dreams_without_the_channel_carry_no_state(tmp_path: Path) -> None:
    world, core, glimmer = World(tmp_path), ImaginingCore(), Glimmer()
    world.plan(UID, NIGHT)
    rows = Rows(world)
    result = _night(world, core=core, trainer=rows, terms=DREAMS_ON, dreams=core_dreams(core, world.evidence, glimmer, seed=7))
    assert result.status == "promoted" and core.asked == [] and len(glimmer.calls) == 12
    assert rows.rows and all(row.core is None for row in rows.rows)
    assert "dreams_in_channel" not in step(result.log_path, "dataset")["terms"]


def test_a_dream_state_the_core_cant_read_stops_the_dreams(tmp_path: Path) -> None:
    world, core = World(tmp_path), DreamingCore()  # it reads back no imagined state
    world.plan(UID, NIGHT)
    channel = _setup(world, SoftReplies(), _projector(tmp_path / "shared.projector.npz"))
    dreams = core_dreams(core, world.evidence, Glimmer(), seed=7, channel=channel)
    assert _night(world, core=core, terms=DREAMS_ON, dreams=dreams).status == "failed"


def test_the_switch_puts_the_dreams_through_the_channel_only_when_it_is_on(tmp_path: Path) -> None:
    channel = _setup(World(tmp_path), SoftReplies(), tmp_path / "shared.projector.npz")
    assert dreams_channel({}, channel) is None and dreams_channel({DREAMS_CHANNEL_ENV: "0"}, channel) is None
    assert dreams_channel({DREAMS_CHANNEL_ENV: "1"}, channel) is channel
    with pytest.raises(ValueError, match="channel"):
        dreams_channel({DREAMS_CHANNEL_ENV: "1"}, None)
    with pytest.raises(ValueError, match=DREAMS_CHANNEL_ENV):
        dreams_channel({DREAMS_CHANNEL_ENV: "yes"}, channel)
