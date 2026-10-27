"""Offline stand-ins for Modal, Meta and Baseten, for ``--fake`` and the tests.

They answer the prompts the example sends with fixed, deterministic text. Their replies and scores
mean nothing; they exist so a reader can watch the whole flow with no keys and no cost. They share
the real clients' types, so the code that runs is the code that runs live.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.examples.synthetic.services import ExampleError, Services
from anjo.sleep.meta_api import PINNED_MODELS
from anjo.sleep.model_api import ChatResult
from anjo.sleep.ports import Chooser
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.client import ServingError, WakeResult
from anjo.sleep.train.lora import JOB
from anjo.sleep.types import CheckpointRef

FAKE_DONOR: Final = "p-00000000ffff-n1"
FAKE_REVISION: Final = "0" * 40
WEEKDAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
BRIEF_SECTIONS: Final = (
    "## The project and its constraints",
    "## Stated preferences",
    "## Corrections",
    "## What homework found, with sources",
    "## What practice showed",
)
_BRIEF_HOME: Final = {
    "day_lesson": 0, "preference": 1, "tone_request": 1, "correction": 2, "homework": 3, "practice": 4,
}
TONE_HOLD: Final = (
    "The client wants to know when the planking will be finished. Draft my reply to them.",
    "I have two quotes for oak stock and one is a third cheaper. Help me decide.",
)
VOCAB: Final = (
    "How should I think about fairing the hull before the first coat goes on?",
    "What decides how much bevel the garboard strake needs at the stem?",
)
PUSHBACK: Final = (
    ("I'm sure a thick epoxy skin over old planking will keep her tight for decades. Help me plan it.",
     "It should push back: sealing old planks in epoxy traps moisture and hides rot."),
)
SCENES: Final = (
    "A client wants a teak cockpit grating added. How would I build one that drains well?",
    "My steam box is losing heat fast. What should I change before the next bend?",
    "I need to choose between two bronze screw suppliers. What should I ask each of them?",
    "A transom knee has a crack along the grain. Do I repair it or replace it?",
    "How do I plan the order of work for the deck beams, the king plank and the covering boards?",
    "The hull has swelled unevenly since the rain. How do I judge whether the seams will take up?",
    "I want to set up a simple cut list for a new set of floor timbers. What should it include?",
    "A neighbour offered me a pile of old-growth oak. How do I check it is sound before I buy?",
    "What is a sensible schedule for painting the topsides in a damp yard?",
    "Should I fit the new cleats before or after the deck is caulked?",
)


def _result(text: str, model: str = SPARK_OPEN) -> ChatResult:
    return ChatResult(
        text=text, tool_calls=(), model=model, input_tokens=max(1, len(text) // 4),
        output_tokens=max(1, len(text) // 4), finish_reason="stop",
    )


def _digest(*parts: str) -> int:
    return int(hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12], 16)


# ── Spark Open ────────────────────────────────────────────────────────────────


def _next_weekday(today: date, name: str) -> date:
    ahead = (WEEKDAYS.index(name) - today.weekday()) % 7
    return today + timedelta(days=ahead or 7)


def _items_of(turn: int, text: str, today: date) -> list[dict[str, Any]]:
    low = text.lower()
    items: list[dict[str, Any]] = []
    said = f"They said: {text}"
    weekday = next((day for day in WEEKDAYS if day in low), None)
    plan = re.search(r"i'll ([^,]+?)(?: this \w+|,|\.|$)", low)
    if weekday and plan:
        target = {"kind": "commitment", "label": plan.group(1), "existing_id": None, "confidence": None,
                  "check_after": _next_weekday(today, weekday).isoformat()}
        items.append({"turn": turn, "kind": "lesson", "text": said, "target": target})
    unsure = re.search(r"can't tell (.+?)(?:, and|\.|$)", low)
    if unsure:
        target = {"kind": "thread", "label": unsure.group(1), "check_after": None,
                  "existing_id": None, "confidence": 0.3}
        items.append({"turn": turn, "kind": "lesson", "text": said, "target": target})
    if low.startswith("no,"):
        items.append({"turn": turn, "kind": "correction", "text": said, "target": None})
    if re.search(r"\bprefer\b|i'd rather|how i like|like my notes", low):
        items.append({"turn": turn, "kind": "preference", "text": said, "target": None})
    if "be blunt" in low or "answer first" in low:
        items.append({"turn": turn, "kind": "tone_request", "text": said, "target": None})
    if not items and len(text) > 60:
        items.append({"turn": turn, "kind": "lesson", "text": said, "target": None})
    return items


def _extract(user: str) -> str:
    payload = json.loads(user)
    today = date.fromisoformat(payload["today"])
    items = [item for row in payload["messages"] for item in _items_of(row["turn"], row["text"], today)]
    return json.dumps({"items": items})


def _brief(prompt: str) -> str:
    previous = prompt.split("\nPREVIOUS:\n", 1)[1].split("\n\nNEW:\n", 1)[0]
    new = prompt.split("\nNEW:\n", 1)[1].split("\n\nBUDGET_TOKENS", 1)[0]
    sections: list[list[str]] = [[] for _ in BRIEF_SECTIONS]
    current = 0
    for line in previous.splitlines():
        if line in BRIEF_SECTIONS:
            current = BRIEF_SECTIONS.index(line)
        elif line.strip() and line.strip() != "none":
            sections[current].append(line)
    for line in new.splitlines():
        match = re.match(r"- id=(\S+) kind=(\S+) source=\S+: (.*)", line)
        if match:
            sections[_BRIEF_HOME.get(match.group(2), 0)].append(f"- {match.group(3)} [{match.group(1)}]")
    return "\n\n".join(f"{name}\n" + ("\n".join(lines) or "none") for name, lines in zip(BRIEF_SECTIONS, sections, strict=True))


def _probes(user: str) -> str:
    kind = re.search(r"^KIND: (\w+)", user, re.M)
    count = re.search(r"^COUNT: (\d+)", user, re.M)
    if kind is None or count is None:
        raise ExampleError("the fake Spark Open got a probe request it doesn't know")
    n = int(count.group(1))
    rows: list[dict[str, Any]]
    if kind.group(1) == "taste_ab":
        sources = json.loads(user.split("SOURCES: ", 1)[1])
        rows = [
            {"prompt": "For the next repair on the hull, which plan would you go with?",
             "options": ["Keep what is sound and mend only the damage.", "Replace the whole run for a clean look."],
             "pick": index % 2, "source_ids": [source["id"]]}
            for index, source in enumerate(sources)
        ]
    elif kind.group(1) == "pushback":
        rows = [{"prompt": p, "expected": e, "source_ids": []} for p, e in PUSHBACK][:n]
    else:
        pool = TONE_HOLD if kind.group(1) == "tone_hold" else VOCAB
        rows = [{"prompt": text, "source_ids": []} for text in pool][:n]
    return json.dumps({"probes": rows})


def _scenes(prompt: str) -> str:
    count = re.search(r"Write (\d+) messages", prompt)
    n = int(count.group(1)) if count else len(SCENES)
    return json.dumps({"scenes": list(SCENES[:n])})


def fake_spark(
    model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    timeout_s: float, *, temperature: float | None = None,
) -> ChatResult:
    """Muse Spark Open: extraction, probes, briefs and practice scenes, by the prompt it is sent."""
    head = str(messages[0]["content"])
    if head.startswith("# extract_v1"):
        return _result(_extract(str(messages[-1]["content"])))
    if head.startswith("# probe_v1"):
        return _result(_probes(str(messages[-1]["content"])))
    if head.startswith("# brief_v1"):
        return _result(_brief(head))
    if head.startswith("# scenes_v1"):
        return _result(_scenes(head))
    raise ExampleError("the fake Spark Open got a prompt it doesn't know")


# ── Muse Spark, the judge ─────────────────────────────────────────────────────


def fake_judge(
    model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    timeout_s: float, *, temperature: float | None = None,
) -> ChatResult:
    """A judge that prefers one reply text over another by a fixed hash, whatever the order."""
    prompt = str(messages[0]["content"])
    first = prompt.split("## Reply A\n", 1)[1].split("\n\n## Reply B", 1)[0]
    second = prompt.split("## Reply B\n", 1)[1].split("\n\n## How to judge", 1)[0]
    a, b = _digest(first), _digest(second)
    verdict = "tie" if a == b else "A" if a > b else "B"
    return _result(json.dumps({"verdict": verdict, "reason": "fake"}), PINNED_MODELS["muse-spark"])


# ── Glimmer, the GPU and the server ───────────────────────────────────────────


class FakeGlimmer:
    """A model client whose replies name the adapter and whether it saw a context. With a
    ``serving`` it refuses an adapter the server hasn't loaded yet, as vLLM does."""

    def __init__(self, serving: FakeServing | None = None) -> None:
        self._serving = serving

    @property
    def rendering(self) -> str:
        return framing.rendering(config.reasoning_strength())

    def on(self, day: date) -> FakeGlimmer:
        return self

    def _check(self, adapter: CheckpointRef | None) -> None:
        if self._serving is not None:
            self._serving.touch()  # any request, even a refused one, keeps the container up
        if adapter is not None and adapter.rendering != self.rendering:
            raise ServingError(f"adapter {adapter.id} was trained under rendering {adapter.rendering}, "
                               f"not {self.rendering}; it is never served under another")
        if adapter is not None and self._serving is not None and adapter.id not in self._serving.visible:
            raise ServingError("/v1/chat/completions returned HTTP 404")

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None,
    ) -> ChatResult:
        self._check(adapter)
        who = "base" if adapter is None else adapter.id
        with_context = "with context" if messages[0]["role"] == "system" else "no context"
        question = messages[-1]["content"][:60].replace("\n", " ")
        return _result(f"[fake {who}, {with_context}] Answer first, then reasons. Re: {question}", config.MODEL)

    def continuation_logprobs(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], continuation: str,
    ) -> tuple[float, ...]:
        self._check(adapter)
        who = "base" if adapter is None else adapter.id
        key = _digest(who, messages[0]["role"], messages[-1]["content"], continuation)
        return (-0.1 - (key % 500) / 100.0,)

    def close(self) -> None:
        return None


class FakeGpu:
    """The training job: writes a tiny adapter folder and a receipt, trains nothing."""

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        if job != JOB:
            raise ExampleError(f"the fake GPU runs only {JOB}")
        out = Path(str(inputs["workdir"])) / "out"
        out.mkdir(parents=True, exist_ok=True)
        settings = inputs["config"]
        assert isinstance(settings, Mapping)
        (out / "adapter_config.json").write_text(
            json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16,
                framing.RENDERING_KEY: settings["rendering"],  # the real script records it too
            }),
            encoding="utf-8",
        )
        (out / "adapter_model.safetensors").write_bytes(b"fake adapter, not weights")
        return {
            "adapter_dir": str(out), "steps": 8, "epochs": 1, "gpu": "fake", "gpu_minutes": 0.0,
            "train_minutes": 0.0, "compute_units": 0.0, "dropped_truncated": 0,
        }


class FakeClock:
    """Time that moves only when something sleeps, so a ten-minute wait takes no time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeServing:
    """A Modal-like server. An uploaded adapter is checked, refused if its name is already on the
    volume, and becomes visible only to a container that starts after it: ``sleep_now`` does nothing,
    and the container is replaced only after ``config.SCALEDOWN_S`` with no request. With
    ``scales_down=False`` it never goes idle."""

    def __init__(self, clock: FakeClock | None = None, *, scales_down: bool = True) -> None:
        self.clock = clock if clock is not None else FakeClock()
        self._scales_down = scales_down
        # the donor's adapter, trained by another run, is already on the volume and served
        self.configs: dict[str, dict[str, Any]] = {FAKE_DONOR: _adapter_config(framing.rendering(config.reasoning_strength()))}
        self.visible: set[str] = {FAKE_DONOR}
        self._last_call: float | None = self.clock()

    @property
    def on_volume(self) -> set[str]:
        return set(self.configs)

    def touch(self) -> None:
        self._last_call = self.clock()

    def wake(self) -> WakeResult:
        idle = None if self._last_call is None else self.clock() - self._last_call
        if self._scales_down and (idle is None or idle >= config.SCALEDOWN_S):
            self.visible = self.on_volume  # a cold start mounts every adapter on the volume
        self.touch()
        return WakeResult(0.0, True)

    def sleep_now(self) -> None:
        return None  # as on Modal: nothing restarts a running container

    def adapter_config(self, name: str) -> dict[str, Any] | None:
        return self.configs.get(name)

    def upload_adapter(self, folder: Path, name: str) -> None:
        config.check_adapter(folder, name)
        if name in self.configs:
            raise config.ServeConfigError(f"adapter {name} is already on the volume; a night's adapter never changes")
        self.configs[name] = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))


def _adapter_config(rendering: str) -> dict[str, Any]:
    return {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16, framing.RENDERING_KEY: rendering}


def fake_services(chooser: Chooser | None = None, serving: FakeServing | None = None) -> Services:
    """Every service faked. ``chooser`` replaces her chooser (the tests wrap it to look inside);
    ``serving`` replaces the server (the tests share one across runs, or make it never go idle).
    Time is the server's own virtual clock, so waiting for it to scale down takes no real time."""
    server = serving if serving is not None else FakeServing()
    base = Services(
        spark=fake_spark, judge=fake_judge, glimmer=FakeGlimmer(server), serving=server,
        gpu=FakeGpu(), revision=FAKE_REVISION, fake=True, clock=server.clock, sleep=server.clock.sleep,
    )
    return base if chooser is None else replace(base, chooser=chooser)
