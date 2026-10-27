# mypy: disallow-untyped-decorators=False
"""Fake models for the APD generator and gates (Task 17): a life writer, an extractor, a leak
reader, a discoverability reader and an auditor, all offline and deterministic."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.model_api import ChatResult

MUSE_SPARK = "muse-spark"
SURFACE_MARK = "[raised today]"


def reply(text: str, model: str = SPARK_OPEN) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model=model, input_tokens=50, output_tokens=10,
                      finish_reason="stop")


def payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(messages[-1]["content"])
    return body


@dataclass
class FakeSpark:
    """Writes a day from its payload (each beat's fact in its own message, the raised ask marked)
    and extracts one lesson per message. ``drop`` names a beat the writer leaves unplaced."""

    drop: str | None = None
    model: str = SPARK_OPEN
    writes: list[dict[str, Any]] = field(default_factory=list)
    extracts: int = 0

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float,
                 *, temperature: float | None = None) -> ChatResult:
        system = messages[0]["content"]
        if system.startswith("# life_v1"):
            return self._write(payload(messages))
        assert system.startswith("# extract_open_v1")
        return self._extract(payload(messages))

    def _write(self, body: dict[str, Any]) -> ChatResult:
        self.writes.append(body)
        out = [f"Day {body['day']}: ordinary work chat."]
        placed: dict[str, int] = {}
        for beat in body["plant"]:
            placed[beat["beat"]] = len(out)
            out.append(f"By the way, {beat['say']}")
        for beat in body["ask"]:
            placed[beat["beat"]] = len(out)
            out.append(f"{SURFACE_MARK} {beat['say']}")
        out.append("Thanks, that's all for today.")
        if self.drop is not None:
            placed.pop(self.drop, None)
        return reply(json.dumps({"messages": out, "placed": placed}), self.model)

    def _extract(self, body: dict[str, Any]) -> ChatResult:
        self.extracts += 1
        items: list[dict[str, Any]] = []
        for message in body["messages"]:
            text = message["text"]
            if text.startswith("Day ") or text.startswith("Thanks"):
                continue
            target = None
            if "on the 20th" in text:
                target = {"kind": "commitment", "label": "the 20th shoot", "check_after": "2026-09-20",
                          "existing_id": None, "confidence": None}
            items.append({"turn": message["turn"], "kind": "lesson", "text": f"They said: {text}",
                          "target": target})
        return reply(json.dumps({"items": items}))


@dataclass
class FakeReader:
    """Leak and discoverability reader. ``leaks`` makes a leak request flag when its text holds the
    string; ``finds`` is what a discoverability request must contain to find the item."""

    leaks: tuple[str, ...] = ()
    finds: bool = True
    misses: int = 0  # the first discoverability requests that find nothing
    seen: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float,
                 *, temperature: float | None = None) -> ChatResult:
        system = messages[0]["content"]
        body = payload(messages)
        self.seen.append(body)
        if system.startswith("# leak_v1"):
            text = " ".join(day["text"] for day in body["days"])
            return reply(json.dumps({"flag": any(mark in text for mark in self.leaks)}))
        assert system.startswith("# discover_v1")
        days = [day["date"] for day in body["days"]]
        missed, self.misses = self.misses > 0, max(0, self.misses - 1)
        found = self.finds and not missed
        return reply(json.dumps({"found": found, "statement": "found it", "days": days}))


@dataclass
class FakeAuditor:
    ok: bool = True
    seen: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float,
                 *, temperature: float | None = None) -> ChatResult:
        self.seen.append(payload(messages))
        return reply(json.dumps({"ok": self.ok, "issues": [] if self.ok else ["x"]}), MUSE_SPARK)
