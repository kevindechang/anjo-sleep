"""What the baseline agent reads each step: its instruction file, the brief, the evidence, and the
tools' answers so far. Pure, but for reading its one instruction file at import.

``messages`` is the whole input of one step: the instruction file as the system turn, then the
brief, the evidence the store holds and tonight's scratchpad (the tool answers and the notes it has
written), newest last. A step sees only what this night has shown it, so a resumed or rerun pass
reads exactly what its own call read.

The prompt is bounded: ``SHOWN`` evidence items (the newest), and each line is flattened, so a long
extraction can't crowd the token cap out of the model's reach.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from anjo.sleep.types import Evidence

_PROMPTS: Final = Path(__file__).parent / "prompts"
PROMPT: Final = (_PROMPTS / "agent_night_v1.md").read_text(encoding="utf-8")
SHOWN: Final = 24  # evidence items one step's prompt shows, newest of her store


def _line(text: str) -> str:
    return " ".join(text.split())


def _item(item: Evidence) -> str:
    return f"- [{item.id}] {item.day.isoformat()}: {_line(item.text)}"


def messages(brief: str | None, evidence: Sequence[Evidence], scratch: Sequence[str]) -> list[dict[str, Any]]:
    """One step's whole input: the instructions, then what the agent has to work from."""
    sections = [
        "## The brief you are given", brief.strip() if brief else "(no brief yet)",
        "## The evidence you hold", *([_item(item) for item in evidence] or ["(none)"]),
        "## Tonight so far", *(scratch or ["(nothing yet)"]),
        "Reply with one JSON object.",
    ]
    return [{"role": "system", "content": PROMPT.rstrip()}, {"role": "user", "content": "\n".join(sections)}]


def answer(tool: str, query: str, found: Sequence[Evidence]) -> str:
    """One tool call's answer, as the scratchpad keeps it: the items by id, or that nothing came."""
    lines = [f"### {tool} \"{_line(query)}\""]
    lines.extend(_item(item) for item in found)
    if not found:
        lines.append("(nothing found)")
    return "\n".join(lines)


def unoffered(tool: str, query: str) -> str:
    """A tool this night wasn't wired with, said plainly, so the agent picks another operation."""
    return f"### {tool} \"{_line(query)}\"\n(it isn't offered tonight)"


def wrote(note_id: str) -> str:
    """A note the agent wrote, as the scratchpad keeps it: its id, never its words again."""
    return f"### note {note_id}\n(written; arm 2A reads it in the morning)"
