"""The style-neutral rewrite (Task 2, spec 5.1): the same points in a plain voice.

Muse Spark rewrites a reply with the registered ``neutral_v1.md`` at temperature 0. It is used
only to judge ``new_work`` and ``vocab`` again with style flattened, as a diagnostic: a win that
survives isn't tone, length or confidence alone, and one that doesn't proves nothing either way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.template import render

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "neutral_v1.md"
MODEL: Final = "muse-spark"
TEMPERATURE: Final = 0.0
TIMEOUT_S: Final = 120.0


def rewrite(reply: str, *, chat: ChatFn | None = None) -> str:
    """``reply`` in a plain voice, under 150 words, with every point kept."""
    if chat is None:
        from anjo.sleep import meta_api

        chat = meta_api.chat
    prompt = render(PROMPT_PATH.read_text(encoding="utf-8"), {"REPLY": reply})
    result = chat(MODEL, [{"role": "user", "content": prompt}], None, TIMEOUT_S, temperature=TEMPERATURE)
    return result.text.strip()
