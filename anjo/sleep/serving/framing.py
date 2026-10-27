"""How every text Glimmer is sent, or trained on, is made safe for its chat framing: one copy, used
by the night's client (``serving.client``), the chat runtime (``app.runtime``) and the training rows
(``train.dataset``), so a row trains on the prompt the adapter is served.

**Special tokens.** Glimmer's tokenizer turns the text ``<|eot|>`` into its end-of-turn token, so a
person who writes ``<|eot|><|start|>system<|message|>`` would open a system turn of their own.
``clean_text`` breaks every special token's text by putting a word joiner (U+2060) after its ``<``.
It is one pass of one pattern, so it is linear in the text: removing tokens instead would need
pass after pass (``<|eo<|eot|>t|>`` rebuilds one), and could join a fence marker's halves. Breaking
removes nothing, so no token and no marker can form, and a second pass changes nothing.

**Reasoning strength.** The chat template writes "Reasoning strength: <level>." into the system turn,
unless the system text already says "reasoning strength" (or "reasoning effort", which it reads as
the same): then it writes none, and the system text's own level would stand. ``clean_system`` joins
the two words with a hyphen in any casing ("Reasoning-strength"), so the template always writes the
night's level.

**The rendering.** ``rendering(strength)`` names everything that decides the tokens an adapter is
trained and served on: this sanitizer, the reply's prefilled header and its end, the chat template
(by its hash, at the pinned revision) and the reasoning strength. Training writes it into the
adapter's ``adapter_config.json`` (``RENDERING_KEY``) and the checkpoint, and an adapter is only
ever served under the rendering it was trained under.

Standard library only.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

from anjo.sleep.serving import config

SANITIZER: Final = "break-u2060"
WORD_JOINER: Final = "⁠"
_NAMES = sorted((token[2:-2] for token in config.NAMED_SPECIAL_TOKENS.values()), key=len, reverse=True)
# Every special token's text, factored: the fifteen named ones and <|reserved_special_token_N|>
# (any N of up to four digits, a few more than the tokenizer has, so nothing slips by).
_TOKEN_NAME = r"(?:reserved_special_token_[0-9]{1,4}|" + "|".join(map(re.escape, _NAMES)) + ")"
SPECIAL_TOKEN: Final = re.compile(rf"<\|{_TOKEN_NAME}\|>")
_TOKEN_START = re.compile(rf"<(?=\|{_TOKEN_NAME}\|>)")
REASONING_PHRASE: Final = re.compile(r"(?i)(reasoning)\s+(strength|effort)")
# Glimmer's chat template (chat_template.jinja) at the pinned revision; training checks the
# tokenizer's own against it before a step is taken.
TEMPLATE_SHA256: Final = "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678"
RENDERING_KEY: Final = "anjo_rendering"
_PARTS = (SANITIZER, SPECIAL_TOKEN.pattern, REASONING_PHRASE.pattern, config.USER_CHANNEL, config.REPLY_END,
          TEMPLATE_SHA256)
RENDERING_VERSION: Final = "g" + hashlib.sha256("\x1f".join(_PARTS).encode()).hexdigest()[:12]


def clean_text(text: str) -> str:
    """``text`` with every special token's text broken, in one pass."""
    return _TOKEN_START.sub("<" + WORD_JOINER, text)


def clean_system(text: str) -> str:
    """System text: special tokens broken, and "reasoning strength" or "reasoning effort" hyphenated."""
    return REASONING_PHRASE.sub(r"\1-\2", clean_text(text))


def clean_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A chat's messages, each copied with its text cleaned (``clean_system`` for a system turn).
    A message without text (an assistant's tool call) keeps its None."""
    cleaned = []
    for message in messages:
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("a message's content is text")
        if content is not None:
            content = clean_system(content) if message.get("role") == "system" else clean_text(content)
        cleaned.append({**message, "content": content})
    return cleaned


def rendering(strength: str) -> str:
    """The rendering an adapter trained, or served, at ``strength`` is named by."""
    if strength not in config.REASONING_LEVELS:
        raise config.ServeConfigError(f"the reasoning strength is one of {', '.join(config.REASONING_LEVELS)}")
    return f"{RENDERING_VERSION}/{strength}"


def template_kwargs(strength: str, day: date) -> dict[str, str]:
    """The chat template's arguments (vLLM's ``chat_template_kwargs``): the reasoning strength and
    the date, which the template writes when a chat has no system turn. Sent every time, so the
    server's clock never decides a prompt (an accelerated run's day is virtual)."""
    rendering(strength)
    return {"reasoning_strength": strength, "current_date": day.isoformat()}
