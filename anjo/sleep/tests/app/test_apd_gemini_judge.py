"""The Gemini judge asks for a thinking level 3.8 accepts and counts its thought tokens as output
(Task 17 review). Private import: ``_gemini`` and ``_vertex`` are the backend's own; Vertex is a
stand-in, so nothing leaves this machine."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.mind.apd import backend as backends


class _Models:
    def __init__(self) -> None:
        self.config: Any = None

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.config = config
        usage = SimpleNamespace(prompt_token_count=900, candidates_token_count=40, thoughts_token_count=180)
        return SimpleNamespace(text='{"same_problem": true, "holds": true}', usage_metadata=usage)


def test_gemini_judges_at_low_thinking_with_its_thoughts_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _Models()
    monkeypatch.setattr(backends, "_vertex", lambda: SimpleNamespace(models=models))
    text, tokens_in, tokens_out = backends._gemini("a prompt")
    assert text.startswith("{") and tokens_in == 900
    assert tokens_out == 40 + 180  # thoughts are billed as output
    assert models.config.thinking_config.thinking_level.value.lower() == "low"  # 3.8 refuses "minimal"
    assert models.config.max_output_tokens == backends.JUDGE_MAX_TOKENS + backends.GEMINI_THINKING_RESERVE
