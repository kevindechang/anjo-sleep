"""``/tokenize`` asked once per body for a client's life, whichever model the body names
(``VllmClient.reusing_tokens``): for a run that scores the same lines on many adapters, the post-hoc
knowledge scorer's (``tools.knowledge_score``).

Live 2026-10-29, one request at a time: a side of the knowledge set (202 choices) took 277-300 s, of
which 405 posts were ``/tokenize`` (two a choice and the reply's header), each a round trip of about
0.45 s to Modal, asking every adapter for ids the base had already given.

**Why the model can be left out of the key.** An adapter folder the server loads holds exactly
``config.ADAPTER_FILES``, the adapter's weights and its config: the training job copies those two files
alone onto the adapters volume (``train.modal_job``), and ``config.check_adapter`` refuses a folder with
any other file before an upload or a stage. So no tokenizer or chat template travels with an adapter, and
vLLM tokenizes a request naming one with the base model's; and training renders every row with that same
tokenizer at the pinned revision (``train.colab_train.served_ids``). So the base's ids are the ids each
adapter was trained on, whatever model a body names. The key is the rest of
the body, the date and reasoning strength of a chat included. Only token ids are kept; nothing else is.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from typing import Any, Final

TOKENIZE: Final = "/tokenize"


class TokenMemo:
    """``/tokenize``'s answers by body, the model aside; one memo is shared by a client's copies."""

    def __init__(self) -> None:
        self._ids: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def post(self, body: Mapping[str, Any], send: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """The ids asked for ``body`` before, or ``send``'s answer, kept when it carries a list of ids."""
        key = json.dumps({name: value for name, value in body.items() if name != "model"}, sort_keys=True)
        with self._lock:
            known = self._ids.get(key)
        if known is not None:
            return {"tokens": list(known)}
        answer = send()
        if isinstance(ids := answer.get("tokens"), list):
            with self._lock:
                self._ids.setdefault(key, list(ids))
        return answer
