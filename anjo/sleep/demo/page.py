"""The live page's files (Task 13a): ``anjo/sleep/web``.

``index.html`` holds no words of its own: each ``{{key}}`` is filled from ``words.COPY``, escaped,
and ``{{@copy}}`` becomes the whole mapping as JSON for ``app.js``, escaped so no ``</script>`` can
end the element early. The page loads nothing from anywhere but this host.
"""
from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from anjo.sleep.demo.words import COPY

WEB_DIR: Final = Path(__file__).parents[1] / "web"
ASSETS: Final[Mapping[str, str]] = MappingProxyType({
    "app.js": "text/javascript; charset=utf-8",
    "demo.css": "text/css; charset=utf-8",
    "nights.js": "text/javascript; charset=utf-8",  # the full step: the night slider and the toggles
    "mind.js": "text/javascript; charset=utf-8",  # her mind panel and its chart
    "mind.css": "text/css; charset=utf-8",
})
COPY_SLOT: Final = "{{@copy}}"
_SLOT = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def _json_for_script(copy: Mapping[str, str]) -> str:
    text = json.dumps(dict(copy), ensure_ascii=True, sort_keys=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render(template: str, copy: Mapping[str, str] = COPY) -> str:
    """``template`` with every slot filled. A slot naming no copy is a ``KeyError``."""
    filled = _SLOT.sub(lambda match: html.escape(copy[match.group(1)]), template)
    return filled.replace(COPY_SLOT, _json_for_script(copy))


def page_html() -> str:
    return render((WEB_DIR / "index.html").read_text(encoding="utf-8"))


def asset(name: str) -> tuple[bytes, str] | None:
    """One of the page's files and its media type, or None for any other name."""
    media = ASSETS.get(name)
    if media is None:
        return None
    return (WEB_DIR / name).read_bytes(), media
