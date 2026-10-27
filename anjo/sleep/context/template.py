"""Fills a registered prompt's ``{{SLOT}}`` markers in one pass.

One pass matters: a value (a reply, an evidence line) that itself holds ``{{SLOT}}`` text is
never filled in again, so no input can reach another slot.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

_SLOT = re.compile(r"\{\{([A-Z_]+)\}\}")


def render(template: str, values: Mapping[str, str]) -> str:
    """``template`` with every slot replaced; a slot with no value is an error."""

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"no value for slot {name}")
        return values[name]

    return _SLOT.sub(fill, template)
