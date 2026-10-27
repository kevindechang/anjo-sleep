"""The person's names, for homework's search screen (Task 7): the account's username and the name
they asked Anjo to call them. Read-only; a missing account or name gives nothing."""
from __future__ import annotations

from anjo.core.db import get_db
from anjo.core.self_core import RelationalState


def person_names(user_id: str) -> tuple[str, ...]:
    row = get_db().execute("SELECT username FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        return ()
    preferred = RelationalState.load(user_id).relationship.user_name
    return tuple(name for name in (row["username"], preferred) if name)
