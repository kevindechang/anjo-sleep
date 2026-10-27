# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The person's names for homework's search screen (Task 7): the account's username and the name
they asked Anjo to call them."""
from __future__ import annotations

from typing import Any


def test_person_names_reads_the_username_and_the_preferred_name(auth_client: Any) -> None:
    from anjo.core.db import get_db
    from anjo.core.self_core import RelationalState
    from anjo.sleep.app.names import person_names

    db = get_db()
    db.execute(
        "INSERT INTO users (user_id, username, email, hashed_password, is_adult, created_at) "
        "VALUES ('person-names', 'maraq', 'person-names@example.invalid', '!offline', 1, '2026-10-28T00:00:00+00:00')"
    )
    db.commit()
    state = RelationalState.load("person-names")
    state.relationship.user_name = "Mara Quint"
    state.save()
    assert person_names("person-names") == ("maraq", "Mara Quint")
    assert person_names("person-nobody") == ()
