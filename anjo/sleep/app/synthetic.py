"""Marks accounts as synthetic people, on the sleep host only (Task 4).

    ANJO_SLEEP_HOST=1 python -m anjo.sleep.app.synthetic <user_id> [<user_id> ...]

Run by the operator once per account the persona simulator plays, after the account is made.
Only a synthetic person's session can be cleared, so this stays off the HTTP API: no account can
mark itself synthetic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from anjo.sleep.app.cleared_sessions import register_synthetic
from anjo.sleep.app.flags import require_sleep_host


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mark accounts as synthetic people.")
    parser.add_argument("user_ids", nargs="+")
    args = parser.parse_args(argv)
    require_sleep_host()
    from anjo.core.ids import validate_user_id
    from anjo.core.user_data_guard import account_exists

    for user_id in args.user_ids:
        validate_user_id(user_id)
        if not account_exists(user_id):
            parser.error(f"no account {user_id}")
    for user_id in args.user_ids:
        register_synthetic(user_id, at=datetime.now(timezone.utc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
