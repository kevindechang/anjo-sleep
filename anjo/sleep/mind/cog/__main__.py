"""``python -m anjo.sleep.mind.cog``: see cli.py."""
from anjo.sleep.mind.cog.cli import main

if __name__ == "__main__":  # spawned workers import this module as __mp_main__ and must not rerun it
    raise SystemExit(main())
