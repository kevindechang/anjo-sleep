"""Undo a night (Task 9): the previous checkpoint serves again, and nothing else is touched.

The rollback is recorded in the registry (``CheckpointRegistry.rollback``), so history and
``previous`` skip the undone night and a second undo steps back once more. The evidence stays: the
next night trains from the restored checkpoint (``last_promoted``), and everything added since that
checkpoint's night counts as new again, so the undone night's lessons are learned afresh.
"""
from __future__ import annotations

from anjo.sleep.checkpoints import CheckpointRegistry
from anjo.sleep.types import CheckpointRef


class NothingToUndo(LookupError):
    """The person has no earlier checkpoint to go back to: the first night, or none yet."""


def undo(user_id: str, registry: CheckpointRegistry) -> CheckpointRef:
    previous = registry.previous(user_id)
    if previous is None:
        raise NothingToUndo("there is no earlier version to go back to")
    registry.rollback(user_id, previous)
    return previous
