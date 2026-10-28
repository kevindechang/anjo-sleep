"""Replay's score (Task 17b). Pure.

A branch scores ``-Σᵢ (pᵢ - yᵢ)² - λ x tokens / 1000``: the multi-class Brier score of its credence
vector over the start state's frozen set against the label (``yᵢ`` is 1 for the hypothesis that
held, else 0), less ``penalty`` (λ, ``replay.settings``) per thousand tokens. Every branch of a start state is
scored on the same frozen set and the same label, so none wins by answering an easier question.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from anjo.sleep.mind.replay.types import Branch, Label, Scored, StartState


def brier(credences: Mapping[str, float], frozen_ids: Sequence[str], holds: str) -> float:
    """The negated multi-class Brier score (0 is perfect, -2 the worst)."""
    if tuple(credences) != tuple(frozen_ids) or holds not in frozen_ids:
        raise ValueError("a branch is scored on its start state's frozen hypothesis set only")
    return -sum((credences[h] - (1.0 if h == holds else 0.0)) ** 2 for h in frozen_ids)


def score_branch(branch: Branch, start: StartState, label: Label, *, penalty: float) -> Scored:
    if branch.start_id != start.id or label.start_id != start.id:
        raise ValueError(f"branch {branch.start_id} and label {label.start_id} aren't start state {start.id}'s")
    if label.holds is None:
        raise ValueError(f"start state {start.id} is undetermined: its branches are never scored")
    value = brier(branch.credences, start.frozen_ids, label.holds)
    return Scored(branch=branch, brier=value, score=value - penalty * branch.tokens / 1000)


def score_start(
    branches: Sequence[Branch], start: StartState, label: Label, *, penalty: float,
) -> tuple[Scored, ...]:
    """Every branch of ``start``, against its one label."""
    return tuple(score_branch(branch, start, label, penalty=penalty) for branch in branches)
