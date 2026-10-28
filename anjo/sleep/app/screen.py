"""MIND's domain screen for one account (Task 17): ``anjo.sleep.mind.screen`` with the person's
names, which only the app may read."""
from __future__ import annotations

from anjo.sleep.app.names import person_names
from anjo.sleep.mind.screen import SCREEN_MAX_TOKENS as SCREEN_MAX_TOKENS
from anjo.sleep.mind.screen import SCREEN_PROMPT as SCREEN_PROMPT
from anjo.sleep.mind.screen import SparkScreen as SparkScreen
from anjo.sleep.mind.screen import personal_terms as personal_terms


def screen_for(user_id: str) -> SparkScreen:
    """The screen for one account, with its names."""
    return SparkScreen(names=person_names(user_id))
