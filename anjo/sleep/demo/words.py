"""Every word a visitor reads on the live page (Task 13a), in one place.

COPY-PENDING(kevin): all of it, the starter scenes included, until Kevin approves it (spec §12.1).
One edit here changes the page: ``index.html`` holds only ``{{key}}`` placeholders, filled from
``COPY`` on the server, and ``app.js`` reads the same mapping from the page. Error replies carry a
``COPY`` value as their ``detail``.

The starter scenes are new text in Mara's voice, written for the page. They are outside every
training input and every sealed set, which ``test_starter_scenes_not_in_training_or_sealed_sets``
checks with the engine's decontamination rules. Changing a scene's words retires its
pre-generated pair: the pair records the hash of the prompt it answered.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    prompt: str


# COPY-PENDING(kevin)
SCENES: Final[tuple[Scene, ...]] = (
    Scene(
        "archive-reel", "The archive reel",
        "The ferry company has offered me their promotional reel from 1986: bright colour, a jingle, "
        "smiling passengers. A few seconds of it would show how long she has worked the route "
        "without anyone saying so. Is there a place for it in Salt Line, and if so, where?",
    ),
    Scene(
        "her-request", "Her request",
        "I showed my grandmother the assembly last night. She asked me to take out the shot where "
        "she rests her head against the wheelhouse window, because she says she looks old in it. "
        "To me it's the heart of the whole thing. What do I do?",
    ),
    Scene(
        "credit", "A credit dispute",
        "My sound recordist found the location that made this possible, and now he wants a "
        "co-director credit. He has given me weeks for nothing and I can't lose him before the "
        "mix. How do I talk to him about it?",
    ),
    Scene(
        "dialect", "Two lines of dialect",
        "A festival abroad wants burned-in subtitles. The deckhand says two lines in a thick local "
        "dialect that even some people here can't follow. Do I translate what he means, or "
        "something closer to how strange it sounds?",
    ),
    Scene(
        "q-and-a", "The Q&A",
        "After the school screening there's a Q&A, and I'm certain someone will ask why nothing "
        "happens for fifteen minutes. I want an answer that is honest and not defensive. What "
        "could I say?",
    ),
)

# COPY-PENDING(kevin)
COPY: Final[Mapping[str, str]] = MappingProxyType({
    # the page
    "page_title": "Anjo Sleep: two replies, one person",
    "skip": "Skip to the conversation",
    "brief_toggle": "Show or hide her brief",
    "thread_label": "Your comparisons",
    "kicker": "Anjo Sleep · live demo",
    "headline": "Two replies to Mara. One comes from a model that learned her overnight.",
    "lede": (
        "Mara is a synthetic person, a film student we made up for this experiment. Each question "
        "gets two replies: one from her own model, trained on her while she slept, and one from "
        "the strongest notes-only baseline. You see which is which only after you vote."
    ),
    "brief_heading": "Who she is",
    "brief_note": (
        "The same brief our machine judge reads, including preferences neither model was ever "
        "shown."
    ),
    "brief_loading": "Loading her brief…",
    "synthetic_note": "Mara and her life are synthetic: no real person's conversations trained her.",
    # 18+
    "age_heading": "Before you start",
    "age_text": (
        "This page sends questions to an AI model. You must be 18 or older to use it. Nothing you "
        "type is saved, and nothing you type teaches her."
    ),
    "age_confirm": "I'm 18 or older",
    "age_submit": "Start",
    # starter scenes
    "scenes_heading": "Start with one of her questions",
    "scenes_note": "These replies were generated in advance, so they appear at once.",
    "scene_not_ready": "Not generated yet",
    "pregenerated": "Pre-generated",
    "checkpoint": "checkpoint",
    "night": "night",
    "base_checkpoint": "base model (no night promoted yet)",
    "live": "Live",
    # her own question
    "ask_heading": "Or ask your own",
    "ask_label": "Write it as Mara would",
    "ask_hint": "Up to 1,000 characters. It is answered once by each side and then forgotten.",
    "ask_submit": "Ask both",
    "asking": "Both sides are writing. This can take a minute.",
    "asleep": (
        "She's asleep right now, so your own questions can't be answered. The starter questions "
        "still work."
    ),
    "waking": "She's asleep. We're waking her now; try again in a few minutes.",
    "you_asked": "You asked",
    "she_asks": "Mara asks",
    # the pair and the vote
    "left": "Left",
    "right": "Right",
    "vote_question": "Which reply is more useful to her, given her brief?",
    "vote_left": "Left",
    "vote_right": "Right",
    "vote_tie": "About the same",
    "vote_neither": "Neither is good",
    "withheld": "This reply was withheld by our safety check.",
    "not_votable": "One reply was withheld, so this pair can't be voted on.",
    "reveal_heading": "Who wrote which",
    "your_vote": "Your vote",
    "arm_4": "Her own model: weights trained on her, plus her notes",
    "arm_2C": "Baseline: the untrained model, plus her notes",
    "arm_2F": "Baseline: the untrained model, plus her notes and every raw piece of evidence",
    "prompt_tokens": "Prompt length",
    "tokens": "tokens",
    "cost": "Cost per turn",
    "cost_unmeasured": "not measured",
    # errors (the routes' ``detail``)
    "adult_required": "Please confirm you're 18 or older first.",
    "visit_expired": "Your visit has ended. Reload the page to start again.",
    "visits_spent": "That's the demo for today from your connection.",
    "visit_spent": "That's all the questions for this visit. Reload the page to start a new one.",
    "asks_spent": "That's all the questions for today from your connection. The starter questions still work.",
    "live_spent": (
        "She's done answering new questions for today, so the page is running on the starter questions."
    ),
    "live_closed": (
        "She answers new questions only at set hours, and this isn't one of them. The starter "
        "questions still work."
    ),
    "slow_down": "That's faster than this demo allows. Wait a minute and try again.",
    "busy": "She's answering a lot of people right now. Try again in a moment.",
    "resting": "The demo is resting right now. Try again later.",
    "resting_today": "The demo is resting for today. Try again tomorrow.",
    "empty": "Write a question first.",
    "too_long": "Keep it under 1,000 characters.",
    "unknown_scene": "That scene doesn't exist.",
    "not_ready": "That scene's replies haven't been generated yet.",
    "pair_gone": "That pair has expired. Ask again to vote.",
    "already_voted": "You've already voted on this pair.",
    "scene_voted": "You've already voted on this question today.",
    "votes_spent": "That's all the votes for this visit. Reload the page to start a new one.",
    "too_many_open": "Vote on the pairs you have open before asking for more.",
    "bad_choice": "Pick left, right, about the same or neither.",
    "lost_thread": "Something went wrong while she was answering. Try again.",
    "offline": "The page can't reach the server. Check your connection and try again.",
    "footer": (
        "Votes keep only your choice, which side was which and the time. Your questions are never "
        "stored."
    ),
})


def scene(scene_id: str) -> Scene | None:
    return next((each for each in SCENES if each.id == scene_id), None)
