"""The cog world's text: each event, query and question has words that say what its tokens say."""
from __future__ import annotations

import pytest

from anjo.sleep.mind.cog import render
from anjo.sleep.mind.cog.types import EVENT_KINDS

ACTIVITIES = ("run", "gym", "yoga", "class", "study_group", "call_mom", "dinner_sam", "climbing",
              "choir", "volunteering", "swim", "piano")


def _sample_tokens() -> dict[str, list[tuple[str, ...]]]:
    """Tokens of every event kind the world makes; own-day outcomes leave out the weekday, which
    the event's ``day`` carries for text and tokens alike."""
    own_day = {kind: [(kind, a, "dow=2") for a in ACTIVITIES] for kind in ("did", "skipped", "cancel")}
    plans = [("plan", a, f"in={h}", f"dow={(2 + h) % 7}") for a in ACTIVITIES for h in (1, 2, 3, 4)]
    moods = [("mood", f"level={level}") for level in range(1, 6)]
    messages = [("message", name, when) for name in ("Sam", "Maya") for when in ("late", "day")]
    mentions = [("mention", topic, value) for topic in ("severance", "weather", "warriors")
                for value in render.TOPIC_VALUES[render.topic_kind(topic)]]
    mentions += [("mention", hint, value) for hint, value in render.HINTS]
    answers = [option for subject in ("work", "plans", "severance", "weather", "warriors")
               for option in render.answer_options(subject)]
    return {**own_day, "plan": plans, "mood": moods, "message": messages, "mention": mentions,
            "answer": answers}


def test_every_event_kind_has_at_least_five_phrasings() -> None:
    samples = _sample_tokens()
    assert set(samples) == set(EVENT_KINDS)
    for kind, battery in samples.items():
        for tokens in battery:
            texts = {render.event_text(tokens, choice) for choice in range(40)}
            assert len(texts) >= 5, (kind, tokens, texts)


def test_text_is_a_function_of_tokens_and_choice() -> None:
    for battery in _sample_tokens().values():
        for tokens in battery:
            assert render.event_text(tokens, 7) == render.event_text(tokens, 7)
            assert render.event_text(tokens, 3) == render.event_text(tokens, 3 + 1000 * render.BANK_SPAN)


def test_different_tokens_never_share_a_text() -> None:
    """Text carries the same information as tokens: reading a text back gives one token tuple."""
    seen: dict[str, tuple[str, ...]] = {}
    for battery in _sample_tokens().values():
        for tokens in battery:
            for choice in range(40):
                text = render.event_text(tokens, choice)
                assert seen.setdefault(text, tokens) == tokens, (text, seen[text], tokens)


def test_texts_name_the_day_a_plan_is_for() -> None:
    assert "tomorrow" in render.event_text(("plan", "run", "in=1", "dow=3"), 0).lower()
    for choice in range(10):
        assert "Friday" in render.event_text(("plan", "dinner_sam", "in=3", "dow=4"), choice)
        assert "Sam" in render.event_text(("plan", "dinner_sam", "in=3", "dow=4"), choice)


def test_every_answer_option_reads_differently() -> None:
    for subject in ("work", "plans", "severance", "weather", "warriors"):
        options = render.answer_options(subject)
        assert len(options) == len(set(options)) >= 3
        texts = [render.event_text(option, 0) for option in options]
        assert len(set(texts)) == len(texts)
    assert len(render.answer_options("plans")) == 23  # none + exam 0..7 + visitor 1..3 + here 1..4 + injury 1..7


def test_query_and_question_texts() -> None:
    q = render.query_text(("query", "happen", "run", "h=1", "dow=3"), 0)
    assert q.endswith("?") and "tomorrow" in q.lower()
    kept = {render.query_text(("query", "plan", "dinner_sam", "h=3", "dow=4"), c) for c in range(6)}
    assert all("Sam" in text and "Friday" in text for text in kept) and len(kept) >= 2
    week = render.query_text(("query", "happen", "gym", "h=7", "dow=1"), 0)
    assert "week" in week and "Tuesday" in week
    assert render.question_text(("ask", "work"), 0).endswith("?")
    assert "Severance" in render.question_text(("ask", "severance"), 0)


def test_unknown_tokens_are_refused() -> None:
    with pytest.raises(ValueError):
        render.event_text(("dance", "run"), 0)
    with pytest.raises(ValueError):
        render.event_text(("did", "unicycling", "dow=1"), 0)
    with pytest.raises(ValueError):
        render.answer_options("the moon")
