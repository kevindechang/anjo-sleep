"""The cog world's words: every event, query and question rendered from its tokens. Pure.

Text is a function of ``(tokens, choice)``: ``choice`` picks a phrasing from a template bank and
nothing else, so a text says what its tokens say and no more. Own-day outcome tokens (``did``,
``skipped``, ``cancel``) carry ``dow=k`` for convenience; the event's ``day`` gives the weekday to
text and tokens alike, so their words don't repeat it. Adding ``BANK_SPAN`` to a choice never
changes a text, so a world that draws choices uniformly from ``range(BANK_SPAN)`` gives every bank
a uniform phrasing.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

WEEKDAYS: Final = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
PEOPLE: Final = ("Sam", "Maya", "Leo", "Priya", "Jonah", "Ava", "Theo", "Nina", "Omar", "Iris",
                 "Ben", "Zoe", "Kai", "Lena", "Max", "Ruth")
SHOWS: Final = {"severance": "Severance", "the_bear": "The Bear", "bluey": "Bluey",
                "slow_horses": "Slow Horses", "shogun": "Shogun", "abbott": "Abbott Elementary"}
TEAMS: Final = {"warriors": "Warriors", "arsenal": "Arsenal", "yankees": "Yankees",
                "lakers": "Lakers", "celtics": "Celtics", "packers": "Packers"}
WEATHER: Final = "weather"
TOPIC_VALUES: Final = {
    "show": ("loved_it", "boring", "cliffhanger", "rewatched", "binged", "fell_asleep",
             "new_season", "spoiled"),
    "weather": ("sunny", "rainy", "cloudy", "windy", "snowy", "foggy", "humid", "freezing"),
    "team": ("won", "lost", "drew", "no_game", "big_win", "close_loss", "traded", "injured_star"),
}
# The weak hints of a hidden fact: (subject, value) of a ``mention`` event.
HINTS: Final = (("reading", "lots"), ("guests", "tidying"), ("knee", "sore"))
REGIMES: Final = ("calm", "busy", "crunch")
EXAM_DAYS: Final = 7  # an exam is known from 7 days before ("in=7") to the day itself ("in=0")
VISITOR_LEAD: Final = 3  # a visitor is known from 3 days before arriving
VISITOR_STAY: Final = 4  # at most 4 days of a stay, counting today
INJURY_DAYS: Final = 7  # at most 7 days off sport, counting today


@dataclass(frozen=True)
class Words:
    past: str  # "went for a run"
    base: str  # "go for a run"
    noun: str  # "my run"
    bare: str  # "run"
    query: str  # third person: "go for a run", "call their mom"


_WORDS: Final = {
    "run": Words("went for a run", "go for a run", "my run", "run", "go for a run"),
    "gym": Words("went to the gym", "go to the gym", "the gym", "gym", "go to the gym"),
    "yoga": Words("did yoga", "do yoga", "yoga", "yoga", "do yoga"),
    "class": Words("went to class", "go to class", "class", "class", "go to class"),
    "study_group": Words("met my study group", "meet my study group", "study group", "study group",
                         "meet their study group"),
    "call_mom": Words("called Mom", "call Mom", "my call with Mom", "call with Mom", "call their mom"),
    "climbing": Words("went climbing", "go climbing", "climbing", "climbing", "go climbing"),
    "choir": Words("went to choir", "go to choir", "choir practice", "choir", "go to choir"),
    "volunteering": Words("volunteered at the food bank", "volunteer at the food bank",
                          "volunteering", "volunteering", "volunteer at the food bank"),
    "swim": Words("went for a swim", "go for a swim", "my swim", "swim", "go for a swim"),
    "piano": Words("had my piano lesson", "go to my piano lesson", "my piano lesson", "piano lesson",
                   "go to their piano lesson"),
}
DINNER_PREFIX: Final = "dinner_"


def dinner_id(name: str) -> str:
    return DINNER_PREFIX + name.lower()


def words(activity: str) -> Words:
    if activity.startswith(DINNER_PREFIX):
        name = activity[len(DINNER_PREFIX):].capitalize()
        if name not in PEOPLE:
            raise ValueError(f"unknown dinner companion in {activity!r}")
        dinner = f"dinner with {name}"
        return Words(f"had {dinner}", f"have {dinner}", dinner, dinner, f"have {dinner}")
    found = _WORDS.get(activity)
    if found is None:
        raise ValueError(f"unknown activity {activity!r}")
    return found


def topic_kind(topic: str) -> str:
    if topic == WEATHER:
        return "weather"
    if topic in SHOWS:
        return "show"
    if topic in TEAMS:
        return "team"
    raise ValueError(f"unknown topic {topic!r}")


def topic_name(topic: str) -> str:
    kind = topic_kind(topic)
    return {"show": SHOWS, "team": TEAMS}[kind][topic] if kind != "weather" else "the weather"


# ── template banks ────────────────────────────────────────────────────────────────────────────
_DID: Final = ("{Past}.", "{Past} today.", "Managed to {base} today.", "{Past}, felt good.",
               "Didn't miss {noun} today.", "Yep, {past}.")
_SKIPPED: Final = ("Skipped {noun} today.", "Didn't {base} today.", "No {bare} today.", "Missed {noun}.",
                   "Was meant to {base} but didn't.", "Couldn't face {noun} today.")
_CANCEL: Final = ("Cancelled {noun}.", "Had to cancel {noun}.", "Bailed on {noun} today.",
                  "Called off {noun}.", "{Bare} was off, I cancelled.")
_PLAN: Final = ("Planning to {base} {when}.", "Going to {base} {when}.", "Put {bare} in the calendar {for_when}.",
                "Set up {bare} {for_when}.", "Should {base} {when}.", "Booked in {bare} {for_when}.")
_MOOD: Final = {
    1: ("Feeling awful today.", "Really rough day.", "Totally drained and low.", "Worst day in a while.",
        "Everything feels heavy today."),
    2: ("Pretty tired and a bit low.", "Not a great day.", "Feeling flat today.",
        "A bit stressed and worn out.", "Meh, kind of down."),
    3: ("An okay day.", "Feeling alright.", "Nothing special, fine.", "So-so today.", "Pretty average day."),
    4: ("Feeling good today.", "Nice day, pretty upbeat.", "In a good mood.", "Things feel good.",
        "Pretty happy today."),
    5: ("Great day, full of energy!", "Feeling fantastic.", "Best mood in ages.",
        "On top of the world today.", "Really happy and energized."),
}
_MESSAGE: Final = {
    "late": ("Texted {name} really late.", "Was up messaging {name} past midnight.",
             "Sent {name} a message at 1am.", "Late-night texts with {name} again.",
             "Messaged {name} way after midnight."),
    "day": ("Messaged {name}.", "Quick text to {name}.", "Texted {name} this afternoon.",
            "Sent {name} a message.", "Chatted with {name} over text."),
}
_MENTION_FRAMES: Final = ("{Phrase}.", "Btw, {phrase}.", "Random thought: {phrase}.", "Honestly, {phrase}.",
                          "{Phrase}, by the way.")
_TOPIC_PHRASES: Final = {
    "show": {"loved_it": "loved the new episode of {name}", "boring": "the new {name} episode was boring",
             "cliffhanger": "{name} ended on a cliffhanger", "rewatched": "rewatched an old episode of {name}",
             "binged": "binged three episodes of {name}", "fell_asleep": "fell asleep during {name}",
             "new_season": "{name} got a new season", "spoiled": "someone spoiled {name} for me"},
    "weather": {"sunny": "it was sunny all day", "rainy": "it rained all day", "cloudy": "it was grey and cloudy",
                "windy": "it was really windy", "snowy": "it snowed a little", "foggy": "it was foggy all morning",
                "humid": "it was so humid", "freezing": "it was freezing out"},
    "team": {"won": "the {name} won", "lost": "the {name} lost", "drew": "the {name} drew",
             "no_game": "the {name} had no game", "big_win": "the {name} crushed it",
             "close_loss": "the {name} lost by a point", "traded": "the {name} made a big trade",
             "injured_star": "the {name}' star got hurt"},
}
_HINT_PHRASES: Final = {("reading", "lots"): "there's so much to read lately",
                        ("guests", "tidying"): "I need to tidy the spare room",
                        ("knee", "sore"): "my knee's a bit sore"}
_ANSWER_WORK: Final = ("Work's been {w}.", "Honestly, work has been {w} this week.", "It's been {w} at work.",
                       "Work? {W}.", "{W}, work-wise.")
_WORK_WORDS: Final = {"calm": "pretty calm", "busy": "busy", "crunch": "a total crunch"}
_ANSWER_NONE: Final = ("Nothing big coming up.", "Nope, nothing special.", "Not really, all quiet.",
                       "Nothing on the horizon.", "No, nothing big.")
_ANSWER_FACT: Final = ("I have {phrase}.", "Yeah, {phrase}.", "Well, there's {phrase}.",
                       "Something, yes: {phrase}.", "Since you mention it, {phrase}.")
_ANSWER_TOPIC: Final = ("Since you ask, {phrase}.", "Well, {phrase}.", "Oh, {phrase}.", "To answer you, {phrase}.",
                        "Good question: {phrase}.")
_QUERY: Final = {
    "happen": ("Will they {q} {when}?", "Are they going to {q} {when}?", "Do you think they'll {q} {when}?"),
    "kept": ("Will they keep their plan to {q} {when}?", "Will the plan for {bare} {when} hold?",
             "Is {bare} {when} still going to happen?"),
}
_QUESTION: Final = {
    "work": ("How's work been this week?", "How busy is work right now?", "Is work treating you okay this week?"),
    "plans": ("Anything big coming up?", "Anything going on in the next week or so?", "Got anything big on soon?"),
    "show": ("How's {name} going?", "Still watching {name}?", "Any news on {name}?"),
    "weather": ("How's the weather been?", "What's the weather doing today?", "Nice out today?"),
    "team": ("How are the {name} doing?", "Did you catch the {name}?", "Any {name} news?"),
}
_BANKS: Final = (_DID, _SKIPPED, _CANCEL, _PLAN, *_MOOD.values(), *_MESSAGE.values(), _MENTION_FRAMES,
                 _ANSWER_WORK, _ANSWER_NONE, _ANSWER_FACT, _ANSWER_TOPIC, *_QUERY.values(), *_QUESTION.values())
BANK_SPAN: Final = math.lcm(*(len(bank) for bank in _BANKS))


# ── rendering ─────────────────────────────────────────────────────────────────────────────────
def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _pick(bank: tuple[str, ...], choice: int) -> str:
    return bank[choice % len(bank)]


def _field(tokens: tuple[str, ...], key: str) -> int:
    prefix = key + "="
    for token in tokens:
        if token.startswith(prefix):
            return int(token[len(prefix):])
    raise ValueError(f"tokens lack {key}=")


def _days(n: int) -> str:
    return f"{n} more day" + ("" if n == 1 else "s")


def when_text(ahead: int, weekday: int) -> tuple[str, str]:
    """("tomorrow", "for tomorrow") or ("on Friday", "for Friday") for a day ``ahead`` days on."""
    if not 1 <= ahead <= 7 or not 0 <= weekday <= 6:
        raise ValueError("a day ahead must be 1-7 days on, on a weekday 0-6")
    if ahead == 1:
        return "tomorrow", "for tomorrow"
    if ahead == 7:
        return f"a week from today ({WEEKDAYS[weekday]})", f"for a week from today ({WEEKDAYS[weekday]})"
    return f"on {WEEKDAYS[weekday]}", f"for {WEEKDAYS[weekday]}"


def _fill_activity(template: str, activity: str, **extra: str) -> str:
    w = words(activity)
    return template.format(Past=_cap(w.past), past=w.past, base=w.base, noun=w.noun, bare=w.bare,
                           Bare=_cap(w.bare), q=w.query, **extra)


def _own_day(bank: tuple[str, ...]) -> Callable[[tuple[str, ...], int], str]:
    def render(tokens: tuple[str, ...], choice: int) -> str:
        if len(tokens) != 3:
            raise ValueError("an own-day outcome is (kind, activity, dow=k)")
        _field(tokens, "dow")
        return _fill_activity(_pick(bank, choice), tokens[1])
    return render


def _plan(tokens: tuple[str, ...], choice: int) -> str:
    when, for_when = when_text(_field(tokens, "in"), _field(tokens, "dow"))
    return _fill_activity(_pick(_PLAN, choice), tokens[1], when=when, for_when=for_when)


def _mood(tokens: tuple[str, ...], choice: int) -> str:
    bank = _MOOD.get(_field(tokens, "level"))
    if bank is None:
        raise ValueError("a mood level is 1-5")
    return _pick(bank, choice)


def _message(tokens: tuple[str, ...], choice: int) -> str:
    if len(tokens) != 3 or tokens[1] not in PEOPLE or tokens[2] not in _MESSAGE:
        raise ValueError("a message is (message, name, late|day)")
    return _pick(_MESSAGE[tokens[2]], choice).format(name=tokens[1])


def _topic_phrase(topic: str, value: str) -> str:
    phrases = _TOPIC_PHRASES[topic_kind(topic)]
    if value not in phrases:
        raise ValueError(f"unknown value {value!r} for a topic")
    return phrases[value].format(name=topic_name(topic) if topic != WEATHER else "")


def _framed(bank: tuple[str, ...], phrase: str, choice: int) -> str:
    return _pick(bank, choice).format(Phrase=_cap(phrase), phrase=phrase)


def _mention(tokens: tuple[str, ...], choice: int) -> str:
    if len(tokens) != 3:
        raise ValueError("a mention is (mention, subject, value)")
    hint = _HINT_PHRASES.get((tokens[1], tokens[2]))
    phrase = hint if hint is not None else _topic_phrase(tokens[1], tokens[2])
    return _framed(_MENTION_FRAMES, phrase, choice)


def _fact_phrase(tokens: tuple[str, ...]) -> str:
    kind = tokens[2]
    if kind == "exam":
        c = _field(tokens, "in")
        return "an exam " + ("today" if c == 0 else "tomorrow" if c == 1 else f"in {c} days")
    if kind == "visitor" and "here" in tokens:
        return f"a guest staying with me for {_days(_field(tokens, 'left'))}"
    if kind == "visitor":
        c = _field(tokens, "in")
        return "a visitor arriving " + ("tomorrow" if c == 1 else f"in {c} days")
    if kind == "injury":
        return f"a sore knee keeping me off sport for {_days(_field(tokens, 'left'))}"
    raise ValueError("unknown fact in an answer")


def _answer(tokens: tuple[str, ...], choice: int) -> str:
    if tokens not in answer_options(tokens[1] if len(tokens) > 1 else ""):
        raise ValueError("an answer must be one of its question's options")
    subject = tokens[1]
    if subject == "work":
        w = _WORK_WORDS[tokens[2]]
        return _pick(_ANSWER_WORK, choice).format(w=w, W=_cap(w))
    if subject == "plans":
        if tokens[2] == "none":
            return _pick(_ANSWER_NONE, choice)
        return _framed(_ANSWER_FACT, _fact_phrase(tokens), choice)
    return _framed(_ANSWER_TOPIC, _topic_phrase(subject, tokens[2]), choice)


_EVENT_RENDERERS: Final[dict[str, Callable[[tuple[str, ...], int], str]]] = {
    "did": _own_day(_DID), "skipped": _own_day(_SKIPPED), "cancel": _own_day(_CANCEL), "plan": _plan,
    "mood": _mood, "message": _message, "mention": _mention, "answer": _answer,
}


def event_text(tokens: tuple[str, ...], choice: int) -> str:
    renderer = _EVENT_RENDERERS.get(tokens[0] if tokens else "")
    if renderer is None:
        raise ValueError("unknown event kind")
    return renderer(tokens, choice)


def answer_options(subject: str) -> tuple[tuple[str, ...], ...]:
    """Every answer a person could give to the question about ``subject``, as answer tokens."""
    if subject == "work":
        return tuple(("answer", "work", r) for r in REGIMES)
    if subject == "plans":
        return (("answer", "plans", "none"),
                *(("answer", "plans", "exam", f"in={c}") for c in range(EXAM_DAYS + 1)),
                *(("answer", "plans", "visitor", f"in={c}") for c in range(1, VISITOR_LEAD + 1)),
                *(("answer", "plans", "visitor", "here", f"left={r}") for r in range(1, VISITOR_STAY + 1)),
                *(("answer", "plans", "injury", f"left={r}") for r in range(1, INJURY_DAYS + 1)))
    return tuple(("answer", subject, value) for value in TOPIC_VALUES[topic_kind(subject)])


def query_text(tokens: tuple[str, ...], choice: int) -> str:
    """("query", "happen"|"kept", activity, "in=h", "dow=k") as a question about the person."""
    if len(tokens) != 5 or tokens[0] != "query" or tokens[1] not in _QUERY:
        raise ValueError("a query is (query, happen|kept, activity, in=h, dow=k)")
    when, _ = when_text(_field(tokens, "in"), _field(tokens, "dow"))
    return _cap(_fill_activity(_pick(_QUERY[tokens[1]], choice), tokens[2], when=when))


def question_text(tokens: tuple[str, ...], choice: int) -> str:
    """("ask", subject) as the words put to the person."""
    if len(tokens) != 2 or tokens[0] != "ask":
        raise ValueError("a question is (ask, subject)")
    subject = tokens[1]
    if subject in ("work", "plans"):
        return _pick(_QUESTION[subject], choice)
    kind = topic_kind(subject)
    return _pick(_QUESTION[kind], choice).format(name=topic_name(subject))
