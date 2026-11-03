"""Catalog and phrasing for ``world_indep``: who people can be, and the words they use.

Wording is picked from a text-only random stream (the caller passes it), so a phrase never carries
anything about the person's hidden state: every fact in a text is also in the event's tokens, and
the rest is colour (a destination, a distance, which synonym).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class Spec:
    """What kind of thing an activity is; the world turns this into sensitivities."""
    kind: str  # physical | class | social | chore | outing
    slot: str  # morning | evening | day  (default; a person may move it)
    sched: str  # freq | fixed | weekend | daily | sunday
    outdoor: bool = False
    partner: bool = False  # done with the key friend
    injury: bool = False  # can injure you
    venue: bool = False  # needs a local venue (moving away hurts, travel stops it)


STATEFUL: dict[str, Spec] = {
    "running": Spec("physical", "morning", "freq", outdoor=True, injury=True),
    "gym": Spec("physical", "evening", "freq", venue=True),
    "yoga class": Spec("class", "evening", "fixed", venue=True),
    "swimming": Spec("physical", "morning", "freq", venue=True),
    "climbing": Spec("physical", "evening", "freq", partner=True, venue=True, injury=True),
    "tennis": Spec("physical", "day", "weekend", outdoor=True, partner=True),
    "five-a-side": Spec("social", "evening", "fixed", outdoor=True, injury=True),
    "pub quiz": Spec("social", "evening", "fixed", venue=True),
    "choir": Spec("class", "evening", "fixed", venue=True),
    "cooking": Spec("chore", "evening", "daily"),
    "meal prep": Spec("chore", "day", "sunday"),
    "farmers market": Spec("outing", "day", "weekend", outdoor=True),
    "long walk": Spec("outing", "day", "weekend", outdoor=True),
    "cycling": Spec("physical", "day", "weekend", outdoor=True, injury=True),
    "pottery class": Spec("class", "evening", "fixed", venue=True),
    "visiting grandma": Spec("social", "day", "weekend", venue=True),
    "volunteering": Spec("social", "day", "fixed", venue=True),
    "church": Spec("social", "morning", "sunday", venue=True),
    "board games": Spec("social", "evening", "fixed", partner=True),
    "salsa class": Spec("class", "evening", "fixed", venue=True),
    "jiu-jitsu": Spec("physical", "evening", "fixed", venue=True, injury=True),
}

# Habits that nothing in this world moves: portable, cheap, done whatever the week is like.
COUNT_SCHED: dict[str, str] = {
    "reading": "daily", "duolingo": "daily", "journaling": "freq", "meditation": "daily",
    "family call": "fixed", "crossword": "daily", "stretching": "daily", "sketching": "freq",
}


@dataclass(frozen=True)
class Archetype:
    weight: float
    reg: tuple[float, float]  # regularity of the weekly rhythm
    soc: tuple[float, float]  # plans per day, roughly
    work: str  # "work" | "school" | ""
    crunch_on: tuple[float, float]
    crunch_off: tuple[float, float]
    sick: float  # multiplier on catching things
    trips: tuple[float, float]  # booked trips decided per day
    changes: tuple[str, ...]
    pool: tuple[str, ...]


ARCHETYPES: dict[str, Archetype] = {
    "office": Archetype(0.28, (0.55, 0.95), (0.25, 0.6), "work", (0.012, 0.035), (0.08, 0.18), 1.0,
                        (0.004, 0.02), ("new job", "moving", "relationship", "therapy", "health kick"),
                        ("running", "gym", "yoga class", "swimming", "climbing", "pub quiz", "cooking",
                         "meal prep", "five-a-side", "salsa class", "board games", "tennis", "cycling")),
    "student": Archetype(0.15, (0.25, 0.7), (0.4, 0.9), "school", (0.02, 0.05), (0.1, 0.2), 1.1,
                         (0.003, 0.015), ("semester", "relationship", "moving"),
                         ("gym", "running", "climbing", "pub quiz", "five-a-side", "board games",
                          "salsa class", "cooking", "long walk", "jiu-jitsu")),
    "shift": Archetype(0.10, (0.4, 0.8), (0.15, 0.4), "work", (0.01, 0.03), (0.08, 0.2), 1.1,
                       (0.003, 0.012), ("new job", "moving", "health kick"),
                       ("gym", "running", "swimming", "cooking", "five-a-side", "long walk",
                        "visiting grandma", "cycling")),
    "parent": Archetype(0.20, (0.5, 0.9), (0.08, 0.3), "work", (0.01, 0.03), (0.08, 0.18), 1.7,
                        (0.003, 0.012), ("moving", "new job", "health kick"),
                        ("running", "gym", "cooking", "meal prep", "church", "visiting grandma",
                         "long walk", "swimming", "yoga class", "farmers market")),
    "retiree": Archetype(0.10, (0.7, 0.98), (0.15, 0.45), "", (0.0, 0.0), (0.1, 0.1), 1.3,
                         (0.008, 0.03), ("moving", "health kick"),
                         ("swimming", "long walk", "choir", "farmers market", "church",
                          "volunteering", "pottery class", "cooking", "yoga class", "cycling",
                          "board games")),
    "freelancer": Archetype(0.17, (0.25, 0.7), (0.2, 0.6), "work", (0.03, 0.07), (0.15, 0.3), 1.0,
                            (0.005, 0.025), ("new job", "relationship", "moving", "therapy"),
                            ("running", "yoga class", "climbing", "cooking", "farmers market",
                             "pottery class", "cycling", "long walk", "gym", "board games")),
}

CHANGE_TOPIC = {"new job": "work", "moving": "home", "relationship": "relationships",
                "therapy": "health", "semester": "school", "health kick": "health"}
CHANGE_ANNOUNCED = {"new job": 0.85, "moving": 0.85, "semester": 0.95, "relationship": 0.15,
                    "therapy": 0.3, "health kick": 0.4}

FIRST_NAMES = ("Maya", "Sam", "Priya", "Leo", "Jonah", "Ana", "Tomás", "Kenji", "Ruth", "Dev",
               "Imani", "Oskar", "Lena", "Marcus", "Wen", "Farah", "Nico", "Beth", "Raj", "Zoe",
               "Callum", "Ines", "Kofi", "Hana", "Ollie", "Sofia", "Amir", "Greta", "Tariq", "Mei")
NOISE_TOPICS = ("tv", "news", "football", "music", "food", "pets", "gaming", "money", "books",
                "tech", "garden", "politics")
DESTINATIONS = ("Lisbon", "my sister's", "the coast", "Chicago", "a friend's wedding", "Edinburgh",
                "the lake house", "my parents'", "Montreal", "a conference", "the mountains",
                "Barcelona", "my cousin's", "Austin")

PLAN_KINDS: dict[str, tuple[str, ...]] = {
    "family": ("family dinner", "lunch", "brunch", "a visit"),
    "friend": ("dinner", "drinks", "coffee", "a movie", "a hike", "a concert", "game night",
               "brunch", "a picnic"),
    "partner": ("date night", "dinner", "a movie", "a hike"),
    "date": ("a date", "dinner", "drinks", "a hike"),
    "boss": ("lunch", "after-work drinks"),
}
OUTDOOR_KINDS = frozenset({"a hike", "a picnic"})


def pick(rng: np.random.Generator, seq: Sequence[str]) -> str:
    return seq[int(rng.integers(len(seq)))]


def day_phrase(lead: int, dow: int) -> str:
    if lead == 0:
        return "today"
    if lead == 1:
        return "tomorrow"
    if lead <= 6:
        return f"on {WEEKDAYS[dow]}"
    if lead == 7:
        return f"next {WEEKDAYS[dow]}"
    return f"in {lead} days"


# ---- activities --------------------------------------------------------------------------

# name: (query verb phrase, did phrases, skipped phrases, intention phrases)
ACT: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "running": ("go for a run", ("Went for a run.", "Got a {n}k run in.", "Ran a loop of the park."),
                ("No run today.", "Didn't run today."), ("Running tomorrow, that's the plan.", "Want to get a run in tomorrow.")),
    "gym": ("go to the gym", ("Hit the gym.", "Gym session done.", "Did legs at the gym."),
            ("No gym today.", "Didn't make it to the gym."), ("Gym tomorrow.", "Going to hit the gym tomorrow.")),
    "yoga class": ("make it to yoga class", ("Went to yoga.", "Yoga class was lovely."),
                   ("No yoga today.", "Missed yoga."), ("Yoga tomorrow, I think.", "Planning on yoga tomorrow.")),
    "swimming": ("go swimming", ("Swam {n} lengths.", "Got a swim in."),
                 ("No swim today.", "Didn't swim."), ("Swim tomorrow morning.", "Going to swim tomorrow.")),
    "climbing": ("go climbing", ("Went climbing.", "Climbing session, arms are dead."),
                 ("No climbing today.", "Didn't climb."), ("Climbing tomorrow hopefully.", "Want to climb tomorrow.")),
    "tennis": ("play tennis", ("Played tennis.", "Tennis this afternoon."),
               ("No tennis today.", "Didn't play tennis."), ("Tennis tomorrow?", "Hoping to play tennis tomorrow.")),
    "five-a-side": ("play five-a-side", ("Played five-a-side.", "Football with the lads."),
                    ("No five-a-side today.", "Didn't play football."), ("Five-a-side tomorrow.", "Football tomorrow night.")),
    "pub quiz": ("go to pub quiz", ("Pub quiz night.", "Did the quiz, came {n}th."),
                 ("No quiz today.", "Skipped the quiz."), ("Quiz tomorrow.", "Pub quiz tomorrow, I reckon.")),
    "choir": ("go to choir", ("Choir practice.", "Sang at choir tonight."),
              ("No choir today.", "Missed choir."), ("Choir tomorrow.", "Choir practice tomorrow.")),
    "cooking": ("cook dinner at home", ("Cooked dinner.", "Made a proper dinner at home."),
                ("Didn't cook tonight.", "No cooking, ate whatever."), ("Cooking tomorrow night.", "Going to cook tomorrow.")),
    "meal prep": ("do meal prep", ("Meal prepped for the week.", "Did the meal prep."),
                  ("No meal prep.", "Didn't meal prep."), ("Meal prep tomorrow.", "Prepping meals tomorrow.")),
    "farmers market": ("go to the farmers market", ("Went to the market.", "Farmers market haul."),
                       ("No market today.", "Didn't go to the market."), ("Market tomorrow.", "Hitting the market tomorrow.")),
    "long walk": ("go for a long walk", ("Long walk, {n} miles.", "Went for a long walk."),
                  ("No long walk today.", "Didn't get out for a walk."), ("Long walk tomorrow.", "Want a big walk tomorrow.")),
    "cycling": ("go for a bike ride", ("Went for a ride.", "Rode {n}k."),
                ("No ride today.", "Didn't cycle."), ("Ride tomorrow.", "Planning a ride tomorrow.")),
    "pottery class": ("go to pottery class", ("Pottery class.", "Threw a wonky bowl at pottery."),
                      ("No pottery today.", "Missed pottery."), ("Pottery tomorrow.", "Pottery class tomorrow.")),
    "visiting grandma": ("visit Grandma", ("Visited Grandma.", "Afternoon at Grandma's."),
                         ("Didn't see Grandma today.", "No visit to Grandma."), ("Seeing Grandma tomorrow.", "Visit to Grandma tomorrow.")),
    "volunteering": ("go volunteering", ("Volunteered at the food bank.", "Did my volunteering shift."),
                     ("No volunteering today.", "Missed my volunteering shift."),
                     ("Volunteering tomorrow.", "Food bank shift tomorrow.")),
    "church": ("go to church", ("Went to church.", "Church this morning."),
               ("No church today.", "Missed church."), ("Church tomorrow.", "Going to church tomorrow.")),
    "board games": ("play board games", ("Board game night.", "Played board games."),
                    ("No board games today.", "Didn't play any games."), ("Board games tomorrow.", "Game night tomorrow, maybe.")),
    "salsa class": ("go to salsa class", ("Salsa class!", "Danced at salsa."),
                    ("No salsa today.", "Missed salsa."), ("Salsa tomorrow.", "Salsa class tomorrow.")),
    "jiu-jitsu": ("go to jiu-jitsu", ("Jiu-jitsu, rolled for an hour.", "Went to BJJ."),
                  ("No jiu-jitsu today.", "Missed BJJ."), ("BJJ tomorrow.", "Jiu-jitsu tomorrow.")),
    "reading": ("read before bed", ("Read before bed.", "Read {n} pages tonight."),
                ("Didn't read tonight.", "No reading today."), ()),
    "duolingo": ("do their Duolingo", ("Did my Duolingo.", "Duolingo done, streak alive."),
                 ("No Duolingo today.", "Skipped Duolingo."), ()),
    "journaling": ("write in their journal", ("Wrote in my journal.", "Journaled for a bit."),
                   ("Didn't journal.", "No journaling today."), ()),
    "meditation": ("meditate", ("Meditated for {n} minutes.", "Did my meditation."),
                   ("Didn't meditate.", "No meditation today."), ()),
    "family call": ("do the family call", ("Called the family.", "Long call with the family."),
                    ("No family call today.", "Didn't call home."), ()),
    "crossword": ("do the crossword", ("Did the crossword.", "Finished the crossword."),
                  ("No crossword today.", "Didn't do the crossword."), ()),
    "stretching": ("do their stretches", ("Did my stretches.", "Stretched before bed."),
                   ("Didn't stretch.", "No stretching today."), ()),
    "sketching": ("sketch", ("Sketched for a while.", "Did a little drawing."),
                  ("No sketching today.", "Didn't sketch."), ()),
}

SKIP_REASONS = ("sick", "tired", "work", "weather", "plans", "away", "injury", "partner", "lazy")
_SKIP_REASON_TEXT = {
    "sick": ("still sick", "feeling ill"), "tired": ("too wiped", "no energy"),
    "work": ("stuck at work", "work ran late"), "weather": ("pouring rain", "weather was awful"),
    "plans": ("had plans", "other plans got in the way"), "away": ("out of town", "traveling"),
    "injury": ("nursing an injury", "my knee's still bad"), "partner": ("{key} couldn't make it", "{key} bailed"),
    "lazy": ("just couldn't be bothered", "didn't feel like it"),
}
CANCEL_REASONS = ("sick", "tired", "work", "weather", "away", "them")
_CANCEL_REASON_TEXT = {
    "sick": "I'm sick", "tired": "I'm wiped out", "work": "work blew up", "weather": "the weather's awful",
    "away": "I'm out of town", "them": "{who} cancelled",
}


def _fill(rng: np.random.Generator, s: str, **kw: str) -> str:
    return s.format(n=int(rng.integers(3, 12)), **kw)


def did_text(rng: np.random.Generator, act: str) -> str:
    return _fill(rng, pick(rng, ACT[act][1]))


def skipped_text(rng: np.random.Generator, act: str, reason: str | None, key: str) -> str:
    base = pick(rng, ACT[act][2])
    if reason is None:
        return base
    why = pick(rng, _SKIP_REASON_TEXT[reason]).format(key=key)
    return f"{base[:-1]} — {why}."


def intention_text(rng: np.random.Generator, act: str) -> str:
    return pick(rng, ACT[act][3])


def query_act_text(act: str, lead: int, dow: int) -> str:
    return f"Will they {ACT[act][0]} {day_phrase(lead, dow)}?"


# ---- plans with people -------------------------------------------------------------------


def plan_text(rng: np.random.Generator, kind: str, who: str, lead: int, dow: int, hedge: bool) -> str:
    when = "tonight" if lead == 0 else day_phrase(lead, dow)
    if hedge:
        return pick(rng, (f"Maybe {kind} with {who} {when}, we'll see.",
                          f"Tentatively {kind} with {who} {when}."))
    return pick(rng, (f"Made plans: {kind} with {who} {when}.", f"{kind.capitalize()} with {who} {when}.",
                      f"Booked in {kind} with {who} {when}."))


def plan_done_text(rng: np.random.Generator, kind: str, who: str) -> str:
    return pick(rng, (f"Had {kind} with {who}.", f"Good time with {who}, {kind} as planned."))


def plan_missed_text(rng: np.random.Generator, kind: str, who: str) -> str:
    return pick(rng, (f"{kind.capitalize()} with {who} didn't happen.", f"Ended up not doing {kind} with {who}."))


def cancel_text(rng: np.random.Generator, kind: str, who: str, reason: str) -> str:
    why = _CANCEL_REASON_TEXT[reason].format(who=who)
    return pick(rng, (f"Called off {kind} with {who}: {why}.", f"Had to cancel {kind} with {who} — {why}."))


def query_plan_text(kind: str, who: str, lead: int, dow: int) -> str:
    return f"Will {kind} with {who} {day_phrase(lead, dow)} happen?"


def plan_message_text(rng: np.random.Generator, who: str, content: str, kind: str, lead: int, dow: int) -> str:
    when = day_phrase(lead, dow)
    if content == "doubt":
        return pick(rng, (f"{who} says they might have to push {kind} {when}.",
                          f"{who}: not sure I can make {kind} {when}, will confirm."))
    return pick(rng, (f"{who} is excited for {kind} {when}.", f"{who} confirmed {kind} {when}."))


# ---- mood and talk -----------------------------------------------------------------------

_MOOD = {
    1: ("Awful day. Just want it over.", "Really low today.", "Everything feels heavy."),
    2: ("Pretty rough day.", "Not great, honestly.", "Bit of a slog today."),
    3: ("Okay day. Nothing special.", "Meh. Fine.", "Middling mood today."),
    4: ("Good day overall.", "Feeling pretty good.", "Decent mood today."),
    5: ("Fantastic day!", "Feeling on top of the world.", "Best I've felt in ages."),
}


def mood_text(rng: np.random.Generator, level: int) -> str:
    return pick(rng, _MOOD[level])


_TALK: dict[tuple[str, str], tuple[str, ...]] = {
    ("work", "stressed"): ("Work is crushing me right now.", "So much on my plate at work."),
    ("work", "busy"): ("Busy day at work.", "Back-to-back meetings all day."),
    ("work", "fine"): ("Work was fine.", "Decent day at work."),
    ("work", "quiet"): ("Work's quiet this week.", "Slow day at work."),
    ("school", "stressed"): ("Exams are killing me.", "Deadlines everywhere this week."),
    ("school", "busy"): ("Lots of reading for class.", "Long day of lectures."),
    ("school", "fine"): ("Class was okay.", "Decent day at uni."),
    ("school", "quiet"): ("Easy week at school.", "Barely any classes today."),
    ("health", "sick"): ("Think I'm coming down with something.", "Sick in bed today."),
    ("health", "tired"): ("So tired today.", "Running on empty."),
    ("health", "great"): ("Feeling strong today.", "Body feels great."),
    ("health", "sore"): ("My knee's still bugging me.", "Back's sore again."),
    ("weather", "rain"): ("Pouring all day.", "Grey and wet out."),
    ("weather", "sunny"): ("Gorgeous day out!", "Sun's finally out."),
}


def talk_text(rng: np.random.Generator, topic: str, content: str) -> str:
    return pick(rng, _TALK[(topic, content)])


def keen_text(rng: np.random.Generator, act: str, content: str) -> str:
    if content == "loving":
        return pick(rng, (f"Really loving {act} lately.", f"Can't get enough of {act} these days."))
    return pick(rng, (f"Should really get back into {act}.", f"Haven't done much {act} lately, feel bad."))


def key_message_text(rng: np.random.Generator, who: str, content: str, act: str) -> str:
    if content == "swamped":
        return pick(rng, (f"{who} says they're swamped this week.", f"{who}: drowning at the moment, rain check?"))
    if content == "invite":
        return pick(rng, (f"{who} asked if I'm up for {act} this week.", f"{who} wants to do {act} soon."))
    return pick(rng, (f"{who} sent a meme.", f"Texting with {who} about nothing much."))


def boss_message_text(rng: np.random.Generator, who: str, content: str) -> str:
    if content == "deadline":
        return pick(rng, (f"{who} moved the deadline up.", f"{who} needs the report by end of week."))
    return pick(rng, (f"{who} sent round the meeting notes.", f"{who} shared the schedule for next month."))


_NOISE = {
    "tv": ("Started a new show, three episodes in.", "That finale was a mess.", "New season drops this week!"),
    "news": ("Read a long piece about housing.", "The news is exhausting today.", "Some good news for once."),
    "football": ("Watched the highlights.", "We lost again. Unbelievable.", "What a win last night!"),
    "music": ("Got a new album on repeat.", "Concert tickets are so expensive now.", "Found a band I love."),
    "food": ("Tried a new recipe.", "Burned the rice. Again.", "Best tacos I've had in months."),
    "pets": ("The cat knocked everything off the desk.", "Vet bill was brutal.", "The dog learned a new trick."),
    "gaming": ("Played a bit of that new game.", "Servers down all evening.", "Finally beat the boss."),
    "money": ("Sorting out my budget.", "Rent's going up again.", "Got a little tax refund."),
    "books": ("Halfway through a thriller.", "That ending made no sense.", "My library hold came in."),
    "tech": ("Phone update moved everything.", "Laptop crashed mid-email.", "New headphones arrived."),
    "garden": ("Repotted the basil.", "Slugs ate the lettuce.", "First tomatoes!"),
    "politics": ("Argued about the election at lunch.", "Can't watch the debates anymore.", "Local vote went our way."),
    "family": ("Mum sent photos from her trip.", "Family group chat is chaos.", "My cousin's engaged!"),
    "home": ("Finally fixed the leaky tap.", "Neighbours are drilling again.", "Got a new plant for the hall."),
}
NOISE_CONTENTS = ("chat", "rant", "excited")


def noise_text(topic: str, content: str) -> str:
    return _NOISE[topic][NOISE_CONTENTS.index(content)]


def chat_text(rng: np.random.Generator, who: str) -> str:
    return pick(rng, (f"{who} sent a funny video.", f"Caught up with {who} over text.", f"{who} asked how I'm doing."))


# ---- trips, holidays, changes ------------------------------------------------------------


def trip_plan_text(rng: np.random.Generator, lead: int, days: int) -> str:
    dest = pick(rng, DESTINATIONS)
    return f"Booked a trip to {dest}: leaving in {lead} days, away for {days} days."


def trip_start_text(rng: np.random.Generator) -> str:
    return pick(rng, ("Off on the trip today!", "Travel day. Airport snacks.", "Made it, trip's begun."))


def trip_cancel_text(rng: np.random.Generator) -> str:
    return pick(rng, ("Trip's off, staying home after all.", "Had to cancel the trip."))


def spontaneous_trip_text(rng: np.random.Generator, days: int) -> str:
    dest = pick(rng, DESTINATIONS)
    return f"Last-minute getaway to {dest}, back in {days} days."


def travel_noise_text(rng: np.random.Generator) -> str:
    return pick(rng, ("Exploring today.", "Great food here.", "Hotel bed is terrible.", "Long day sightseeing."))


def holiday_text(lead: int, dow: int) -> str:
    return f"Long weekend coming up: the {WEEKDAYS[dow]} {lead} days from now is a public holiday."


_CHANGE = {
    "new job": ("Got the new job! I start in {lead} days.", "Wrapping things up at the old job.",
                "First day at the new job!", "Still getting used to the new job."),
    "moving": ("Signed the lease, moving in {lead} days.", "Packing boxes again.",
               "Moved into the new place!", "Still learning the new neighbourhood."),
    "relationship": ("{who} and I are moving in together in {lead} days!", "Nervous about moving in with {who}.",
                     "Things just got serious with {who}.", "Loving having {who} around."),
    "therapy": ("Booked therapy, first session in {lead} days.", "Therapy starts soon.",
                "Had my first therapy session.", "Therapy's helping, I think."),
    "semester": ("Semester starts in {lead} days.", "Buying textbooks for the semester.",
                 "First day of the semester!", "Settling into the new timetable."),
    "health kick": ("Doctor's orders: new health regime starts in {lead} days.", "Gearing up for the health kick.",
                    "Started my health kick today.", "Sticking with the new routine."),
}
CHANGE_CONTENTS = ("announce", "upcoming", "started", "new_routine")


def change_text(kind: str, content: str, lead: int = 0, who: str = "") -> str:
    return _CHANGE[kind][CHANGE_CONTENTS.index(content)].format(lead=lead, who=who)


# ---- questions ---------------------------------------------------------------------------

QUESTION_OPTIONS: dict[str, tuple[str, ...]] = {
    "health": ("good", "tired", "sick"),
    "mood": ("good", "okay", "low"),
    "work": ("calm", "busy", "slammed"),
    "injury": ("none", "minor", "injured"),
    "change": ("nothing_new", "something_coming", "big_change"),
    "habit": ("loving_it", "regular", "fallen_off"),
    "friend": ("around", "busy"),
    "plan": ("yes", "probably", "unsure"),
}
QUESTION_COST = {"health": 0.08, "mood": 0.15, "work": 0.1, "injury": 0.06, "change": 0.12,
                 "habit": 0.06, "friend": 0.07, "plan": 0.05}


def question_text(kind: str, subject: str, extra: str = "") -> str:
    return {
        "health": "How are you feeling physically these days?",
        "mood": "How have you been feeling in yourself lately?",
        "work": f"How's {subject} going this week?",
        "injury": "Any aches, pains or injuries at the moment?",
        "change": "Anything new going on in your life?",
        "habit": f"How's {subject} going for you lately?",
        "friend": f"Is {subject} around much these days?",
        "plan": f"Is {extra} still on?",
    }[kind]


_ANSWER = {
    ("health", "good"): "Feeling good, thanks.", ("health", "tired"): "Pretty tired, honestly.",
    ("health", "sick"): "I'm sick at the moment.",
    ("mood", "good"): "Really good lately.", ("mood", "okay"): "Okay, up and down.",
    ("mood", "low"): "Pretty low, if I'm honest.",
    ("work", "calm"): "Calm, actually.", ("work", "busy"): "Busy but fine.",
    ("work", "slammed"): "Completely slammed.",
    ("injury", "none"): "Nope, all good.", ("injury", "minor"): "Bit of a niggle.",
    ("injury", "injured"): "Yeah, I'm injured right now.",
    ("change", "nothing_new"): "Nothing much new.", ("change", "something_coming"): "Something's coming up, yeah.",
    ("change", "big_change"): "Big changes lately!",
    ("habit", "loving_it"): "Loving it!", ("habit", "regular"): "Ticking along.",
    ("habit", "fallen_off"): "I've fallen off it, to be honest.",
    ("friend", "around"): "Yeah, we see each other a lot.", ("friend", "busy"): "They're really busy lately.",
    ("plan", "yes"): "Yes, definitely on.", ("plan", "probably"): "Probably, I think.",
    ("plan", "unsure"): "Not sure, it's up in the air.",
}


def answer_text(kind: str, option: str) -> str:
    return _ANSWER[(kind, option)]
