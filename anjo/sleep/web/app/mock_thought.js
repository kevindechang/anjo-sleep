// Development only: her line of thought for ?mock=1, the engine's own output for a synthetic life
// (Mira, a film student, over 5 nights: anjo.sleep.mind.thought_demo), as it wrote it. Loaded only
// by the fake API, which applies the judge's disputes and hides on top. Never a real person.

export const THOUGHT = Object.freeze({
 "version": 1,
 "as_of": "2026-10-31T23:00:00+00:00",
 "credence_note": "Her own estimates, not calibrated odds.",
 "hunches": [
  {
   "id": "p7",
   "question_id": "q5",
   "plain": "About 7 in 10: Mira asks her tutor for more time this week. I'll know by Nov 3.",
   "exact": "Mira asks her tutor for more time this week",
   "about": {
    "text": "Will the edit be ready for Friday?",
    "withheld": false
   },
   "claim": {
    "text": "Mira asks her tutor for more time this week",
    "withheld": false
   },
   "p": 0.7416243654822335,
   "odds": "about 7 in 10",
   "due": "2026-11-03",
   "due_in_days": 3,
   "disputed": false
  },
  {
   "id": "p13",
   "question_id": "q11",
   "plain": "About 6 in 10: Mira looks up the festival's entry fee. I'll know by Nov 5.",
   "exact": "Mira looks up the festival's entry fee",
   "about": {
    "text": "Will Mira enter the March festival?",
    "withheld": false
   },
   "claim": {
    "text": "Mira looks up the festival's entry fee",
    "withheld": false
   },
   "p": 0.5820000000000001,
   "odds": "about 6 in 10",
   "due": "2026-11-05",
   "due_in_days": 5,
   "disputed": false
  }
 ],
 "settled": [
  {
   "id": "p8",
   "question_id": "q5",
   "plain": "Mira says the grade is done by Wednesday: it didn't happen. I had it at about 6 in 10, so I got this one wrong.",
   "exact": "Mira says the grade is done by Wednesday",
   "claim": {
    "text": "Mira says the grade is done by Wednesday",
    "withheld": false
   },
   "p_at_bet": 0.606,
   "outcome": "false",
   "verdict": "wrong",
   "settled_at": "2026-10-31T23:00:00+00:00"
  },
  {
   "id": "p4",
   "question_id": "q1",
   "plain": "The shoot happens on Saturday: it happened. I had it at about 7 in 10, so I called it.",
   "exact": "The shoot happens on Saturday",
   "claim": {
    "text": "The shoot happens on Saturday",
    "withheld": false
   },
   "p_at_bet": 0.745,
   "outcome": "true",
   "verdict": "right",
   "settled_at": "2026-10-30T23:00:00+00:00"
  },
  {
   "id": "p3",
   "question_id": "q1",
   "plain": "Mira mentions a backup location before Friday: it happened. I had it at about 6 in 10, so I called it.",
   "exact": "Mira mentions a backup location before Friday",
   "claim": {
    "text": "Mira mentions a backup location before Friday",
    "withheld": false
   },
   "p_at_bet": 0.55,
   "outcome": "true",
   "verdict": "right",
   "settled_at": "2026-10-29T23:00:00+00:00"
  }
 ],
 "scorecard": {
  "scored": 3,
  "right": 2,
  "hit_rate": 0.6666666666666666,
  "brier": 0.21158699999999997,
  "enough": false,
  "bins": [
   {
    "from": 0.5,
    "to": 0.7,
    "n": 2,
    "came_true": 1
   },
   {
    "from": 0.7,
    "to": 0.9,
    "n": 1,
    "came_true": 1
   }
  ],
  "ran_out_of_time": 0
 },
 "question": {
  "id": "q11",
  "question_id": "q11",
  "plain": "Quick question: Will Mira enter the March festival? I'm torn between two answers.",
  "exact": "Will Mira enter the March festival?",
  "ask": {
   "text": "Will Mira enter the March festival?",
   "withheld": false
  },
  "why": "close_call",
  "choices": [
   {
    "id": "q11.h12a",
    "text": {
     "text": "She submits for March",
     "withheld": false
    },
    "credence": 0.4
   },
   {
    "id": "q11.h12b",
    "text": {
     "text": "She waits for the autumn round",
     "withheld": false
    },
    "credence": 0.38
   },
   {
    "id": "q11.else",
    "text": {
     "text": "Something else",
     "withheld": false
    },
    "credence": 0.22
   }
  ],
  "disputed": false
 },
 "morning": {
  "since": "2026-10-30T00:00:00+00:00",
  "settled": [
   "p4",
   "p8"
  ],
  "new_hunches": [
   "p13"
  ],
  "shifts": [
   {
    "id": "shift:q1",
    "question_id": "q1",
    "plain": "I'm surer now: 57% to 96% on “Will Saturday's courtyard shoot survive the rain?”. I now lean to “It moves indoors to the kitchen”.",
    "exact": "It moves indoors to the kitchen",
    "about": {
     "text": "Will Saturday's courtyard shoot survive the rain?",
     "withheld": false
    },
    "now_leading": {
     "text": "It moves indoors to the kitchen",
     "withheld": false
    },
    "from": 0.5727272727272726,
    "to": 0.9569241723033107
   },
   {
    "id": "shift:q5",
    "question_id": "q5",
    "plain": "I changed my mind on “Will the edit be ready for Friday?”. I now lean to “She asks her tutor for more time”.",
    "exact": "She asks her tutor for more time",
    "about": {
     "text": "Will the edit be ready for Friday?",
     "withheld": false
    },
    "now_leading": {
     "text": "She asks her tutor for more time",
     "withheld": false
    },
    "from": 0.4,
    "to": 0.7106598984771574
   }
  ],
  "findings": [
   {
    "id": "f10@22",
    "question_id": "q1",
    "kind": "resolution",
    "plain": "I worked it out: Bad weather moves her shoots indoors, not later.",
    "exact": "Bad weather moves her shoots indoors, not later",
    "statement": {
     "text": "Bad weather moves her shoots indoors, not later",
     "withheld": false
    },
    "p": 0.9569241723033107
   }
  ],
  "empty": false,
  "plain": "Since you were last here: 2 of my bets settled (1 right), 1 new hunch, I changed my mind 2 times, 1 conclusion."
 },
 "hidden": 0
});
