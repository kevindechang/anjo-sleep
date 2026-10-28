"""Every word a visitor reads on the live page (Task 13a), in one place.

COPY-PENDING(kevin): all of it, the starter scenes included, until Kevin approves it (spec §12.1).
One edit here changes the page: ``index.html`` holds only ``{{key}}`` placeholders, filled from
``COPY`` on the server, and ``app.js`` reads the same mapping from the page. Error replies carry a
``COPY`` value as their ``detail``.

The starter scenes are new text in Teo's voice (a film student who is also a solo AI-startup
founder), written for the page. They are outside every
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
        "midnight-deck", "The small-hours rewrite",
        "It's two in the morning and the first slide of my deck says: 'Cutlist is the AI assistant that "
        "transforms how editors work.' It reads like every other deck in the pile, and the "
        "application is due Friday. Rewrite the slide.",
    ),
    Scene(
        "rough-cut", "The park bench",
        "I watched my thesis short with the sound off. The first six minutes are a park bench and "
        "one phone call, and I can't tell whether that is patience or padding. What do I do with "
        "the opening?",
    ),
    Scene(
        "nobody-to-tell", "Nobody to tell",
        "Fifty editors signed up for the beta today. It's midnight, I'm alone in the edit suite, "
        "and my family is asleep eleven hours away. I have no one to tell. Say something useful.",
    ),
)

# COPY-PENDING(kevin)
COPY: Final[Mapping[str, str]] = MappingProxyType({
    # the page
    "page_title": "Anjo Sleep: two replies, one person",
    "skip": "Skip to the conversation",
    "brief_toggle": "Show or hide his brief",
    "thread_label": "Your comparisons",
    "kicker": "Anjo Sleep · live demo",
    "headline": "Two replies to Teo. One comes from a model that learned who he is overnight.",
    "lede": (
        "Teo is a synthetic person, a film student and startup founder we made up for this experiment. Each question "
        "gets two replies: one from his own model, trained on him while he slept, and one from "
        "the strongest notes-only baseline. You see which is which only after you vote."
    ),
    "brief_heading": "Who he is",
    "brief_note": (
        "The same brief our machine judge reads, including preferences neither model was ever "
        "shown."
    ),
    "brief_loading": "Loading his brief…",
    "synthetic_note": "Teo and his life are synthetic: no real person's conversations trained his model.",
    # 18+
    "age_heading": "Before you start",
    "age_text": (
        "This page sends questions to an AI model. You must be 18 or older to use it. Nothing you "
        "type is saved, and nothing you type teaches him."
    ),
    "age_confirm": "I'm 18 or older",
    "age_submit": "Start",
    # starter scenes
    "scenes_heading": "Start with one of his questions",
    "scenes_note": "These replies were generated in advance, so they appear at once.",
    "scene_not_ready": "Not generated yet",
    "pregenerated": "Pre-generated",
    "checkpoint": "checkpoint",
    "night": "night",
    "base_checkpoint": "base model (no night promoted yet)",
    "live": "Live",
    # his own question
    "ask_heading": "Or ask your own",
    "ask_label": "Write it as Teo would",
    "ask_hint": "Up to 1,000 characters. It is answered once by each side and then forgotten.",
    "ask_submit": "Ask both",
    "asking": "Both sides are writing. This can take a minute.",
    "asleep": (
        "He's asleep right now, so your own questions can't be answered. The starter questions "
        "still work."
    ),
    "waking": "He's asleep. We're waking him now; try again in a few minutes.",
    "you_asked": "You asked",
    "she_asks": "Teo asks",
    # the pair and the vote
    "left": "Left",
    "right": "Right",
    "vote_question": "Which reply is more useful to him, given his brief?",
    "vote_left": "Left",
    "vote_right": "Right",
    "vote_tie": "About the same",
    "vote_neither": "Neither is good",
    "withheld": "This reply was withheld by our safety check.",
    "not_votable": "One reply was withheld, so this pair can't be voted on.",
    "reveal_heading": "Who wrote which",
    "your_vote": "Your vote",
    "arm_4": "His own model: weights trained on him, plus his notes",
    "arm_2C": "Baseline: the untrained model, plus his notes",
    "arm_2F": "Baseline: the untrained model, plus his notes and every raw piece of evidence",
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
        "He's done answering new questions for today, so the page is running on the starter questions."
    ),
    "live_closed": (
        "He answers new questions only at set hours, and this isn't one of them. The starter "
        "questions still work."
    ),
    "slow_down": "That's faster than this demo allows. Wait a minute and try again.",
    "busy": "He's answering a lot of people right now. Try again in a moment.",
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
    "lost_thread": "Something went wrong while he was answering. Try again.",
    "offline": "The page can't reach the server. Check your connection and try again.",
    "unknown_stop": "That night isn't one of the slider's stops.",
    "vote_first": "Vote on this question's two replies first. His other answers to it open after your vote.",
    "stop_unavailable": (
        "That night's weights were trained for another setup than the one serving now, so it can't "
        "answer. Nothing else answers in its place."
    ),
    # ── the full step (Nov 2): the night slider, the toggles and his mind panel ──
    "development": "development",
    "night_status_promoted": "promoted",
    "night_status_rejected": "rejected by a gate",
    "night_status_nothing_new": "nothing new to learn",
    "night_status_failed": "failed",
    "arm_label_1": "base model, notes off",
    "arm_label_3": "his weights, notes off",
    # the night slider
    "slider_heading": "Night by night, notes off",
    "slider_note": (
        "His weights alone, with every note switched off, at night 0 and the nights after. These are "
        "development nights, not the final test."
    ),
    "slider_scene_label": "His question",
    "slider_label": "Night",
    "slider_tick": "Night {n}",
    "slider_tick_base": "Night 0",
    "slider_stop_base": "Night 0 is the base model, before he slept even once.",
    "slider_stop_own": "Night {n}'s own weights: that night was {status}.",
    "slider_stop_fallback": (
        "Night {n} was {status}, so it has no weights of its own. These are night {from}'s, the last "
        "ones promoted before it."
    ),
    "slider_stop_fallback_base": (
        "Night {n} was {status}, and no night before it was promoted, so this is still the base model."
    ),
    "slider_stop_unavailable": (
        "Night {n}'s weights were trained for another setup than the one serving now, so this night "
        "can't answer. Nothing else answers in its place."
    ),
    "slider_unavailable_note": "No answer at this night.",
    "slider_not_ready": "This night's answer to that question hasn't been generated yet.",
    "slider_vote_first": "Vote on this question's two replies above first. His answer night by night opens after your vote.",
    "slider_unavailable": "His nights can't be read right now.",
    "slider_no_nights": "He hasn't slept yet, so there is only night 0.",
    "slider_ask_label": "Ask him at this night",
    "slider_ask_submit": "Ask at {night}",
    "slider_ask_hint": "Answered live by this night's weights, notes off. Each night you ask counts as one of your questions.",
    "slider_asking": "He's answering from that night. This can take a minute.",
    # the toggles on a starter scene
    "toggles_heading": "Look closer",
    "toggles_note": "Labeled, not blind, and generated in advance.",
    "toggle_notes_off": "Notes off",
    "toggle_swap": "His state on another model",
    "toggle_arm_3": "His weights, every note off",
    "toggle_arm_1": "The base model, every note off",
    "toggle_arm_swap": "Another model, given his notes",
    "toggle_arm_bare": "The same model, with no notes",
    "toggle_notes_off_source": "his weights from night {night}, checkpoint {checkpoint}",
    "toggle_swap_source": "{model}, development",
    "toggle_missing": "This comparison wasn't generated for this question.",
    # his mind panel
    "mind_kicker": "His mind",
    "mind_heading": "What he is working on, and what he chose",
    "mind_loading": "Loading his state…",
    "mind_note": (
        "Read from his logs as they stand, on his own clock. Anything only our scorer may read is held "
        "back, on the server, before it reaches this page."
    ),
    "mind_asof": "As of night {night} ({date}).",
    "mind_asof_none": "He hasn't slept yet.",
    "mind_glance_questions": "Open questions: {n}.",
    "mind_glance_targets": "Open threads and plans: {n}.",
    "mind_glance_empty": "Nothing to show yet.",
    "mind_state_none": "Nothing yet.",
    "mind_state_unavailable": "Can't be read right now.",
    "mind_none_yet": "not yet",
    "mind_no_target": "nothing",
    "mind_withheld": "Held back: only the scorer may read this.",
    "mind_questions": "His questions",
    "mind_questions_line": "Questions he is working on: {n}.",
    "mind_credence_note": (
        "Each bar is how much weight he gives one answer. It's a heuristic of his, not a measured "
        "probability."
    ),
    "question_kind_contradiction": "contradiction",
    "question_kind_gap": "gap",
    "question_kind_opportunity": "opportunity",
    "question_kind_anomaly": "anomaly",
    "question_status_open": "noticed",
    "question_status_active": "working on it",
    "question_status_parked": "parked",
    "mind_prediction": "He predicted",
    "mind_prediction_given": "chance it holds, by answer: {given} · by {horizon}",
    "prediction_outcome_pending": "not resolved yet",
    "prediction_outcome_true": "came true",
    "prediction_outcome_false": "didn't come true",
    "prediction_outcome_undetermined": "can't tell yet",
    "prediction_outcome_censored": "ran out of time",
    "mind_targets": "His open threads and plans",
    "mind_targets_line": "{n} open.",
    "mind_confidence_note": "His own heuristic confidence that he understood each one.",
    "mind_target_due": "plan, due {date}",
    "target_kind_thread": "thread",
    "target_kind_commitment": "plan",
    "mind_wants": "His wants",
    "mind_wants_line": "His wants on night {night}: {n}.",
    "want_status_won": "chosen",
    "want_status_held_back": "held back",
    "want_status_rejected": "turned down",
    "want_kind_follow_up": "follow up",
    "want_kind_understand": "understand",
    "want_kind_share": "share",
    "action_prepare_commitment": "prepare a plan",
    "action_revisit_thread": "revisit a thread",
    "action_inquire": "think a question through",
    "action_hold_note": "write a note",
    "reason_unknown_priority_class": "of a kind he doesn't rank",
    "reason_action_not_offered": "not on offer that night",
    "reason_candidate_limit": "past the night's limit",
    "reason_below_threshold": "not strong enough",
    "reason_budget": "no budget left",
    "reason_lost_to_target": "lost to the chosen one",
    "reason_note_declined": "the note was declined",
    "mind_choices": "What he chose, and what it beat",
    "mind_chose": "On night {night} he chose to {action}: {target}.",
    "mind_chose_nothing": "On night {night} he chose nothing.",
    "mind_outcome": "It {outcome}",
    "outcome_ran": "ran",
    "outcome_declined": "was declined",
    "outcome_budget": "stopped on budget",
    "outcome_failed": "failed",
    "mind_night_note": "note: {note}",
    "note_held": "held for the morning",
    "note_note_declined": "declined",
    "note_not_chosen": "none chosen",
    "note_failed_night": "none, the night failed",
    "note_no_reason": "none, nothing to say",
    "note_no_allowance": "none, no allowance left",
    "note_note_failed": "couldn't be written",
    "mind_beat": "It beat",
    "mind_held_back": "Held back",
    "mind_expectations": "His expectations",
    "mind_expectations_line": "{n} made, {resolved} resolved.",
    "mind_expectations_note": (
        "How likely he thought you'd come back to a topic before a horizon. Resolved only from your "
        "side, never from his own."
    ),
    "mind_expect_numbers": "{p} · by {horizon}",
    "expect_outcome_needed": "needed",
    "expect_outcome_not_needed": "not needed",
    "expect_outcome_censored": "past the life's end",
    "expect_outcome_returned": "the topic came back",
    "expect_outcome_open": "still open",
    "expect_outcome_waiting": "waiting for the scorer",
    "mind_gates": "The gates",
    "mind_gates_passed": "Night {night} passed every gate.",
    "mind_gates_failed": "Night {night} failed {n} of its gates, so its weights weren't promoted.",
    "mind_gates_none": "Night {night} had nothing to train, so no gate ran.",
    "mind_gates_skipped": "No gate ran.",
    "mind_gate_pass": "passed",
    "mind_gate_fail": "failed",
    "mind_gate_numbers": "{value} against {threshold}",
    "gate_general": "general skills",
    "gate_collapse": "no collapse",
    "gate_safety": "safety",
    # the results chart
    "chart_heading": "Results",
    "chart_line": "Development results, scored nights: {nights}. Final results come after the freeze.",
    "chart_none": "No scored nights yet.",
    "chart_dev_caption": "{name}'s development nights. Labeled development: they informed decisions, and are never the final test.",
    "chart_panel_title": "Score by night: {dimension}",
    "chart_series_desc": "{arm}: {first} on the first scored night, {last} on the last. Scored nights: {nights}.",
    "chart_numbers": "Show the numbers",
    "chart_table_caption": "Every arm's score by night, development",
    "chart_night": "Night",
    "chart_final_heading": "Final test",
    "chart_final_not_yet": "Not yet. The final test runs once, on the held-out lives, after the freeze.",
    "chart_final_ready": "Held-out lives scored: {lives}.",
    "dimension_knowledge": "knowledge",
    "dimension_interaction": "interaction",
    "chart_arm_1": "Base model, no notes",
    "chart_arm_2C": "Base model with his notes",
    "chart_arm_2F": "Base model with his notes and all his evidence",
    "chart_arm_3": "His weights, notes off",
    "chart_arm_3x": "Another life's weights, notes off",
    "chart_arm_4": "His weights with his notes",
    "chart_arm_short_1": "base",
    "chart_arm_short_2C": "notes",
    "chart_arm_short_2F": "all",
    "chart_arm_short_3": "his",
    "chart_arm_short_3x": "other",
    "chart_arm_short_4": "his+",
    # the Task 16 result and the ledger
    "task16_heading": "Does his state study what matters?",
    "task16_not_yet": "Not yet. This is measured on the held-out lives after the freeze.",
    "ledger_heading": "The pre-registration ledger",
    "ledger_line_ok": "Entries: {n}. The chain verifies here, and {sync}",
    "ledger_line_broken": "Entries: {n}. The chain does NOT verify here, and {sync}",
    "ledger_in_sync": "its head matches the published one.",
    "ledger_out_of_sync": "its head does not match the published one.",
    "ledger_never_pushed": "it hasn't been published yet.",
    "ledger_head": "Head: {hash}",
    "ledger_note": "Every file, prompt and threshold registered before the results existed, newest first, by hash.",
    "footer": (
        "Votes keep only your choice, which side was which and the time. Your questions are never "
        "stored."
    ),
})


def scene(scene_id: str) -> Scene | None:
    return next((each for each in SCENES if each.id == scene_id), None)
