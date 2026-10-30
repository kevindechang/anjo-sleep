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
# Dailies' before and after (``dailies``): the one scene each morning's card asks both checkpoints.
# It is not a starter scene, so showing its answers never unblinds a starter scene's pair
# (``service``'s ``vote_first``); the decontamination test holds it to the same rules.
DAILY_SCENE: Final = Scene(
    "export-bug", "The apology email",
    "The beta's export bug cost one of my editors three hours of work tonight. I have to email her "
    "before I sleep, and I don't want it to read like a support ticket. How do I open it?",
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
        "type is saved, and nothing you type teaches her."
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
        "She's asleep right now, so your own questions can't be answered. The starter questions "
        "still work."
    ),
    "waking": "She's asleep. We're waking her now; try again in a few minutes.",
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
    "arm_4": "His own model: her weights trained on him, plus her notes on him",
    "arm_2C": "Baseline: the untrained model, plus her notes on him",
    "arm_2F": "Baseline: the untrained model, plus her notes on him and every raw piece of evidence",
    "prompt_tokens": "Prompt length",
    "tokens": "tokens",
    "cost": "Cost per turn",
    "cost_unmeasured": "not measured",
    # the person's own model (Task 9): what changed, undo and delete
    "yours_kicker": "Yours",
    "yours_heading": "Your model, night by night",
    "yours_note": "These controls are yours alone: they read and change only your own sleep data.",
    "yours_loading": "Reading your model…",
    "yours_retry": "Try again",
    "yours_unavailable": "Sign in to the account Anjo Sleep runs for to use these controls.",
    "yours_nights": "Your nights",
    "yours_nights_empty": "No night has finished yet.",
    "yours_changes": "What changed",
    "yours_changes_empty": "Nothing changed in that time.",
    "yours_since": "Since",
    "yours_since_hint": "Blank shows the last seven days.",
    "yours_undo": "Undo last night",
    "yours_undo_ask": "Undo the last night, and serve the version before it?",
    "yours_undo_yes": "Yes, undo it",
    "yours_undo_no": "Keep it",
    "yours_undo_done": "Undone. The version before night {night} serves again.",
    "yours_undo_hint": "The evidence stays; the next night trains from the restored version.",
    "yours_delete": "Delete my sleep data",
    "yours_delete_ask": "Delete every night, your weights and the evidence behind them? This can't be undone.",
    "yours_delete_yes": "Yes, delete everything",
    "yours_delete_no": "Keep my data",
    "yours_delete_done": "Your sleep data is deleted.",
    "yours_delete_hint": "Weights, evidence and every night record go. This can't be undone.",
    "yours_failed": "That didn't work. Try again.",
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
    "unknown_stop": "That night isn't one of the slider's stops.",
    "vote_first": "Vote on this question's two replies first. Her other answers to it open after your vote.",
    "stop_unavailable": (
        "That night's weights were trained for another setup than the one serving now, so it can't "
        "answer. Nothing else answers in its place."
    ),
    # ── the full step (Nov 2): the night slider, the toggles and her mind panel ──
    "development": "development",
    "night_status_promoted": "promoted",
    "night_status_rejected": "rejected by a gate",
    "night_status_nothing_new": "nothing new to learn",
    "night_status_failed": "failed",
    "night_status_napped": "napped",
    # how a night slept (night/sleep_pressure.py): a nap runs no GPU step, a deep sleep is the full night
    "sleep_kind_nap": "napped",
    "sleep_kind_deep": "deep sleep",
    "sleep_note": "pressure {pressure} against {threshold}: {reason}",
    "sleep_reason_forced": "a deep sleep was asked for",
    "sleep_reason_max_days": "the days since her last deep sleep reached the limit",
    "sleep_reason_pressure": "it reached the threshold",
    "sleep_reason_below_threshold": "under the threshold",
    "sleep_reason_too_little_new": "too little new since her last deep sleep",
    "sleep_reason_month_cap": "a deep sleep would pass the month's budget",
    "arm_label_1": "base model, notes off",
    "arm_label_3": "her weights, notes off",
    # the night slider
    "slider_heading": "Night by night, notes off",
    "slider_note": (
        "Her weights alone, with every note switched off, at night 0 and the nights after. These are "
        "development nights, not the final test."
    ),
    "slider_scene_label": "His question",
    "slider_label": "Night",
    "slider_tick": "Night {n}",
    "slider_tick_base": "Night 0",
    "slider_stop_base": "Night 0 is the base model, before she slept even once.",
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
    "slider_vote_first": "Vote on this question's two replies above first. Her answer night by night opens after your vote.",
    "slider_unavailable": "Her nights can't be read right now.",
    "slider_no_nights": "She hasn't slept yet, so there is only night 0.",
    "slider_ask_label": "Ask her at this night",
    "slider_ask_submit": "Ask at {night}",
    "slider_ask_hint": "Answered live by this night's weights, notes off. Each night you ask counts as one of your questions.",
    "slider_asking": "She's answering from that night. This can take a minute.",
    # the toggles on a starter scene
    "toggles_heading": "Look closer",
    "toggles_note": "Labeled, not blind, and generated in advance.",
    "toggle_notes_off": "Notes off",
    "toggle_swap": "Her state on another model",
    "toggle_arm_3": "Her weights, every note off",
    "toggle_arm_1": "The base model, every note off",
    "toggle_arm_swap": "Another model, given her notes",
    "toggle_arm_bare": "The same model, with no notes",
    "toggle_notes_off_source": "her weights from night {night}, checkpoint {checkpoint}",
    "toggle_swap_source": "{model}, development",
    "toggle_missing": "This comparison wasn't generated for this question.",
    # her mind panel
    "mind_kicker": "Her mind",
    "mind_heading": "What she is working on, and what she chose",
    "mind_loading": "Loading her state…",
    "mind_note": (
        "Read from her logs as they stand, on her own clock. Anything only our scorer may read is held "
        "back, on the server, before it reaches this page."
    ),
    "mind_asof": "As of night {night} ({date}).",
    "mind_asof_none": "She hasn't slept yet.",
    "mind_glance_questions": "Open questions: {n}.",
    "mind_glance_targets": "Open threads and plans: {n}.",
    "mind_glance_empty": "Nothing to show yet.",
    "mind_state_none": "Nothing yet.",
    "mind_state_unavailable": "Can't be read right now.",
    "mind_none_yet": "not yet",
    "mind_no_target": "nothing",
    "mind_withheld": "Held back: only the scorer may read this.",
    "mind_questions": "Her questions",
    "mind_questions_line": "Questions she is working on: {n}.",
    "mind_credence_note": (
        "Each bar is how much weight she gives one answer. It's a heuristic of hers, not a measured "
        "probability."
    ),
    "question_kind_contradiction": "contradiction",
    "question_kind_gap": "gap",
    "question_kind_opportunity": "opportunity",
    "question_kind_anomaly": "anomaly",
    "question_status_open": "noticed",
    "question_status_active": "working on it",
    "question_status_parked": "parked",
    "mind_prediction": "She predicted",
    "mind_prediction_given": "chance it holds, by answer: {given} · by {horizon}",
    "prediction_outcome_pending": "not resolved yet",
    "prediction_outcome_true": "came true",
    "prediction_outcome_false": "didn't come true",
    "prediction_outcome_undetermined": "can't tell yet",
    "prediction_outcome_censored": "ran out of time",
    "mind_targets": "His open threads and plans",
    "mind_targets_line": "{n} open.",
    "mind_confidence_note": "Her own heuristic confidence that she understood each one.",
    "mind_target_due": "plan, due {date}",
    "target_kind_thread": "thread",
    "target_kind_commitment": "plan",
    "mind_wants": "Her wants",
    "mind_wants_line": "Her wants on night {night}: {n}.",
    "want_status_won": "chosen",
    "want_status_held_back": "held back",
    "want_status_rejected": "turned down",
    "want_kind_follow_up": "follow up",
    "want_kind_understand": "understand",
    "want_kind_share": "share",
    "want_kind_know": "keep asking",
    "want_kind_settle": "settle it",
    "action_prepare_commitment": "prepare a plan",
    "action_revisit_thread": "revisit a thread",
    "action_inquire": "think a question through",
    "action_hold_note": "write a note",
    "reason_unknown_priority_class": "of a kind she doesn't rank",
    "reason_action_not_offered": "not on offer that night",
    "reason_candidate_limit": "past the night's limit",
    "reason_below_threshold": "not strong enough",
    "reason_budget": "no budget left",
    "reason_lost_to_target": "lost to the chosen one",
    "reason_note_declined": "the note was declined",
    "mind_choices": "What she chose, and what it beat",
    "mind_chose": "On night {night} she chose to {action}: {target}.",
    "mind_chose_nothing": "On night {night} she chose nothing.",
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
    "mind_expectations": "Her expectations",
    "mind_expectations_line": "{n} made, {resolved} resolved.",
    "mind_expectations_note": (
        "How likely she thought he'd come back to a topic before a horizon. Resolved only from his "
        "side, never from her own."
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
    "mind_gates_passed_skipping": "Night {night} passed every gate that ran; {n} skipped.",
    "mind_gates_failed": "Night {night} failed {n} of its gates, so its weights weren't promoted.",
    "mind_gates_none": "Night {night} had nothing to train, so no gate ran.",
    "mind_gates_skipped": "No gate ran.",
    "mind_gate_pass": "passed",
    "mind_gate_fail": "failed",
    "mind_gate_skip": "skipped",
    "mind_gate_off": "off for the hackathon",
    "mind_gate_numbers": "{value} against {threshold}",
    "gate_general": "general skills",
    "gate_collapse": "no collapse",
    "gate_safety": "safety",
    # the results chart
    "chart_heading": "Results",
    "chart_line": "Development results, scored nights: {nights}. Final results come after the freeze.",
    "chart_none": "No scored nights yet.",
    "chart_no_verdict": "{n} no verdict",
    "chart_dev_caption": "{name}'s development nights. Labeled development: they informed decisions, and are never the final test.",
    "chart_panel_title": "Score by night: {dimension}",
    "chart_series_desc": "{arm}: {first} on the first scored night, {last} on the last. Scored nights: {nights}.",
    "chart_numbers": "Show the numbers",
    "chart_probes_subset": "daily probes",
    "chart_table_caption": "Every arm's score by night, development",
    "chart_night": "Night",
    "chart_final_heading": "Final test",
    "chart_final_not_yet": "Not yet. The final test runs once, on the held-out lives, after the freeze.",
    "chart_final_ready": "Held-out lives scored: {lives}.",
    "dimension_knowledge": "knowledge",
    "dimension_interaction": "interaction",
    "chart_arm_1": "Base model, no notes",
    "chart_arm_2C": "Base model with her notes",
    "chart_arm_2F": "Base model with her notes and all his evidence",
    "chart_arm_3": "Her weights, notes off",
    "chart_arm_3x": "Another life's weights, notes off",
    "chart_arm_4": "Her weights with her notes",
    "chart_arm_short_1": "base",
    "chart_arm_short_2C": "notes",
    "chart_arm_short_2F": "all",
    "chart_arm_short_3": "hers",
    "chart_arm_short_3x": "other",
    "chart_arm_short_4": "hers+",
    # the Task 16 result and the ledger
    "task16_heading": "Does her state study what matters?",
    "task16_not_yet": "Not yet. This is measured on the held-out lives after the freeze.",
    "ledger_heading": "The pre-registration ledger",
    "ledger_line_ok": "Entries: {n}. The chain verifies here, and {sync}",
    "ledger_line_broken": "Entries: {n}. The chain does NOT verify here, and {sync}",
    "ledger_in_sync": "its head matches the published one.",
    "ledger_out_of_sync": "its head does not match the published one.",
    "ledger_never_pushed": "it hasn't been published yet.",
    "ledger_head": "Head: {hash}",
    "ledger_note": "Every file, prompt and threshold registered before the results existed, newest first, by hash.",
    # ── the night as one loop (demo/night_view.py, web/night.js) ──
    "story_kicker": "One system",
    "story_heading": "Last night, as one loop",
    "story_lede": (
        "She lives with you by day, dreams at night about what she's unsure of, rehearses exactly that, "
        "and wakes up a better mind of you."
    ),  # only when that is what ran (``night_view.rehearsed_the_dream``); else the plain line
    "story_lede_plain": "She lived with you by day and slept on it last night.",
    "story_night": "Night {night} · {date} · {status}",
    "story_loading": "Loading last night…",
    "story_none": "She hasn't slept yet.",
    "story_unavailable": "Last night's loop can't be read right now.",
    "story_sleep_failed": "Her sleep step failed, so the night stopped there.",
    "story_nap": "She napped: learning tomorrow.",
    "story_brief_kept": "Her brief couldn't be rewritten tonight, so the one from {night} was kept.",
    "story_brief_kept_empty": "Her first brief couldn't be written tonight, so she went on without one.",
    "story_stage_dream": "Dream",
    "story_stage_rehearse": "Rehearse",
    "story_stage_learn": "Learn",
    "story_stage_check": "Check",
    "story_stage_wake": "Wake",
    "story_line_dream": "She dreamt about {n} things she's unsure of.",
    "story_line_rehearse": "She rehearsed {n} scenes.",
    "story_line_reused": "She reused {n} scenes she rehearsed for {night} in an earlier run.",
    "story_line_learn": "She learned from {n} rows of the day.",
    "story_line_check": "{n} of {of} gates passed: her new weights went live.",
    "story_line_check_held": "{n} of {of} gates passed: yesterday's weights still serve.",
    "story_line_wake": "The morning scored {n} replies.",
    "story_line_wake_uncounted": "The morning scored her new weights.",
    "story_not_run": "didn't run",
    "story_reused": "reused",
    "story_failed": "stopped",
    "story_failed_line": "The {step} step failed, so the night stopped here.",
    "story_off": "off",
    "story_evidence": "See the evidence",
    "story_cites": "The events it rests on",
    "story_number_questions": "questions opened",
    "story_number_ignitions": "ignitions",
    "story_number_threads": "threads picked",
    "story_number_boundaries": "boundary states",
    "story_number_flagged": "flagged by the critic",
    "story_number_corrected": "corrected by the teacher",
    "story_number_dreams": "Glimmer's dreams kept",
    "story_number_dream_candidates": "dreams written",
    "story_number_dream_keep_rate": "keep rate",
    "story_number_seeded": "picks practice was seeded on",
    "story_number_seeded_threads": "threads they came from",
    "story_number_surprise": "surprise weights applied",
    "story_number_surprise_rows": "rows weighted",
    "story_number_channel_loss": "thought-channel loss",
    "story_number_stamp": "projector",
    "story_number_seeding": "seeding",
    "story_number_seed_rows": "seeding rows",
    "story_number_seeding_kl_before": "seeding KL before",
    "story_number_seeding_kl_after": "seeding KL after",
    "story_number_imitation": "imitation",
    "story_number_imitation_rows": "imitation rows",
    "story_number_imitation_reward": "imitation reward",
    "story_number_fold": "folded to rank",
    "story_number_protect": "protected slot",
    "story_number_protect_minutes": "protect minutes",
    "story_number_core_adaptation": "core adaptation loss",
    "story_number_no_verdict": "no verdict",
    "story_number_self_model": "self-model error",
    "story_number_self_model_predicted": "error she predicted",
    "story_number_self_model_actual": "error she made",
    # ── dailies: the morning sheet ──
    "unknown_night": "That night isn't one she has slept yet.",
    "dailies_kicker": "Dailies",
    "dailies_heading": "The morning after each night",
    "dailies_note": (
        "In film, dailies are what the crew reviews each morning. Here: what she, his model, took from his day, "
        "one line she'd now say differently, and what the night cost. Read from her logs, on her own clock."
    ),
    "dailies_loading": "Loading this morning's sheet…",
    "dailies_none": "She hasn't slept yet, so there are no dailies.",
    "dailies_unavailable": "This morning's sheet can't be read right now.",
    "dailies_pick": "Morning after",
    "dailies_option": "Night {night} · {date}",
    "dailies_slate_night": "Night",
    "dailies_slate_morning": "Morning of",
    "dailies_slate_status": "The night",
    "dailies_slate_sleep": "Sleep",
    "dailies_gate_skipped": "gate skipped, off for the hackathon",
    "dailies_no_verdict": "the judge gave no verdict on {n}",
    "dailies_withheld": "Held back by a check before it reached this page.",
    "dailies_learned": "What she took from his day",
    "dailies_learned_none": "Nothing new reached her that night, and she chose nothing.",
    "dailies_opened": "new",
    "dailies_moved": "came back",
    "dailies_more": "and {n} more",
    "dailies_question": "A new question she's asking",
    "dailies_line": "One line she'd now say differently",
    "dailies_asked": "The same question, asked the evening before and the morning after:",
    "dailies_before": "Before · night {night}",
    "dailies_before_base": "Before · the base model",
    "dailies_before_unknown": "Before · an earlier checkpoint",
    "dailies_line_missing": "Nothing here: this reply ended sooner.",
    "dailies_after": "After · night {night}",
    "dailies_line_source": "Pre-generated · her weights alone, every note off · generated {date}",
    "dailies_line_same": "She'd still say it the same way.",
    "dailies_line_no_new_checkpoint": (
        "No new weights were promoted that night, so there is no before and after. Nothing was generated."
    ),
    "dailies_line_not_generated": "This night's before and after hasn't been generated yet.",
    "dailies_cost": "What the night cost",
    "dailies_cost_note": "At list prices. Dollars from {source}; GPU-minutes only where a step timed itself.",
    "cost_source_summary": "the night's own summary, the day and the morning from the budget ledger",
    "cost_source_ledger": "the budget ledger's settlements for this night",
    "cost_part_day": "His simulated day",
    "cost_part_practice": "Practice",
    "cost_part_training": "Training",
    "cost_part_morning": "The morning run",
    "cost_part_rest": "Everything else that night",
    "cost_col_part": "Part",
    "cost_col_usd": "Dollars",
    "cost_col_gpu": "GPU-min",
    "cost_total": "Total",
    "cost_unrecorded": "not recorded",
    "cost_month": "A month at this pace so far: {month} of the {ceiling} ceiling.",
    "cost_month_worst": "If every night trained: {worst}.",
    "cost_month_label": "Projection, development",
    "cost_month_none": "Not enough nights to project a month yet.",
    "cost_ceiling": "{ceiling} ceiling",
    "footer": (
        "Votes keep only your choice, which side was which and the time. Your questions are never "
        "stored. Built with Muse Spark Open."
    ),
    # ── the return moment (demo/return_view.py, web/app.js) ──
    # Her message itself is never in COPY: it is the record's own words, from the note or finding.
    "return_kicker": "While you were away",
    "return_heading": "She kept working, and left you this",
    "return_loading": "Reading what she left…",
    "return_days_one": "1 quiet day",
    "return_days_many": "{n} quiet days",
    "return_kind_held_note": "A note she wrote at the end of a night.",
    "return_kind_share": "A finding she raised from her own questions.",
    "return_record_note": "note from the night of {night}",
    "return_record_share": "finding {id} · logged at {event}",
    "return_withheld": "Her message is held back by a check before it reaches this page.",
    "return_none": "Nothing was waiting this time.",
    "return_unavailable": "Her message can't be read right now.",
    "return_trace": "Trace how she got here",
    "return_sources": "Where it came from",
    "return_source": "source",
    # ── how she got here (demo/timeline.py, web/timeline.js) ──
    "timeline_heading": "How I got here",
    "timeline_note": (
        "Every node is one of her logged records, in order. The plain line says what happened; open "
        "it for her exact words and the event's id."
    ),
    "timeline_loading": "Reading her quiet days…",
    "timeline_none": "Nothing in these quiet days.",
    "timeline_unavailable": "Her timeline can't be read right now.",
    "timeline_days_one": "1 quiet day",
    "timeline_days_many": "{n} quiet days",
    "timeline_counts": (
        "{questions} questions · {credences} credences moved · {predictions} predictions · "
        "{resolutions} resolved · {searches} searches · {decisions} decisions"
    ),
    "timeline_day": "Day {date}",
    "timeline_kind_question": "question",
    "timeline_kind_credence": "credence",
    "timeline_kind_prediction": "prediction",
    "timeline_kind_resolution": "resolved",
    "timeline_kind_search": "search",
    "timeline_kind_decision": "decision",
    "timeline_kind_practice": "practice",
    "timeline_plain_question": "She opened a {kind} question, {origin}.",
    "timeline_plain_credence_one": "One answer moved.",
    "timeline_plain_credence_many": "{n} answers moved.",
    "timeline_plain_prediction": "She committed to a prediction.",
    "timeline_plain_resolution": "A prediction resolved: {verdict}.",
    "timeline_plain_search": "She searched the web.",
    "timeline_plain_practice": "She practised {target}, because she was working on it.",
    "timeline_plain_decision": "That night she chose to {action}.",
    "timeline_plain_decision_none": "That night she chose nothing.",
    "timeline_origin_her": "on her own",
    "timeline_origin_person": "from his evidence",
    "timeline_verdict_right": "right",
    "timeline_verdict_wrong": "wrong",
    "timeline_verdict_open": "no verdict yet",
    "timeline_detail_question": "The question",
    "timeline_detail_origin": "Raised",
    "timeline_detail_kind": "Kind",
    "timeline_detail_status": "Status",
    "timeline_detail_subject": "In her words",
    "timeline_detail_cites": "What raised it",
    "timeline_detail_credence": "What moved",
    "timeline_detail_statement": "Answer",
    "timeline_detail_before": "before",
    "timeline_detail_after": "after",
    "timeline_detail_moved_by": "Moved by",
    "timeline_detail_source": "From",
    "timeline_detail_evidence": "Evidence",
    "timeline_detail_prediction": "The claim",
    "timeline_detail_horizon": "By",
    "timeline_detail_given": "Chance it holds, by answer",
    "timeline_detail_outcome": "Outcome",
    "timeline_detail_verdict": "Verdict",    "timeline_detail_seen": "What the resolver saw",
    "timeline_detail_search": "The query",
    "timeline_detail_answer": "What came back",
    "timeline_detail_scenes": "Focus scenes",
    "timeline_detail_kept": "Kept her own reply",
    "timeline_detail_results": "Results",
    "timeline_detail_decision": "The choice",
    "timeline_detail_action": "Action",
    "timeline_detail_target": "Thread or plan",
    "timeline_detail_confidence": "Confidence",
    "timeline_detail_outcome_decision": "It {outcome}",
    "timeline_detail_record": "Logged record",
    "timeline_detail_night": "Night",
    "timeline_detail_at": "At",
    "timeline_event": "Event {id}",
    "timeline_no_confidence": "not recorded",
    "timeline_no_answer": "no answer came back",
    "timeline_no_results": "no results",
    "timeline_no_action": "nothing",
    "timeline_catch_all": "catch-all",
})


def scene(scene_id: str) -> Scene | None:
    return next((each for each in SCENES if each.id == scene_id), None)
