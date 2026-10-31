"""The morning probe run's wiring on the sleep host (Task 12a): ``build`` gives ``Morning`` over Anjo's
stores for one synthetic person.

    python -m anjo.sleep.sim.accelerated --morning anjo.sleep.app.morning:build ...

- arms: Glimmer's vLLM server (``VllmClient``, which answers replies and log-probabilities; SageMaker or Modal),
  context C from Anjo's own builder over the served briefs, 2F over the person's evidence store, 2A and 4M
  with the agent's notes and MIND's records (``full_arms`` adds them to a run with ``ANJO_SLEEP_ARMS_EXTRA=1``);
- adapters: the checkpoint registry (the last promoted checkpoint serves), donors from
  ``$ANJO_SLEEP_DONORS`` (``user=donor,user=donor``: the registered donor account of each life);
- probes: the person's sealed file, ``$ANJO_SLEEP_SEALED_DIR/<persona>.jsonl``, re-keyed to the
  account that plays the persona;
- judge: Muse Spark or DeepSeek, selected by ``ANJO_SLEEP_JUDGE``, with the persona's judge brief;
- the arms' and the judge's calls side by side, ``ANJO_SLEEP_CONCURRENCY`` at once (8 by default); a
  judge call the API answers 429 or 5xx goes again (``model_api.Retries``), and the DeepSeek judge
  waits the longer busy schedule between tries; a DeepSeek reply that never came whole is sent again
  up to ``DEEPSEEK_JUDGE_REPLY_ATTEMPTS`` attempts in all: one such call cost a whole morning on
  2026-10-30 and a 3 h 48 min adapter-strength read on 2026-10-31; the night log counts those with
  the requests Glimmer's client sent again.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep import fireworks_api, meta_api, parallel
from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.context_builder import RECALL, anjo_context_builder
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.night import SEALED_ENV, prices
from anjo.sleep.app.storage import (
    SLEEP_DIR,
    brief_store,
    checkpoint_registry,
    data_root,
    evidence_store,
    workspace_log,
)
from anjo.sleep.budget import LEDGER_NAME, Ledger
from anjo.sleep.model_api import Retries
from anjo.sleep.context.brief_store import ServedBriefs
from anjo.sleep.mind import records
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import RunStart, WorkspaceEvent
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.generate import load_sealed
from anjo.sleep.probes.judge_brief import judge_brief
from anjo.sleep.probes.morning import Morning
from anjo.sleep.probes.score import JudgeTokens, MuseJudge
from anjo.sleep.serving import backend
from anjo.sleep.types import Probe

DONORS_ENV: Final = "ANJO_SLEEP_DONORS"
JUDGE_ENV: Final = "ANJO_SLEEP_JUDGE"
# A judge call whose reply never came whole is sent again, up to this many attempts in all: one
# ReadError among ~240 calls ended a 3 h 48 min adapter-strength read on 2026-10-31.
DEEPSEEK_JUDGE_REPLY_ATTEMPTS: Final = 4


def make_judge(persona: str, tokens: JudgeTokens, resent: _Resent) -> MuseJudge:
    """Select the judge before any calls; Spark Open is never a selectable judge."""
    selected = os.environ.get(JUDGE_ENV, "muse").strip() or "muse"
    if selected == "muse":
        return MuseJudge(partial(meta_api.chat, retries=Retries(counted=resent)), lambda _person: judge_brief(persona),
                         tokens=tokens)
    if selected == "deepseek":
        # A reply that never came whole failed a whole morning on 2026-10-30 and a whole
        # adapter-strength read on 2026-10-31: it is sent again, up to DEEPSEEK_JUDGE_REPLY_ATTEMPTS
        # attempts, and a busy 429/5xx waits the longer schedule. The resends stay inside the one
        # call's hold (probes.morning._HeldDeepseekChat), so one per-call bound still covers them.
        retries = Retries(counted=resent, connect_attempts=3,
                          reply_attempts=DEEPSEEK_JUDGE_REPLY_ATTEMPTS,
                          backoff_s=fireworks_api.SPARK_BUSY_BACKOFF_S)
        return MuseJudge(partial(fireworks_api.chat, retries=retries), lambda _person: judge_brief(persona),
                         tokens=tokens, model=fireworks_api.DEEPSEEK_V4P1_FLASH)
    raise ValueError(f"{JUDGE_ENV} must be muse or deepseek")


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def donors() -> dict[str, str]:
    pairs = (part.split("=", 1) for part in _env(DONORS_ENV).split(",") if part.strip())
    return {user.strip(): donor.strip() for user, donor in pairs}


def _through_night(events: Sequence[WorkspaceEvent], night: date) -> tuple[WorkspaceEvent, ...]:
    """The log through the night's last step. A later night's ``RunStart`` may wipe the workspace
    (M-persist), so folding the live log would serve a backfill of an older night later findings;
    cutting at the first start of a later night keeps the workspace that night left."""
    return events[:next((n for n, event in enumerate(events)
                         if isinstance(event.step.entry, RunStart) and event.step.entry.night > night), len(events))]


def mind_notes(user_id: str, night: date) -> tuple[str, ...]:
    """Arm 4M's records through ``night``: MIND's findings and the questions it noticed, from the
    person's own workspace log as that night left it. An account without one has no records, so 4M
    asks exactly what arm 4 asks; a backfill of an older night never sees a later night's."""
    return records.notes(kernel.fold(_through_night(workspace_log(user_id).events(), night)))


class _Resent:
    """The judge's calls sent again, counted across its threads."""

    def __init__(self) -> None:
        self.count = 0
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            self.count += 1


def build(user_id: str, persona: str, night: date) -> Morning:
    require_sleep_host()
    if not is_synthetic(user_id):
        raise ValueError(f"{persona} runs only on an account registered as synthetic")
    serving = backend.from_env()  # SageMaker, or Modal (ANJO_SLEEP_BACKEND)
    client = serving.client().on(night)  # prompts dated on the night's clock
    sealed = [replace(p, person_id=user_id) for p in load_sealed(Path(_env(SEALED_ENV)) / f"{persona}.jsonl")]

    def probes(uid: str) -> list[Probe]:
        return sealed

    arms = ArmDeps(
        model=client, adapters=checkpoint_registry(), contexts=anjo_context_builder(
            ServedBriefs(brief_store), until=night, settings=RECALL),
        evidence=evidence_store, logprobs=client, donors=donors(), until=night, mind_notes=mind_notes)
    resent = _Resent()
    tokens = JudgeTokens()  # the judge settles at its tokens, priced at ANJO_SLEEP_PRICES' Spark lines
    judge = make_judge(persona, tokens, resent)
    return Morning(arms=arms, sealed=probes, judge=judge, prices=prices(), ledger=Ledger(data_root() / SLEEP_DIR / LEDGER_NAME),
                   wake=serving.wake, sleep=serving.sleep_now, concurrency=parallel.concurrency(os.environ),
                   retries=lambda: client.retries() + resent.count, judge_tokens=tokens,
                   judge_model=meta_api.PINNED_MODELS["muse-spark"] if judge.model == "muse-spark" else judge.model)
