"""The morning probe run's wiring on the sleep host (Task 12a): ``build`` gives ``Morning`` over Anjo's
stores for one synthetic person.

    python -m anjo.sleep.sim.accelerated --morning anjo.sleep.app.morning:build ...

- arms: Glimmer's vLLM server (``VllmClient``, which answers replies and log-probabilities; SageMaker or Modal),
  context C from Anjo's own builder over the served briefs, 2F over the person's evidence store;
- adapters: the checkpoint registry (the last promoted checkpoint serves), donors from
  ``$ANJO_SLEEP_DONORS`` (``user=donor,user=donor``: the registered donor account of each life);
- probes: the person's sealed file, ``$ANJO_SLEEP_SEALED_DIR/<persona>.jsonl``, re-keyed to the
  account that plays the persona;
- judge: Muse Spark with the persona's judge brief (``MuseJudge``), which only the judge reads;
- the arms' and the judge's calls side by side, ``ANJO_SLEEP_CONCURRENCY`` at once (8 by default).
"""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep import meta_api, parallel
from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.context_builder import RECALL, anjo_context_builder
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.night import SEALED_ENV, prices
from anjo.sleep.app.storage import SLEEP_DIR, brief_store, checkpoint_registry, data_root, evidence_store
from anjo.sleep.budget import LEDGER_NAME, Ledger
from anjo.sleep.context.brief_store import ServedBriefs
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.generate import load_sealed
from anjo.sleep.probes.judge_brief import judge_brief
from anjo.sleep.probes.morning import Morning
from anjo.sleep.probes.score import MuseJudge
from anjo.sleep.serving import backend
from anjo.sleep.types import Probe

DONORS_ENV: Final = "ANJO_SLEEP_DONORS"


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def donors() -> dict[str, str]:
    pairs = (part.split("=", 1) for part in _env(DONORS_ENV).split(",") if part.strip())
    return {user.strip(): donor.strip() for user, donor in pairs}


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
        evidence=evidence_store, logprobs=client, donors=donors(), until=night)
    return Morning(arms=arms, sealed=probes, judge=MuseJudge(meta_api.chat, lambda _person: judge_brief(persona)),
                   prices=prices(), ledger=Ledger(data_root() / SLEEP_DIR / LEDGER_NAME),
                   wake=serving.wake, sleep=serving.sleep_now, concurrency=parallel.concurrency(os.environ),
                   retries=client.retries)
