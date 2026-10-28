"""The night's wiring on the sleep host (Task 5): ``build`` gives ``NightDeps`` over Anjo's stores.

``python -m anjo.sleep.night --wiring anjo.sleep.app.night:build`` calls it with the trainer, which
the composition root builds: the app never imports the training code (the import contract keeps
it off the serving path). Everything else is here:

- evidence, targets, briefs and practice transcripts in the person's encrypted ``sleep/`` folder;
- extraction through the history store (``HistorySource``) on Spark Open; briefs, practice scenes,
  the teacher candidate and the critic on Spark Open; full practice's personal Glimmer candidates,
  the gates and the note on Glimmer's vLLM server at serving precision (``VllmClient``; SageMaker,
  or Modal, per ``serving.backend``), its spend held on ``modal_usd`` (``ReservedGpu``);
- homework (Task 7): Muse Wonder on the Meta Model API, searching Wikipedia's public API and
  fetching public pages only (``WonderHomework``), each call held on ``meta_usd``;
- her chooser (``StateChooser``) with the registered policy and the outcome ledger's intentions;
- probes: a synthetic person's were sealed once (``$ANJO_SLEEP_SEALED_DIR/<persona>.jsonl``);
  a real person's are written from the day's evidence by Muse Spark and sealed before training;
- the safety gate on Anjo's crisis corpus and reply check (``safety_gate``);
- the held note through outreach (``delivery``), its final text through the reply check;
- every paid step's hold on the sleep host's budget ledger;
- practice's calls side by side, ``ANJO_SLEEP_CONCURRENCY`` at once (8 by default, ``parallel``).

Notice and inquiry (Task 17) are no-op ports unless the night runs with ``--mind``
(``NightOptions.mind_inquiry``), which builds MIND's ports (``app/mind_night.py``). Task 16's
estimator stays ``None`` until its coefficients are registered, so no expectation is recorded before then.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep import baseten_api, meta_api, parallel
from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.context_builder import RECALL
from anjo.sleep.app.delivery import OutreachNotifier, passes_reply_check
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.history_source import HistorySource
from anjo.sleep.app.mind_night import build_mind
from anjo.sleep.app.ownership import SEALED_ENV as SEALED_ENV  # re-exported: the morning run and tests read it here
from anjo.sleep.app.ownership import cancel_flag, night_logs, purge, retry_pending
from anjo.sleep.app.intentions import LedgerIntentions
from anjo.sleep.app.policy import build_policy
from anjo.sleep.app.recall import MiniLmRecall
from anjo.sleep.app.safety_gate import load_cases
from anjo.sleep.app.storage import (
    SLEEP_DIR,
    brief_store,
    checkpoint_registry,
    data_root,
    evidence_store,
    target_store,
    transcript_store,
)
from anjo.sleep.budget import LEDGER_NAME, Ledger
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.brief_store import TrainingBriefs
from anjo.sleep.context.training import TrainingContext
from anjo.sleep.evidence.extract import spark_open
from anjo.sleep.gates import THRESHOLDS
from anjo.sleep.gates.collapse import CollapseGate
from anjo.sleep.gates.general import GeneralGate
from anjo.sleep.gates.safety import SafetyGate
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.night.deps import (
    GateCheck, NightDeps, NightOptions, NightSettings, NoInquiry, NoNotice, Prices, Trainer,
)
from anjo.sleep.night.gpu import ReservedGpu
from anjo.sleep.note.hold import GlimmerNoteWriter
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.app.names import person_names
from anjo.sleep.homework.agent import MetaWonder, WonderHomework
from anjo.sleep.homework.tools import SafeFetcher, WikipediaSearch
from anjo.sleep.practice.full import FullPractice
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.probes.generate import generate_day_probes, load_sealed, seal_probes
from anjo.sleep.serving import backend
from anjo.sleep.serving.client import VllmClient
from anjo.sleep.types import Probe

PRICES_ENV: Final = "ANJO_SLEEP_PRICES"  # "spark_in,spark_out,train_gpu_hour,serve_gpu_hour", USD list prices
FIELDS_ENV: Final = "ANJO_SLEEP_FIELDS"  # a real person's field domains for the collapse gate, comma-separated
logger = logging.getLogger(__name__)
PERSONAS: Final = Path(__file__).parents[1] / "personas"
PROBE_MODEL: Final = "muse-spark"  # evaluation only (§3)


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def prices() -> Prices:
    parts = _env(PRICES_ENV).split(",")
    if len(parts) != 4:
        raise ValueError(f"{PRICES_ENV} holds four prices")
    spark_in, spark_out, train, serve = (float(part) for part in parts)
    return Prices(spark_in_per_m=spark_in, spark_out_per_m=spark_out, train_gpu_per_hour=train,
                  serve_gpu_per_hour=serve)


def settings() -> NightSettings:
    registered = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    return NightSettings(token_cap=int(registered["night_token_cap"]),
                         min_new_examples=int(registered["min_new_examples"]))


class SealedProbes:
    """``ProbeRegistrar``: a synthetic person's probes, sealed once; a real person's, written from
    the day's evidence and sealed tonight, before any training input exists."""

    def __init__(self, sealed_dir: Path, personas: Mapping[str, str], *, seed: int, chat: ChatFn) -> None:
        self._dir = sealed_dir
        self._personas = personas
        self._seed = seed
        self._chat = chat

    def register(self, user_id: str, night: date, store: EvidenceStore) -> tuple[list[Probe], str]:
        persona = self._personas.get(user_id)
        if persona is not None:
            return load_sealed(self._dir / f"{persona}.jsonl"), "sealed once"
        day = [item for item in store.for_day(night) if not item.held_out]
        probes = generate_day_probes(day, store.held_out(night), self._seed, chat=self._chat, model=PROBE_MODEL)
        digest = seal_probes(probes, self._dir / user_id / night.isoformat())
        return probes, f"sealed tonight ({digest[:12]})"


class AppGates:
    """The three gates on one server. General and safety keep the base side for the whole run;
    collapse is per person, over their fields."""

    def __init__(self, client: VllmClient, fields: Callable[[str], tuple[str, ...]]) -> None:
        self._client = client
        self._fields = fields
        self._general = GeneralGate(client)
        self._safety = SafetyGate(client, load_cases)

    def for_person(self, user_id: str) -> Mapping[str, GateCheck]:
        collapse = CollapseGate(self._client, self._fields(user_id))
        return {"general": self._general.check, "collapse": collapse.check, "safety": self._safety.check}


def field_domains(personas: Mapping[str, str]) -> Callable[[str], tuple[str, ...]]:
    """Each synthetic person's fields from their persona file; anyone else's from ``$ANJO_SLEEP_FIELDS``."""
    known = {uid: load_person_file(PERSONAS / f"{name}.json").visible.field_domains for uid, name in personas.items()}
    real = tuple(part.strip() for part in os.environ.get(FIELDS_ENV, "").split(",") if part.strip())
    return lambda user_id: known.get(user_id, real)


def build(trainer: Trainer, options: NightOptions) -> NightDeps:
    require_sleep_host()
    for user_id, persona in options.personas.items():
        if not is_synthetic(user_id):
            raise ValueError(f"{persona} runs only on an account registered as synthetic")
    if (outstanding := retry_pending()):  # remote deletes an earlier delete or erasure could not finish
        logger.warning("%d remote cleanups are still outstanding", len(outstanding))
    registry = checkpoint_registry()
    serving = backend.from_env()  # SageMaker, or Modal (ANJO_SLEEP_BACKEND)
    client = serving.client().on(options.night)  # prompts dated on the night's clock
    night_settings = replace(settings(), practice_scenes=options.practice_scenes,
                             practice_teacher=options.practice_teacher, mind_inquiry=options.mind_inquiry)
    night_prices = prices()
    recall = MiniLmRecall(evidence_store, until=options.night)
    context = TrainingContext(TrainingBriefs(brief_store), recall, RECALL)
    ledger = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)
    calls = parallel.concurrency(os.environ)  # a value it can't run at stops the night before it starts

    def practice(chat: ChatFn, night: date) -> FullPractice:
        return FullPractice(chat=chat, client=client, adapter=registry.last_promoted, snapshot=context.snapshot,
                            seed=night_settings.seed, scenes=night_settings.practice_scenes,
                            teacher=night_settings.practice_teacher, concurrency=calls)

    homework = WonderHomework(wonder=MetaWonder(), search=WikipediaSearch(), fetcher=SafeFetcher(),
                              evidence=evidence_store, ledger=ledger, names=person_names)
    mind = (build_mind(client, ledger, night_prices, seed=night_settings.seed, wake=serving.wake)
            if options.mind_inquiry else None)

    return NightDeps(
        extractor=lambda chat: HistorySource(chat=chat), extract_chat=spark_open,
        evidence=evidence_store, targets=target_store, transcripts=transcript_store, briefs=brief_store,
        spark_chat=baseten_api.chat,
        probes=SealedProbes(Path(_env(SEALED_ENV)), options.personas, seed=night_settings.seed, chat=meta_api.chat),
        chooser=StateChooser(), policy=build_policy(), practice=practice,
        decontam=OverlapDecontaminator(), trainer=trainer, gates=AppGates(client, field_domains(options.personas)),
        registry=registry, notifier=OutreachNotifier(),
        note_writer=GlimmerNoteWriter(client, registry.last_promoted, seed=night_settings.seed),
        note_check=passes_reply_check, ledger=ledger,
        logs=night_logs, cancel=cancel_flag().watcher(), purge=purge, is_synthetic=is_synthetic,
        settings=night_settings, prices=night_prices,
        gpu=ReservedGpu(ledger, options.night, usd_per_hour=night_prices.serve_gpu_per_hour),
        intentions=LedgerIntentions(), homework=homework,
        wake=serving.wake, sleep=serving.sleep_now, retries=client.retries,
        inquiry=mind or NoInquiry(), notice=mind or NoNotice(),
    )

