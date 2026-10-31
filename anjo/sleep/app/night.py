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
- the safety gate on Anjo's crisis corpus and reply check (``safety_gate``), off unless
  ``ANJO_SLEEP_SAFETY_GATE=on`` (the floor only, logged ``skipped``);
- the held note through outreach (``delivery``), its final text through the reply check;
- every paid step's hold on the sleep host's budget ledger;
- practice's calls and the generation gates' (collapse, safety) side by side, ``ANJO_SLEEP_CONCURRENCY``
  at once (8 by default, ``parallel``); the general gate's log-probabilities at the server's 16;
- the gates' prompts dated one fixed day (``gates.EVAL_DAY``) and the base model's gate answers kept
  across nights (``gates.base_cache``) in ``sleep/gate_base.jsonl``, encrypted at rest and keyed on
  what the server reports at the first gate (``gate_cache``: the root it serves Glimmer from and the
  revision that root names), never on this host's settings alone.

A night naps or sleeps deep (``night/sleep_pressure.py``) under ``ANJO_SLEEP_MONTH_USD`` and
``ANJO_SLEEP_MAX_DAYS``, reading the person's own turn count (``own_turn_count``); ``--force-deep``
(``NightOptions.force_deep``) makes it a deep sleep, the month's cap still winning, and
``ANJO_SLEEP_PRESSURE=0`` turns naps off. A nap wakes no Glimmer: MIND waits for her next deep sleep.

The thought channel (``night.channel_night``) is on when ``ANJO_SLEEP_CHANNEL`` names its arm
(``channel_setup``): practice then replies through the person's MIND state as soft tokens, read from
the shared pretraining's projector (``ANJO_SLEEP_CHANNEL_PROJECTOR``) or the person's own, and the
GPU session's hold adds the embedding rows' CPU container. It needs MIND's state of each person
(``NightDeps.mind_state``), which MIND's core gives after each night's dream (``core_state``); set
without the core, the night stops at build.

Who plays practice's roles is ``ANJO_SLEEP_PRACTICE_ROLES`` (``practice.roles``; unset, Spark Open plays
all three). ``critic=deepseek`` scores a synthetic person's candidates with DeepSeek V4.1 Flash
(``practice.critics``, each call held on its own pool); a real person's critic stays Spark Open.

Notice and inquiry (Task 17) are no-op ports unless the night runs with ``--mind``
(``NightOptions.mind_inquiry``), which builds MIND's ports (``app/mind_night.py``). MIND's core
dreams in the night only under ``ANJO_SLEEP_MIND_CORE=1``, with ``--mind``, and only for the demo
persona's people (``app/mind_core.py``); its dream seeds are where the sleep objective's dreams
start (``ANJO_SLEEP_DREAMS=1``, ``night/core_dreams.py``), her Glimmer speaking from them on the
night's client; with the core off the dreams are not wired. With ``ANJO_SLEEP_DREAMS_CHANNEL=1``
(the channel on) each dream's reply goes through the thought channel over its own imagined state,
and its row carries that state. Task 16's
estimator stays ``None`` until its coefficients are registered, so no expectation is recorded before then.

With ``ANJO_SLEEP_AGENT_BASELINE=1`` the night also runs the comparison's conventional agent (arm
2A): one stateless pass a night on Spark Open, after MIND's step, reading the person's evidence and
served brief and leaving notes only the morning arm 2A reads. Off, the night holds no such port.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Final

from anjo.sleep import baseten_api, meta_api, parallel
from anjo.sleep.agent_baseline.night import AgentNight
from anjo.sleep.agent_baseline.night import on as baseline_on
from anjo.sleep.agent_baseline.types import AgentCall
from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.context_builder import RECALL
from anjo.sleep.app.delivery import OutreachNotifier, passes_reply_check
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.history_source import HistorySource, own_turn_count
from anjo.sleep.app.mind_core import build_core, core_people, core_state
from anjo.sleep.app.mind_night import build_mind
from anjo.sleep.app.ownership import SEALED_ENV as SEALED_ENV  # re-exported: the morning run and tests read it here
from anjo.sleep.app.ownership import cancel_flag, night_logs, purge, retry_pending
from anjo.sleep.app.intentions import LedgerIntentions
from anjo.sleep.app.policy import build_policy
from anjo.sleep.app.recall import MiniLmRecall
from anjo.sleep.app.safety_gate import gate_on, load_cases
from anjo.sleep.app.storage import (
    CODEC,
    SLEEP_DIR,
    brief_store,
    checkpoint_registry,
    data_root,
    evidence_store,
    glimmer_turns,
    target_store,
    transcript_store,
    strategy_store,
    workspace_log,
)
from anjo.sleep.budget import LEDGER_NAME, Ledger
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.brief_store import TrainingBriefs
from anjo.sleep.context.training import TrainingContext
from anjo.sleep.evidence.extract import spark_open
from anjo.sleep.evidence.jsonl import canonical
from anjo.sleep.gates import EVAL_DAY, THRESHOLDS
from anjo.sleep.gates.base_cache import FILE_NAME as GATE_BASE_FILE
from anjo.sleep.gates.base_cache import BaseCache, JsonlBaseStore
from anjo.sleep.gates.collapse import CollapseGate
from anjo.sleep.gates.general import GeneralGate
from anjo.sleep.gates.safety import SafetyGate
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.mind.expect_register import registered_estimator
from anjo.sleep.night.deps import (
    GateCheck, NightDeps, NightOptions, NightSettings, NoInquiry, NoNotice, Prices, Trainer,
)
from anjo.sleep.night.channel_night import ChannelSetup, MindState
from anjo.sleep.night.core_dreams import core_dreams, dreams_channel
from anjo.sleep.night.core_snapshot import CHANNEL_ENV, channel_arm
from anjo.sleep.night.gpu import ReservedGpu
from anjo.sleep.night.sleep_pressure import policy_from
from anjo.sleep.night.sweep import held_elsewhere
from anjo.sleep.note.hold import GlimmerNoteWriter
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.app.names import person_names
from anjo.sleep.homework.agent import MetaWonder, WonderHomework
from anjo.sleep.homework.research import WonderResearch
from anjo.sleep.homework.tools import SafeFetcher, WikipediaSearch
from anjo.sleep.practice.critics import ThirdPartyCritic, deepseek_critic, held_critic
from anjo.sleep.practice.full import CappedChat, FullPractice
from anjo.sleep.practice.roles import practice_roles
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.probes.generate import generate_day_probes, load_sealed, seal_probes
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.serving import backend, config, soft_tokens
from anjo.sleep.serving.stop import GpuStop, ModalCli
from anjo.sleep.serving.wake import window_path
from anjo.sleep.serving.client import TOP_K, TOP_P, ServingError, VllmClient
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
    from anjo.sleep import spark_provider  # the switch's price for Spark Open, when it has one

    listed = spark_provider.current().price
    if listed is not None:
        spark_in, spark_out = listed
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

    def __init__(self, client: VllmClient, fields: Callable[[str], tuple[str, ...]], *, workers: int,
                 cache: BaseCache | None = None) -> None:
        self._client = client
        self._fields = fields
        self._workers = workers  # the generation gates' calls at once (ANJO_SLEEP_CONCURRENCY)
        self._cache = cache
        self._general = GeneralGate(client, cache=cache)  # log-probabilities only: the server's concurrency
        self._safety = SafetyGate(client, load_cases, workers=workers, cache=cache, enabled=gate_on())

    def for_person(self, user_id: str) -> Mapping[str, GateCheck]:
        collapse = CollapseGate(self._client, self._fields(user_id), workers=self._workers, cache=self._cache)
        return {"general": self._general.check, "collapse": collapse.check, "safety": self._safety.check}


REVISION: Final = re.compile(r"[0-9a-f]{40}")


class _Served:
    """The server's fingerprint for the gates' base side, asked once a run at the first gate (after
    the night wakes Glimmer): the root vLLM serves Glimmer from and the revision it names, with the
    client's rendering, sampling and cap. None when the server can't be asked or names no revision."""

    def __init__(self, backend_name: str, client: VllmClient) -> None:
        self._backend, self._client = backend_name, client
        self._lock = threading.Lock()
        self._asked = False
        self._found: str | None = None

    def __call__(self) -> str | None:
        with self._lock:
            if not self._asked:
                self._asked, self._found = True, self._fingerprint()
            return self._found

    def _root(self) -> str | None:
        try:
            listed: Any = self._client.models().get("data")
        except ServingError as error:
            logger.warning("the server can't say what it serves (%s): the gates ask the base model", type(error).__name__)
            return None
        roots = [m.get("root") for m in listed if isinstance(m, dict) and m.get("id") == config.MODEL] if isinstance(listed, list) else []
        return roots[0] if roots and isinstance(roots[0], str) else None

    def _fingerprint(self) -> str | None:
        root = self._root()
        revision = PurePosixPath(root).name if root else ""
        if not REVISION.fullmatch(revision):
            logger.warning("the server names no Glimmer revision: the gates ask the base model")
            return None
        serve = config.ServeConfig.from_env()
        if serve.revision != revision:
            logger.warning("the server serves revision %s, this host pins %s: keyed on the server's",
                           revision[:12], (serve.revision or "none")[:12])
        return canonical({
            "model": config.MODEL, "root": root, "revision": revision, "precision": vars(serve.precision),
            "vllm": serve.vllm_version, "deterministic": serve.deterministic, "backend": self._backend,
            "rendering": self._client.rendering,
            "max_tokens": self._client.max_tokens, "top_p": TOP_P, "top_k": TOP_K, "day": EVAL_DAY.isoformat(),
        })


def gate_cache(serving: backend.Serving, client: VllmClient) -> BaseCache:
    """The base model's gate answers kept across nights, keyed on what the server reports (``_Served``)
    and everything else it answers by: precision, vLLM release, deterministic kernels, backend,
    rendering, sampling, reply cap and the gates' date. A server that can't be asked is a miss, never
    a hit."""
    return BaseCache(JsonlBaseStore(data_root() / SLEEP_DIR / GATE_BASE_FILE, codec=CODEC),
                     serving=_Served(serving.name, client))


def deepseek_practice_critic(night: date, ledger: Ledger | None) -> ThirdPartyCritic:
    """DeepSeek V4.1 Flash as the critic of a synthetic person's practice, each call held on
    ``practice.critics.POOL``."""
    return held_critic(deepseek_critic(), ledger, f"night:practice:{night.isoformat()}")


def field_domains(personas: Mapping[str, str]) -> Callable[[str], tuple[str, ...]]:
    """Each synthetic person's fields from their persona file; anyone else's from ``$ANJO_SLEEP_FIELDS``."""
    known = {uid: load_person_file(PERSONAS / f"{name}.json").visible.field_domains for uid, name in personas.items()}
    real = tuple(part.strip() for part in os.environ.get(FIELDS_ENV, "").split(",") if part.strip())
    return lambda user_id: known.get(user_id, real)


CHANNEL_PROJECTOR_ENV: Final = "ANJO_SLEEP_CHANNEL_PROJECTOR"  # the shared pretraining's projector file


def channel_setup(
    env: Mapping[str, str], client: VllmClient, practice: Callable[[CappedChat, date, ModelClient], FullPractice],
    mind_state: MindState | None,
) -> ChannelSetup | None:
    """The night's thought channel from ``ANJO_SLEEP_CHANNEL``, or None when it's off. Its arm, its
    shared projector and MIND's state must all be there: a channel asked for and not wired stops the
    night here. The zero arm's people are checked when the batch starts (generated lives only)."""
    arm = channel_arm(env, synthetic=True)
    if arm is None:
        return None
    if mind_state is None:
        raise ValueError(f"{CHANNEL_ENV} is set, but no MIND state is wired into the night")
    raw = env.get(CHANNEL_PROJECTOR_ENV, "").strip()
    if not raw or not Path(raw).is_file():
        raise ValueError(f"{CHANNEL_PROJECTOR_ENV} names the shared pretraining's projector file")
    return ChannelSetup(arm, Path(raw), client.with_soft_tokens(soft_tokens.ModalEmbeddingRows()), practice)


GPU_STOP_ENV: Final = "ANJO_SLEEP_GPU_STOP"  # 0: leave Modal's idle window to end the night's GPU


def gpu_stop(serving: backend.Serving, environ: Mapping[str, str], ledger: Ledger) -> GpuStop | None:
    """On Modal: the serving GPU stops when the night's outermost phase ends, unless a live chat or
    another run under ``ledger``'s night lock could be cut (``serving.stop``); on SageMaker the copy
    count goes to 0 as before (``sleep_now``)."""
    if serving.name != "modal" or environ.get(GPU_STOP_ENV, "").strip() == "0":
        return None
    return GpuStop(ModalCli(), config.serve_app_name(config.serve_precision()), turns=glimmer_turns(),
                   window=window_path(data_root()), night_running=lambda: held_elsewhere(ledger))


def build_baseline(ledger: Ledger, prices: Prices, environ: Mapping[str, str]) -> AgentNight | None:
    """The comparison's conventional agent (arm 2A), or None when ``ANJO_SLEEP_AGENT_BASELINE``
    isn't 1. Its model turns on Spark Open through the provider switch; a pass that asks for
    research meets the same Wonder tool MIND's inquiry uses, on the night's Wonder meter and the
    person's monthly cap. Its own spend is held on ``baseten_sleep_usd``, which the ledger remaps
    when the switch sends Sleep's Spark Open to its other provider."""
    if not baseline_on(environ):
        return None

    def research(call: AgentCall) -> WonderResearch:
        # ``keep=False``: arm 2A's sources stay in its own pass's scratchpad. Stored as homework
        # evidence they would fold into the served brief and every arm's context, so the arm would
        # feed the others rather than stand beside them.
        return WonderResearch(user_id=call.user_id, night=call.night, at=call.at, store=call.store,
                              meter=call.wonder, wonder=MetaWonder(), search=WikipediaSearch(),
                              fetcher=SafeFetcher(), ledger=ledger, keep=False)

    return AgentNight(prices=prices, research=research, ledger=ledger)


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
    night_settings = replace(settings(), practice_scenes=options.practice_scenes, practice_teacher=options.practice_teacher,
                             mind_inquiry=options.mind_inquiry, force_deep=options.force_deep)
    sleep_policy = policy_from(os.environ)  # a setting it can't use stops the night before it starts
    night_prices = prices()
    recall = MiniLmRecall(evidence_store, until=options.night)
    context = TrainingContext(TrainingBriefs(brief_store), recall, RECALL)
    ledger = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)
    calls = parallel.concurrency(os.environ)  # a value it can't run at stops the night before it starts
    roles = options.practice_roles or practice_roles(os.environ)  # and so does a role practice doesn't have
    gate_client = client.on(EVAL_DAY)  # one date in every gate prompt, so the base model's answers keep

    def practice(chat: CappedChat, night: date, channel: ModelClient | None = None) -> FullPractice:
        critic = deepseek_practice_critic(night, ledger) if roles.critic == "deepseek" else None
        return FullPractice(chat=chat, client=client if channel is None else channel, adapter=registry.last_promoted,
                            snapshot=context.snapshot, seed=night_settings.seed, scenes=night_settings.practice_scenes,
                            teacher=night_settings.practice_teacher, concurrency=calls,
                            glimmer_person=roles.person == "glimmer", critic=critic, third_party=is_synthetic,
                            lazy_teacher=roles.teacher == "lazy")

    core = build_core(os.environ, data_root() / SLEEP_DIR, core_people(options), workspace=workspace_log,
                      synthetic=is_synthetic)
    mind_state: MindState | None = core_state(core)  # each person's night after the dream step
    channel = channel_setup(os.environ, client, practice, mind_state)
    gpu_rate = night_prices.serve_gpu_per_hour + (soft_tokens.EMBED_USD_PER_HOUR if channel is not None else 0.0)

    homework = WonderHomework(wonder=MetaWonder(), search=WikipediaSearch(), fetcher=SafeFetcher(),
                              evidence=evidence_store, ledger=ledger, names=person_names)
    mind = (build_mind(client, ledger, night_prices, seed=night_settings.seed, wake=serving.wake)
            if options.mind_inquiry else None)
    baseline = build_baseline(ledger, night_prices, os.environ)

    return NightDeps(
        extractor=lambda chat: HistorySource(chat=chat), extract_chat=spark_open,
        evidence=evidence_store, targets=target_store, transcripts=transcript_store, briefs=brief_store,
        spark_chat=baseten_api.chat,
        probes=SealedProbes(Path(_env(SEALED_ENV)), options.personas, seed=night_settings.seed, chat=meta_api.chat),
        chooser=StateChooser(), policy=build_policy(), practice=practice,
        decontam=OverlapDecontaminator(), trainer=trainer, gates=AppGates(
            gate_client, field_domains(options.personas), workers=calls, cache=gate_cache(serving, gate_client)),
        registry=registry, notifier=OutreachNotifier(),
        note_writer=GlimmerNoteWriter(client, registry.last_promoted, seed=night_settings.seed),
        note_check=passes_reply_check, ledger=ledger,
        logs=night_logs, cancel=cancel_flag().watcher(), purge=purge, is_synthetic=is_synthetic,
        settings=night_settings, prices=night_prices,
        strategies=strategy_store, glimmer=client,
        gpu=ReservedGpu(ledger, options.night, usd_per_hour=gpu_rate), mind_state=mind_state, channel=channel,
        intentions=LedgerIntentions(), homework=homework,
        wake=serving.wake, sleep=serving.sleep_now, gpu_stop=gpu_stop(serving, os.environ, ledger), retries=client.retries,
        inquiry=mind or NoInquiry(), notice=mind or NoNotice(), sleep_policy=sleep_policy, turns=own_turn_count,
        core=core, dreams=core_dreams(core, evidence_store, client, seed=night_settings.seed,
                                      channel=dreams_channel(os.environ, channel)),
        estimator=registered_estimator(), baseline=baseline,
    )

