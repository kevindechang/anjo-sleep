"""The ports a night runs through (Task 5), and the no-op ones for tasks not built yet.

``NightDeps`` holds every port. The app builds the private ones (``anjo.sleep.app.night``), the
command line adds the trainer, and tests pass fakes. Nothing here imports training, so the app may
import it.

**No-op ports.** Until their tasks land, these do nothing and say so in the night log:

- ``Homework``: Task 7's ``WonderHomework`` (``anjo.sleep.homework.agent``) in the app.
  ``NoHomework`` stays for tests and the public example: its ``available`` is False, so the night
  offers no study action (the chooser records ``action_not_offered``).
- ``MindCore`` (MIND's core: the dream and the night's unit): ``NoCore``; the app builds
  ``anjo.sleep.mind.core.night.CoreNight`` under ``ANJO_SLEEP_MIND_CORE=1``.
- ``Inquiry`` (Task 17's ``inquire``) and ``Notice`` (Task 17's step 2b): ``NoInquiry`` and
  ``NoNotice``. ``anjo.sleep.night.mind.MindNight`` is both, and the night calls it only while
  ``NightSettings.mind_inquiry`` is on; off, the night is Task 7a's as specced.
- ``AgentBaseline`` (the comparison's conventional agent, arm 2A): ``None``, which is off. Switched
  on (``ANJO_SLEEP_AGENT_BASELINE=1``), ``anjo.sleep.agent_baseline`` runs one stateless pass a
  night after MIND's step and leaves its notes where only the morning arm 2A reads them.

``estimator`` is Task 16's registered ``p_needed`` estimator: ``None`` until its coefficients are
registered, and a chosen study then records no expectation.

The chooser is her own (Task 7a's ``StateChooser``); ``RuleChooser`` stays as the labeled fallback.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from anjo.sleep.agent_baseline.types import AgentBaseline
from anjo.sleep.budget import Ledger
from anjo.sleep.checkpoints import CheckpointRegistry
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.brief_store import JsonlBriefStore
from anjo.sleep.mind.expect import Coefficients
from anjo.sleep.mind.points import QuestionState
from anjo.sleep.night.channel_night import ChannelSetup, MindReading, MindState
from anjo.sleep.night.dreams import DreamSource, NoDreams, Terms
from anjo.sleep.night.meter import ExtractChat, SpendMeter, WonderMeter
from anjo.sleep.night.sleep_pressure import SleepPolicy, SleepState, sleep_pressure
from anjo.sleep.note.hold import Fact, NoteWriter
from anjo.sleep.ports import (
    Chooser,
    Decontaminator,
    EvidenceSource,
    EvidenceStore,
    IntentionSource,
    Notifier,
    TargetStore,
)
from anjo.sleep.practice.full import CappedChat
from anjo.sleep.practice.roles import Roles
from anjo.sleep.practice.run import PracticeRun
from anjo.sleep.practice.transcripts import TranscriptStore
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.serving.stop import GpuStop
from anjo.sleep.types import (
    CheckpointRef,
    Commitment,
    Evidence,
    GateResult,
    Policy,
    Probe,
    Result,
    Target,
)

if TYPE_CHECKING:  # both are torch-free, but the night needs only their types
    from anjo.sleep.mind.core.bridge import CoreView
    from anjo.sleep.mind.core.state import CoreState

STUDY_ACTIONS: Final = frozenset({"prepare_commitment", "revisit_thread"})
ALL_ACTIONS: Final = frozenset({"prepare_commitment", "revisit_thread", "inquire", "hold_note"})


@dataclass(frozen=True)
class StudyOutcome:
    """What a study left: a committed result (or none), the homework items it stored, why it
    stopped early (``budget`` when Wonder's calls or dollars ran out) and its receipt's counts."""

    result: Result | None
    evidence: tuple[Evidence, ...] = ()
    stopped: str | None = None
    receipt: Mapping[str, object] = field(default_factory=dict)


class Homework(Protocol):
    """Task 7's homework agent: studies ``target`` for the chosen ``action``, its items stamped
    ``at``. It debits every Wonder call it makes, retries and failures included, from ``meter``,
    and raises ``WonderSpent`` when not one call fits."""

    available: bool

    def study(
        self, user_id: str, target: Target, night: date, meter: WonderMeter, *, action: str, at: datetime,
    ) -> StudyOutcome: ...


class NoHomework:
    """No homework (tests and the public example): no study is offered."""

    available: bool = False

    def study(
        self, user_id: str, target: Target, night: date, meter: WonderMeter, *, action: str, at: datetime,
    ) -> StudyOutcome:
        raise RuntimeError("no homework port: the night never offers a study")


@dataclass(frozen=True)
class InquiryOutcome:
    """An inquiry's result (an anchored result, or none), its thinker cycles (her own Glimmer, never
    debited from the night's actions), its research calls (Wonder, debited), why it stopped early
    (``budget``: a hold found no room; ``failed``) and its receipt's content-free counts."""

    result: Result | None
    cycles: int = 0
    research_calls: int = 0
    stopped: str | None = None
    receipt: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MindCall:
    """What a MIND step reads of one person's night (Task 17): her stores, her meters, the night's
    clock (``as_of``) and her records' stamp (``at``: the window's wall-clock end, never after
    ``as_of``), the step's job key for its holds, and the cancel flag, read before each write (RF2)."""

    user_id: str
    night: date
    as_of: datetime
    at: datetime
    store: EvidenceStore
    targets: TargetStore
    spend: SpendMeter
    wonder: WonderMeter
    actions_left: int
    job: str
    cancelled: Callable[[], bool]
    core_view: CoreView | None = None  # tonight's dream, for the inquiry's decider (MIND's core on)
    serving: str | None = None  # the adapter that serves her as the dream starts (MIND's core pairs its projector)


class Inquiry(Protocol):
    """Task 17's step 4 and the share side of step 10: the questions her workspace holds (for the
    snapshot), the close marks the questions closed tonight owe (``settle``), the results that still
    stand (``results``), an inquiry into one question, the results her findings raise ``share``
    with, and a shared finding's facts for the note."""

    available: bool

    def questions(self, call: MindCall) -> tuple[QuestionState, ...]: ...

    def settle(self, call: MindCall) -> list[str]: ...

    def results(self, call: MindCall) -> tuple[Result, ...]: ...

    def inquire(self, call: MindCall, question_id: str | None) -> InquiryOutcome: ...

    def share(self, call: MindCall) -> tuple[Result, ...]: ...

    def facts(self, call: MindCall, target_id: str) -> list[Fact]: ...


class NoInquiry:
    """No inquiry: no question, so no inquiry runs and nothing is shared."""

    available: bool = False

    def questions(self, call: MindCall) -> tuple[QuestionState, ...]:
        return ()

    def settle(self, call: MindCall) -> list[str]:
        return []

    def results(self, call: MindCall) -> tuple[Result, ...]:
        return tuple(call.targets.results())

    def inquire(self, call: MindCall, question_id: str | None) -> InquiryOutcome:
        raise RuntimeError("no inquiry port")

    def share(self, call: MindCall) -> tuple[Result, ...]:
        return ()

    def facts(self, call: MindCall, target_id: str) -> list[Fact]:
        return []


@dataclass(frozen=True)
class Dreamt:
    """What the night's dream (step 2d) left for the steps after it: her core state after every
    event so far, consolidated (``state``: what practice reads and the night's rows are made under;
    ``state_id``, its id in her store and the thought channel's ``state_sha``), the evidence
    practice is seeded on (empty: today's newest), the surprise weights of the evidence it observed
    tonight by evidence id, the core's view for the inquiry (``view.predictions`` is
    ``Core.predict(state)``), the projector her last kept unit names (``MindStore.before(night)``;
    None: the shared one), and the dream's receipt (ids and numbers, for the night log)."""

    state_id: str | None = None
    state: CoreState | None = None
    rehearse: tuple[str, ...] = ()
    weights: Mapping[str, float] = field(default_factory=dict)
    view: CoreView | None = None
    projector: str | None = None
    receipt: Mapping[str, object] = field(default_factory=dict)

    def reading(self) -> MindReading | None:
        """What the thought channel reads of this dream (``channel_night.MindReading``), or None
        when nothing was dreamt."""
        if self.state is None or self.state_id is None or self.view is None:
            return None
        return MindReading(self.state, self.state_id, self.view.predictions, self.projector)


class MindCore(Protocol):
    """MIND's core in the night (the one system), for the people it ``serves`` only: ``dream``
    observes her evidence it hasn't yet, dreams, opens questions, picks what practice rehearses and
    consolidates; ``dreamt`` is tonight's dream (its receipt holds the imagined states the sleep
    objective's dreams start from, ``night/core_dreams.py``) and ``reading`` the thought channel's
    ``MindState`` (tonight's dream as the channel reads it), each None for anyone it doesn't serve
    and refused when asked before the dream; ``imagined`` is one of the states tonight's dream kept,
    by its id, as the channel reads it (the dreams through the channel, ``night/core_dreams.py``);
    ``keep`` saves the night's unit (her state, the thought channel's projector and her adapter).
    Ids only cross: no row's words."""

    def serves(self, user_id: str) -> bool: ...

    def dream(self, call: MindCall, evidence: Sequence[Evidence]) -> Dreamt: ...

    def dreamt(self, user_id: str, night: date) -> Dreamt | None: ...

    def reading(self, user_id: str, night: date) -> MindReading | None: ...

    def imagined(self, user_id: str, night: date, state_ref: str) -> MindReading | None: ...

    def keep(
        self, call: MindCall, dreamt: Dreamt, *, lora: str | None, projector: str | None,
    ) -> Mapping[str, object]: ...


class NoCore:
    """MIND's core off: no dream, practice seeds on the newest evidence, rows are unweighted."""

    def serves(self, user_id: str) -> bool:
        return False

    def dream(self, call: MindCall, evidence: Sequence[Evidence]) -> Dreamt:
        return Dreamt()

    def dreamt(self, user_id: str, night: date) -> Dreamt | None:
        return None

    def reading(self, user_id: str, night: date) -> MindReading | None:
        return None

    def imagined(self, user_id: str, night: date, state_ref: str) -> MindReading | None:
        return None

    def keep(
        self, call: MindCall, dreamt: Dreamt, *, lora: str | None, projector: str | None,
    ) -> Mapping[str, object]:
        return {}


class Notice(Protocol):
    """Task 17's step 2b: one thinker pass over new evidence; returns its receipt (the questions it
    opened, as counts). It acts on nothing."""

    available: bool

    def notice(self, call: MindCall, new: Sequence[Evidence]) -> Mapping[str, object]: ...


class NoNotice:
    available: bool = False

    def notice(self, call: MindCall, new: Sequence[Evidence]) -> Mapping[str, object]:
        return {"questions_opened": 0}


class NoIntentions:
    """Synthetic people, and anyone until the intentions ledger is wired: none."""

    def intentions(self, user_id: str, as_of: datetime) -> tuple[Commitment, ...]:
        return ()


class ProbeRegistrar(Protocol):
    """Step 3: for Kevin, generate and seal the night's non-``new_work`` probes; for a synthetic
    person, load the probes sealed once. Returns every sealed probe and how they came."""

    def register(self, user_id: str, night: date, store: EvidenceStore) -> tuple[list[Probe], str]: ...


class PracticeRunner(Protocol):
    """Step 5. ``max_spark_calls`` bounds its Spark Open calls, for the step's hold."""

    @property
    def max_spark_calls(self) -> int: ...

    def run(
        self, user_id: str, night: date, evidence: Sequence[Evidence], hw: object | None, *, at: datetime,
    ) -> PracticeRun: ...


class TrainingData(Protocol):
    """The night's dataset as the trainer sees it (``anjo.sleep.train.dataset.Dataset``)."""

    @property
    def examples(self) -> Sequence[object]: ...

    def to_jsonl(self) -> str: ...


class Trainer(Protocol):
    """Step 7: train ``name`` on ``data`` (warm from ``prev`` when the recipe says so), copy it to
    the serving volume, and return the checkpoint and the training receipt (counts and times)."""

    def train(
        self, user_id: str, night: date, prev: CheckpointRef | None, data: TrainingData, name: str,
    ) -> tuple[CheckpointRef, Mapping[str, object]]: ...


GateCheck = Callable[[CheckpointRef], GateResult]


class Gates(Protocol):
    """Step 8: the person's three gates, in order general, collapse, safety."""

    def for_person(self, user_id: str) -> Mapping[str, GateCheck]: ...


class GpuSession(Protocol):
    """The batch's one GPU job: the multi-LoRA server is held (and its spend reserved) while the
    due people practise, train and are gated. ``seconds`` is how long it has been held."""

    def open(self, due: Sequence[str]) -> AbstractContextManager[None]: ...


@dataclass
class NoGpu:
    """A session that holds nothing: tests and the public example."""

    opened: list[tuple[str, ...]] = field(default_factory=list)

    @contextmanager
    def _held(self, due: tuple[str, ...]) -> Iterator[None]:
        self.opened.append(due)
        yield

    def open(self, due: Sequence[str]) -> AbstractContextManager[None]:
        return self._held(tuple(due))


@dataclass(frozen=True)
class Prices:
    """List prices for the cost rule (§5.5): Spark Open per million tokens, and the on-demand GPU
    hour of the training and the serving GPU at a production cloud."""

    spark_in_per_m: float
    spark_out_per_m: float
    train_gpu_per_hour: float
    serve_gpu_per_hour: float


@dataclass(frozen=True)
class NightSettings:
    """The registered night configuration (``thresholds.json`` and the recipe)."""

    token_cap: int
    min_new_examples: int
    dropout: float = 0.6
    seed: int = 20261028
    homework_call_cap: int = 12
    wonder_usd_month: float = 3.0
    rolling_actions: int = 4
    practice_scenes: int = 16  # full practice's scenes a night, 16 to 20
    practice_teacher: bool = True  # full practice's Spark Open teacher candidate
    mind_inquiry: bool = False  # MIND's inquiry in the night (Task 17: steps 2b, 4 and 10); off is Task 7a's night
    quiet_inquiry: bool = False  # accelerated alone nights: run the controller even without a chosen question
    force_deep: bool = False  # a deep sleep tonight whatever the pressure (``--force-deep``); the month's cap still wins


@dataclass(frozen=True)
class NightDeps:
    extractor: Callable[[ExtractChat], EvidenceSource]
    extract_chat: ExtractChat
    evidence: Callable[[str], EvidenceStore]
    targets: Callable[[str], TargetStore]
    transcripts: Callable[[str], TranscriptStore]
    briefs: Callable[[str], JsonlBriefStore]
    spark_chat: ChatFn
    probes: ProbeRegistrar
    chooser: Chooser
    policy: Policy
    practice: Callable[[CappedChat, date], PracticeRunner]  # the night's Spark Open chat, metered
    decontam: Decontaminator
    trainer: Trainer
    gates: Gates
    registry: CheckpointRegistry
    notifier: Notifier
    note_writer: NoteWriter
    logs: Callable[[str], Path]
    is_synthetic: Callable[[str], bool]
    settings: NightSettings
    prices: Prices
    gpu: GpuSession = field(default_factory=NoGpu)
    homework: Homework = field(default_factory=NoHomework)
    inquiry: Inquiry = field(default_factory=NoInquiry)
    notice: Notice = field(default_factory=NoNotice)
    core: MindCore = field(default_factory=NoCore)  # MIND's core (ANJO_SLEEP_MIND_CORE=1); off, the night is today's
    # The comparison's conventional agent (arm 2A, ``ANJO_SLEEP_AGENT_BASELINE=1``); None is off, and
    # the night is byte-identical to one run before the arm.
    baseline: AgentBaseline | None = None
    estimator: Coefficients | None = None  # Task 16's registered estimator
    intentions: IntentionSource = field(default_factory=NoIntentions)
    cancel: Callable[[str], bool] = field(default=lambda user_id: False)
    # Removes what a step wrote after the person was deleted mid-step (RF2, Task 9); idempotent.
    purge: Callable[[str], None] = field(default=lambda user_id: None)
    timer: Callable[[], float] | None = None
    hosted_in_training: bool = False
    # Every paid step holds its worst case here first (holds.py); None holds nothing (tests).
    ledger: Ledger | None = None
    # The app's reply-side crisis check on the note's final text, sources and all; False declines it.
    note_check: Callable[[str], bool] = field(default=lambda text: True)
    # Wakes Glimmer before the first Modal step (practice or gates); None wakes nothing (tests, fakes).
    wake: Callable[[], WakeResult] | None = None
    # Puts Glimmer back to sleep when the batch ends, however it ends; None sleeps nothing (Modal idles out).
    sleep: Callable[[], None] | None = None
    # On Modal: stops the serving GPU when the batch's outermost phase ends, unless a live chat could be cut
    # (``serving.stop``); in place of ``sleep``, and each night logs its outcome (``gpu_stop``).
    gpu_stop: GpuStop | None = None
    # The Glimmer client's running count of requests it sent again (``VllmClient.retries``); logged per step.
    retries: Callable[[], int] = field(default=lambda: 0)
    # sft_dpo_seed's extra rows (``night/dreams.py``): off unless the night command's switches say so.
    terms: Terms = field(default_factory=Terms)
    dreams: DreamSource = field(default_factory=NoDreams)
    # Nap or deep sleep (``sleep_pressure.py``); None: every night may train, the night before naps (tests, fakes).
    sleep_policy: SleepPolicy | None = None
    # The person's own turns inside the night's window, a count; None counts none.
    turns: Callable[[str, tuple[datetime, datetime]], int] | None = None
    # MIND's core's ``sleep_pressure(state)`` drops in here; the stand-in until it ships.
    sleep_pressure: Callable[[SleepState], float] = sleep_pressure
    # The thought channel (``channel_night``): MIND's state of a person as of the night, and the channel's
    # setup. Either None is today's night: no soft tokens, no ``core`` on any row.
    mind_state: MindState | None = None
    channel: ChannelSetup | None = None


@dataclass(frozen=True)
class NightOptions:
    """What the command line (``python -m anjo.sleep.night``) hands the app's wiring with the
    trainer: the batch's night, and each synthetic person's persona (their sealed probes and the
    collapse gate's fields). A person without a persona is real, and runs only with ``allow_real``.
    ``practice_scenes`` and ``practice_teacher`` set full practice for the night (Task 7);
    ``mind_inquiry`` turns on MIND's inquiry (Task 17); ``force_deep`` makes tonight a deep sleep
    whatever its pressure (``night/sleep_pressure.py``); ``practice_roles`` who plays practice's roles
    (``practice.roles``; None reads ``ANJO_SLEEP_PRACTICE_ROLES``, as the night does)."""

    night: date
    personas: Mapping[str, str]
    allow_real: bool = False
    practice_scenes: int = 16
    practice_teacher: bool = True
    mind_inquiry: bool = False
    force_deep: bool = False
    practice_roles: Roles | None = None


Wiring = Callable[[Trainer, NightOptions], NightDeps]
