"""``sft_dpo_seed``: the sleep paradigm's one loss, in the night's one training job (2026-10-29).

After Behrouz, Hashemi and Mirrokni, "Language Models Need Sleep" (arXiv 2606.03979). Glimmer is dense,
so the person's LoRA is the expansion slot: only its parameters ever train, and the base stays frozen.
One job trains every term, with these weights (``WEIGHTS``):

- ``sft`` 1.0: the night's rows as today (her best replies and the critic's Spark Open corrections);
  each row's loss, and each DPO pair's, is multiplied by its own ``weight`` before the mean: MIND's
  surprise at the evidence it came from (the Sleep link, 0.5 to 2.0 at mean 1; 1.0 when it has none);
- ``dpo`` 1.0: today's DPO pass, unchanged in strength (``_dpo_block``): the (best, worst) pairs, beta
  0.1, against the same model with its adapter off, on their own AdamW at today's DPO rate
  (``Plan.dpo_lr``, 5e-6), with today's accumulation and epochs rule and its own cosine, after the main
  block as today it follows SFT. Inside the main block's optimizer its step would be set by the SFT
  rate (2e-4), some 20x today's at any weight Adam leaves unscaled, so it never shares it;
- ``seed`` 1.0, knowledge seeding: per token, over the full vocabulary, the reverse KL from the
  student (her adapter as it trains tonight, the message alone) to the teacher (her long-term adapter,
  the one the night started from, with the consolidated day before the message), on her own replies.
  The teacher is the night-start adapter (``Teacher``): with the fold, the long-term leading ranks and
  tonight's slot empty; warm without it, last night's adapter; cold, a zero B, so the base. Only the
  day in context separates teacher from student, so the KL moves the day into tonight's weights and
  never pulls her back toward the base. A share ``ON_POLICY_SHARE`` of the
  rows is scored on a fresh sample of the student instead of tonight's practice reply (GKD's lambda).
  **Reverse, not forward:** the teacher holds privileged context, so forward KL would make the
  context-free student cover every mode the context made possible and hedge or guess between them;
  reverse KL is mode-seeking and stays inside the student's own support, so it takes what the
  teacher is sure of without pulling the rest of her behaviour away (less forgetting). On the
  student's own samples it is the on-policy distillation objective (GKD), and the full-vocabulary
  sum gives a gradient at every visited position, not only the sampled token;
- ``imitate`` 0.25, learning to imitate: the student completes a random prefix (30 to 70%) of her
  reply without the context, ``IMITATION_SAMPLES`` times; the reward is edit similarity cut at ``z0``
  (1 - d / max length when d / max length <= 0.5, else 0), optimised by REINFORCE with a
  leave-one-out baseline. Edit similarity only: no cleared scorer of meaning runs in the job, so the
  paper's semantic term is left out (gamma = 0). Its weight with seeding's is the paper's alpha 0.8;
- ``dream`` 0.5: dreams, self-generated and so down-weighted. The candidates are ranked by the norm of
  their SFT gradient on the adapter; the top ``DREAM_TOP_K`` and ``DREAM_RANDOM`` more at random train
  in a block after the rest, at half the peak rate, and the block is undone unless the anchor check
  (``sets/anchors.jsonl``, disjoint from every gate, scored as the general gate scores, smoothed) holds (``ANCHOR_TOLERANCE``).
  The main block's optimizer and schedule train ``sft``, ``seed``, ``imitate`` and ``dream``.
  Each dream carries its seed's kind (``Batch.dream_groups``: ``boundary``, a state at the edge of a
  decision, or ``random``, the control), and the receipt gives the filter's choices by kind: whether
  the boundary's dreams carry more gradient than a random thread's is the cheap test of targeting.

**The matched control** (``Plan.stretch_s``, the live comparison only): the main block trains for a
time budget instead of a step plan, epochs of the rows in fresh seeded orders, each step at the rate
``stretch`` gives the share of the budget used (one warmup, one cosine to zero); then the DPO block, as
any night. It never dreams: it is the control a dreaming arm is held to.

**Capacity** (``Plan.passes``, ``ANJO_SLEEP_CAPACITY``): the main block takes ``passes`` passes over its
rows, one row a step, in place of the step plan; the DPO block keeps its own. Never with a time budget.

**Day dreams** (``ANJO_SLEEP_DAY_DREAMS``): the batch's seed rows may carry a day dream's group. Before
the main block the job holds out ``DREAM_HELD`` dream rows (fixed by the plan seed), tries each group of
``DREAM_GROUP`` for ``DREAM_TRIAL_STEPS`` steps at the main rate, and keeps a group only if the held-out
seeding KL falls by ``DREAM_MIN_GAIN`` while the anchor score stays within ``ANCHOR_BAND`` of the
untouched one. Every trial restores the weights, the optimizer's state and the torch RNG exactly, and the
held-out rows never train: the same keep-if-improves selection the paper runs with isolated adapters, at
the cost of a few steps and one loaded job. The deadline stops the trials as it stops the main loop, and
with ``DREAM_HELD`` dreams or fewer there is nothing to hold out: every dream trains as an ordinary seed
row, untried. With the switch on, the dream block's rollback also restores the projector's params, and
imitation samples ``IMITATION_SAMPLES_DAY`` times.

**Self-distillation** (``Batch.sdft``, ``ANJO_SLEEP_SDFT``), after Shenfeld, Damani, Hübotter and Agrawal,
"Self-Distillation Enables Continual Learning" (arXiv 2601.19897): the model with a demonstration in
context teaches itself without it, on its own samples. Each ``SdftSeq`` is a query the host built from a
seeding row's message (``train/terms.py``: bare, asking for a read, or asking for a choice) with the
consolidated day before it in the teacher's prompt. Every visit draws a fresh sample of the student
(``SDFT_TOKENS``, temperature 1, no top-p or top-k: fully on-policy, one rollout a visit, as the paper) and
scores it with ``seed_kl``: the same per-token, full-vocabulary reverse KL to the same ``Teacher``, so the
term is the seeding term's machinery on the student's own replies, at ``SDFT_WEIGHT``. The paper's teacher
is an EMA of the student (the reference code syncs it every step at 0.01); here it syncs once a night, to
the adapter the night starts from: the paper's stable frozen-teacher ablation within a night (an EMA at
0.01 moves 1 - 0.99^n of the way, a quarter over a step plan's thirty updates), the EMA across nights,
and ``teacher_drift`` stays zero. Reverse, as the seeding term: the paper's analysis argues for it, though its code now defaults to
the forward KL. The rows of ``SDFT_HELD`` sources, fixed by the plan seed, are never distilled, but their
seeding rows train under ``seed`` as every other's do: the reading on them is SDFT-held (seeded, not
distilled), never a held-out one. Each is sampled once from the night-start student, and the receipt's
``sdft.seeded_not_distilled`` reads the KL on those same replies before and after training. The term draws nothing once its own seconds pass ``SDFT_BUDGET_S`` or the deadline passes
(``skipped_late`` counts the visits), so its cost is bounded whatever the step plan; never in a matched
control.

**The fold** (``sleep_fold.py``): with ``long_term_rank`` set, the job trains a LoRA whose first ranks
hold the person's long-term adapter, frozen by gradient masks (``freeze_leading``), and whose last ranks
are tonight's fresh slot (B at zero: the reset, ``load_leading``); after training the job folds it.

**The thought channel** (``Soft``, the recipe's ``channel`` block): a row with a ``state`` reads it as K
soft tokens from the projector (``channel_train.Channel``), spliced just before the prompt's last
``<|eot|>`` as the server splices them, and the projector trains in the same loop: its groups in the main
block's optimizer at their own rates (the first ``warmup_share`` of the main updates the projector's
alone), its MLP in the DPO block's (whose reference reads the projector as the night began). Only ``sft``
and ``dpo`` teach it: seeding's student, imitation and dreams read the state through it detached, and
seeding's teacher reads it as the night began (``sleep_job.py`` says why).

It is standalone, as the training script is: the standard library at the top and torch inside the
functions. Modal's image carries it beside ``colab_train.py``, which loads it by its path. The host's
tests import it as a module and run every function on a tiny CPU model.
"""
from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, Final

WEIGHTS: Final = {"sft": 1.0, "dpo": 1.0, "seed": 1.0, "imitate": 0.25, "dream": 0.5}
SURPRISE_TERMS: Final = ("sft", "dpo")  # the terms whose rows carry MIND's surprise weight
ON_POLICY_SHARE: Final = 0.25  # GKD's lambda: the seed rows scored on a fresh student sample
FRESH_TOKENS: Final = 192
IMITATION_ROWS: Final = 4
IMITATION_SAMPLES: Final = 2
IMITATION_SAMPLES_DAY: Final = 4  # day dreams: four samples, their rows are the volume the switch buys
IMITATION_TOKENS: Final = 48
IMITATION_Z0: Final = 0.5
PREFIX_SHARE: Final = (0.3, 0.7)
DREAM_TOP_K: Final = 6
DREAM_RANDOM: Final = 2
DREAM_RATE: Final = 0.5  # the dream block's share of the peak learning rate
ANCHOR_TOLERANCE: Final = 0.0  # a dream block is kept only if the anchor score doesn't fall
DREAM_KIND: Final = "dream"  # a seed row's group when it is one of the day's dreams
DREAM_HELD: Final = 4  # the day dreams tau is measured on, fixed by the plan seed, never trained
DREAM_GROUP: Final = 4  # day dreams a trial trains and judges together
DREAM_TRIAL_STEPS: Final = 4  # steps a group is tried at, at the main rate
DREAM_MIN_GAIN: Final = 0.01  # a group is kept only if held-out seeding KL falls by this share
ANCHOR_BAND: Final = 0.005  # and the anchor score stays within this of the untouched one
MAX_GRAD_NORM: Final = 1.0
WARMUP: Final = 0.03
MAX_ACCUMULATION: Final = 16
MAX_EPOCHS: Final = 4
TEMPERATURE: Final = 1.0
SDFT_TERM: Final = "sdft"  # self-distillation's term, apart from ``WEIGHTS``: off, the receipt is today's
SDFT_WEIGHT: Final = 1.0  # the seeding KL on the student's own sample: one row weighs as one seed row
SDFT_KINDS: Final = ("knowledge", "tone", "taste")  # what a query asks for (``terms.SDFT_KINDS``)
SDFT_TOKENS: Final = FRESH_TOKENS  # the student's sample a visit scores (``terms.SDFT_REPLY_TOKENS``)
SDFT_HELD: Final = 2  # sources never distilled (their seed rows still train): the SDFT-held reading's rows
SDFT_HELD_MIN: Final = 4  # with fewer sources no source is held from distillation, and the reading is None
SDFT_HELD_LABEL: Final = "SDFT-held (seeded, not distilled)"  # what the receipt's reading is, in its own words
SDFT_BUDGET_S: Final = 15 * 60.0  # the term's seconds a night; past them a visit draws nothing

Ids = tuple[int, ...]


@dataclass(frozen=True)
class Seq:
    prompt: Ids
    reply: Ids
    weight: float = 1.0  # MIND's surprise weight on an SFT row (the Sleep link); a dream's is always 1.0
    state: str | None = None  # the reading its reply was made under (``Soft.readings``), channel on


@dataclass(frozen=True)
class PairSeq:
    prompt: Ids
    chosen: Ids
    rejected: Ids
    weight: float = 1.0  # MIND's surprise weight on the pair
    state: str | None = None


@dataclass(frozen=True)
class SeedSeq:
    """The student's prompt (no context), the teacher's (the consolidated day, then the same message), the reply.
    ``group`` marks one of the day's dreams (``"dream"``); None is every other row, replay included."""

    prompt: Ids
    teacher: Ids
    reply: Ids
    state: str | None = None
    group: str | None = None


@dataclass(frozen=True)
class SdftSeq:
    """A self-distillation row: the student's prompt (the query alone), the teacher's (the consolidated day,
    then the same query), what the query asks for and the seeding row it was built from. It has no reply:
    every visit scores a fresh sample of the student."""

    prompt: Ids
    teacher: Ids
    kind: str
    source: str
    state: str | None = None


@dataclass(frozen=True)
class AnchorSeq:
    prompt: Ids
    choices: tuple[Ids, ...]
    answer: int


@dataclass(frozen=True)
class Batch:
    sft: tuple[Seq, ...] = ()
    pairs: tuple[PairSeq, ...] = ()
    seeds: tuple[SeedSeq, ...] = ()
    dreams: tuple[Seq, ...] = ()
    anchors: tuple[AnchorSeq, ...] = ()
    dream_groups: tuple[str, ...] = ()  # each dream's seed kind, in order; empty means one group
    sdft: tuple[SdftSeq, ...] = ()  # ANJO_SLEEP_SDFT's rows, each scored on a fresh student sample


@dataclass(frozen=True)
class Plan:
    lr: float
    min_steps: int
    beta: float
    seed: int
    eot: int
    pad: int
    dpo_lr: float  # today's DPO rate: the DPO block's own optimizer, never the main block's rate
    long_term_rank: int = 0  # the fold: the leading ranks that hold the long-term adapter, frozen
    dreams: bool = False
    day_dreams: bool = False  # ANJO_SLEEP_DAY_DREAMS: the dream block's rollback covers the projector too
    stretch_s: float = 0.0  # the matched control: the main block trains this long by the clock (0: the step plan)
    deadline: float = math.inf  # the timer's reading past which no update starts (the channel's ``max_minutes``)
    passes: int = 0  # capacity: the main block's passes over its rows, one row a step (0: the step plan)


Spliced = tuple[Any, Any]  # the thought channel (``channel_train.Channel``) and K soft tokens ``(K, H)``


@dataclass(frozen=True)
class Soft:
    """The thought channel in the one loss: ``channel`` (``channel_train.Channel``, the projector on the
    model), the readings the rows' ``state`` names, the projector's groups in the main block's optimizer
    (``Channel.projector_groups``), its tensors the DPO block trains with the adapter (its MLP), and the
    share of the main block's first updates the projector takes alone."""

    channel: Any
    readings: Mapping[str, Any]
    groups: tuple[Mapping[str, Any], ...] = ()
    dpo_params: tuple[Any, ...] = ()
    warmup_share: float = 0.0

    def tokens(self, state: str | None, *, ref: bool = False, detach: bool = False) -> Spliced | None:
        """``state``'s soft tokens with the channel that splices them: from the projector now, or as the
        night began (``ref``); ``detach`` keeps the term's gradient off the projector. None without a state."""
        if state is None:
            return None
        soft = self.channel.soft(self.readings[state], ref=ref)
        return self.channel, soft.detach() if detach else soft


def _tokens(soft: Soft | None, state: str | None, *, ref: bool = False, detach: bool = False) -> Spliced | None:
    return None if soft is None else soft.tokens(state, ref=ref, detach=detach)


# ── the pure parts ───────────────────────────────────────────────────────────


def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def edit_reward(sample: Sequence[int], target: Sequence[int], z0: float = IMITATION_Z0) -> float:
    """The paper's r_abs: 1 - d / max length when that share is at most ``z0``, else 0."""
    longest = max(len(sample), len(target))
    if longest == 0:
        return 1.0
    z = edit_distance(sample, target) / longest
    return 1.0 - z if z <= z0 else 0.0


def select_dreams(norms: Sequence[float], k: int, b: int, seed: int) -> list[int]:
    """The top ``k`` by gradient norm (ties to the earlier), then ``b`` of the rest at random, seeded."""
    order = sorted(range(len(norms)), key=lambda i: (-norms[i], i))
    top, rest = order[:k], order[k:]
    return top + random.Random(seed).sample(rest, min(b, len(rest)))


def dream_split(seeds: Sequence[SeedSeq], seed: int) -> tuple[list[int], list[list[int]]]:
    """The day dreams' trial split: ``DREAM_HELD`` rows fixed by the run's seed as tau's set (never
    trained once a group is tried), then the rest in groups of ``DREAM_GROUP`` the trials judge. With
    ``DREAM_HELD`` dreams or fewer the held set is every dream and there are no groups to try: those
    rows train as ordinary seed rows."""
    dream = [i for i, row in enumerate(seeds) if row.group == DREAM_KIND]
    held = random.Random(seed).sample(dream, min(DREAM_HELD, len(dream)))
    chosen = set(held)
    rest = [i for i in dream if i not in chosen]
    return held, [rest[i:i + DREAM_GROUP] for i in range(0, len(rest), DREAM_GROUP)]


def dream_kept(tau_before: float, tau_after: float, anchor_before: float, anchor_after: float) -> bool:
    """The paper's dream selection, cheap: the held-out seeding KL falls by ``DREAM_MIN_GAIN``
    relative to the untouched adapter, and the anchor score stays within ``ANCHOR_BAND`` of it."""
    return tau_after <= tau_before * (1.0 - DREAM_MIN_GAIN) and anchor_after >= anchor_before - ANCHOR_BAND


def step_plan(n: int, min_steps: int) -> tuple[int, int, int]:
    """Accumulation, epochs and steps an epoch, by the SFT rule (``lora.plan_steps``)."""
    accumulation = max(1, min(MAX_ACCUMULATION, n // min_steps))
    per_epoch = math.ceil(n / accumulation)
    epochs = 1 if per_epoch >= min_steps else min(MAX_EPOCHS, math.ceil(min_steps / per_epoch))
    return accumulation, epochs, per_epoch


def schedule(main_steps: int) -> Callable[[int], float]:
    """The rate's factor: a 3% warmup, a cosine over the main steps, then the dream block's share. Over
    the main steps it is today's trainer's (transformers' ``get_cosine_schedule_with_warmup``, warmup
    ``ceil(3%)`` steps): 0 on the first step, ``step / warm`` through the warmup (a test holds it)."""
    warm = max(1, math.ceil(WARMUP * main_steps))

    def factor(step: int) -> float:
        if step >= main_steps:
            return DREAM_RATE
        if step < warm:
            return step / warm
        return 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, main_steps - warm)))

    return factor


def stretch(share: float) -> float:
    """The rate's factor at ``share`` of a time budget used: a 3% linear warmup, then a cosine to zero.
    The matched control's schedule by the clock; ``colab_train.stretch`` is the same (a test holds it)."""
    share = min(max(share, 0.0), 1.0)
    if share < WARMUP:
        return share / WARMUP
    return 0.5 * (1.0 + math.cos(math.pi * (share - WARMUP) / (1.0 - WARMUP)))


def _until_eot(ids: Sequence[int], eot: int) -> Ids:
    out: list[int] = []
    for token in ids:
        out.append(token)
        if token == eot:
            break
    return tuple(out)


# ── the terms (torch) ────────────────────────────────────────────────────────


def _device(model: Any) -> Any:
    return next(model.parameters()).device


def _spliced(prompt: Ids, reply: Ids, soft: Spliced) -> Any:
    """The rows the model reads with soft tokens: ``prompt`` and ``reply`` through its input embedding, the
    tokens before the prompt's last ``<|eot|>`` (``Channel.splice``), in the model's own dtype, batched."""
    channel, tokens = soft
    rows, _ = channel.splice(prompt, reply, tokens)
    return rows[None].to(channel.dtype)


def reply_logits(model: Any, prompt: Ids, reply: Ids, soft: Spliced | None = None) -> Any:
    """The logits that predict each token of ``reply`` after ``prompt``: only those positions are kept.
    With ``soft`` the model reads the spliced rows; the reply still ends the sequence."""
    import torch

    keep = len(reply) + 1
    if soft is not None:
        return model(inputs_embeds=_spliced(prompt, reply, soft), logits_to_keep=keep).logits[0, :-1].float()
    ids = torch.tensor([[*prompt, *reply]], device=_device(model))
    return model(input_ids=ids, logits_to_keep=keep).logits[0, :-1].float()


def reply_logprobs(model: Any, prompt: Ids, reply: Ids, soft: Spliced | None = None) -> Any:
    import torch

    logp = torch.log_softmax(reply_logits(model, prompt, reply, soft), dim=-1)
    return logp.gather(-1, torch.tensor(reply, device=logp.device).view(-1, 1)).squeeze(-1)


def sft_loss(model: Any, row: Seq, soft: Spliced | None = None) -> Any:
    return -reply_logprobs(model, row.prompt, row.reply, soft).mean()


def dpo_loss(model: Any, pair: PairSeq, beta: float, soft: Spliced | None = None, ref: Spliced | None = None) -> Any:
    """DPO against the same model with its adapter off, as today's pass scores its reference; with the
    channel the policy reads the projector now (``soft``) and the reference the one the night began with (``ref``)."""
    import torch

    chosen = reply_logprobs(model, pair.prompt, pair.chosen, soft).sum()
    rejected = reply_logprobs(model, pair.prompt, pair.rejected, soft).sum()
    with torch.no_grad(), model.disable_adapter():
        reference = (reply_logprobs(model, pair.prompt, pair.chosen, ref).sum()
                     - reply_logprobs(model, pair.prompt, pair.rejected, ref).sum())
    return -torch.nn.functional.logsigmoid(beta * ((chosen - rejected) - reference))


class Teacher:
    """The seeding teacher's adapter: the adapter's values when the night began (the long-term
    adapter). A pass under it swaps those values in, runs without a gradient or dropout, and puts
    tonight's back exactly; the optimizer's state is untouched."""

    def __init__(self, params: Sequence[Any]) -> None:
        import torch

        self._params = list(params)
        self._start = [p.detach().clone() for p in self._params]
        self._live = [torch.empty_like(p) for p in self._params]

    @contextmanager
    def __call__(self, model: Any) -> Iterator[None]:
        import torch

        with torch.no_grad(), _evaluating(model):
            for p, live, start in zip(self._params, self._live, self._start, strict=True):
                live.copy_(p)
                p.copy_(start)
            try:
                yield
            finally:
                for p, live in zip(self._params, self._live, strict=True):
                    p.copy_(live)


def seed_kl(model: Any, row: SeedSeq, teacher: Teacher, soft: Spliced | None = None,
            teacher_soft: Spliced | None = None) -> Any:
    """Knowledge seeding: reverse KL(student || teacher) a reply position, averaged. The teacher is
    the night-start adapter with the day in context, scored without a gradient; the student is
    tonight's adapter on the message alone. With the channel both read the row's state: the student
    through ``soft``, the teacher through ``teacher_soft`` (the projector as the night began)."""
    import torch

    with teacher(model):
        target = torch.log_softmax(reply_logits(model, row.teacher, row.reply, teacher_soft), dim=-1)
    student = torch.log_softmax(reply_logits(model, row.prompt, row.reply, soft), dim=-1)
    return (student.exp() * (student - target)).sum(dim=-1).mean()


def _scored(row: SdftSeq, reply: Ids) -> SeedSeq:
    """A self-distillation row with the sample it is scored on, as a seeding row."""
    return SeedSeq(row.prompt, row.teacher, reply, row.state)


def sdft_kl(model: Any, row: SdftSeq, teacher: Teacher, plan: Plan, soft: Spliced | None = None,
            teacher_soft: Spliced | None = None) -> tuple[Any, Ids]:
    """Self-distillation, one visit: a fresh sample of the student (tonight's adapter, the query alone),
    then ``seed_kl`` on it, the teacher the night-start adapter with the day before the query. Returns the
    loss (None for an empty sample) and the sample."""
    reply = sample(model, row.prompt, SDFT_TOKENS, plan, soft)
    if not reply:
        return None, reply
    return seed_kl(model, _scored(row, reply), teacher, soft, teacher_soft), reply


def sdft_split(rows: Sequence[SdftSeq], seed: int) -> tuple[list[int], list[int]]:
    """The SDFT-held rows and the rows distilled: every row of ``SDFT_HELD`` sources fixed by the run's seed,
    so no held query's message is distilled under another kind; none with fewer than ``SDFT_HELD_MIN``.
    The held sources' seeding rows still train under ``seed``: held from distillation, not from seeding."""
    sources = sorted({row.source for row in rows})
    held = set(random.Random(seed).sample(sources, SDFT_HELD)) if len(sources) >= SDFT_HELD_MIN else set()
    return ([i for i, row in enumerate(rows) if row.source in held],
            [i for i, row in enumerate(rows) if row.source not in held])


@contextmanager
def _evaluating(model: Any) -> Iterator[None]:
    """Sampling and the anchor check run without dropout (and with the cache), then training resumes."""
    was = model.training
    model.eval()
    try:
        yield
    finally:
        if was:
            model.train()


def sample(model: Any, prompt: Ids, max_new: int, plan: Plan, soft: Spliced | None = None) -> Ids:
    """One sample of her adapter (on) after ``prompt``, up to ``max_new`` tokens or her end of turn; with
    ``soft``, from the spliced rows (a run from embeddings returns the new tokens only)."""
    import torch

    settings = {"do_sample": True, "temperature": TEMPERATURE, "top_p": 1.0, "top_k": 0, "max_new_tokens": max_new,
                "eos_token_id": plan.eot, "pad_token_id": plan.pad}
    with _evaluating(model), torch.no_grad():
        if soft is not None:
            rows = _spliced(prompt, (), soft)
            mask = torch.ones(rows.shape[:2], dtype=torch.long, device=rows.device)
            new = model.generate(inputs_embeds=rows, attention_mask=mask, **settings)[0]
        else:
            ids = torch.tensor([list(prompt)], device=_device(model))
            new = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **settings)[0, len(prompt):]
    return _until_eot([int(t) for t in new.tolist()], plan.eot)


def imitation_loss(model: Any, row: SeedSeq, rng: random.Random, plan: Plan,
                   soft: Spliced | None = None) -> tuple[Any, float | None]:
    """Learning to imitate: complete a prefix of the reply without the context; REINFORCE on edit
    similarity, each sample's baseline the others' mean. Returns the loss (None when every sample
    scored alike) and the mean reward (None for a reply too short to cut)."""
    import torch

    low = max(1, math.ceil(PREFIX_SHARE[0] * len(row.reply)))
    high = math.floor(PREFIX_SHARE[1] * len(row.reply))
    if high < low:
        return None, None
    cut = rng.randint(low, high)
    prompt, target = (*row.prompt, *row.reply[:cut]), row.reply[cut:cut + IMITATION_TOKENS]
    samples_taken = IMITATION_SAMPLES_DAY if plan.day_dreams else IMITATION_SAMPLES
    samples = [sample(model, prompt, len(target), plan, soft) for _ in range(samples_taken)]
    rewards = [edit_reward(s, target) for s in samples]
    total, n = sum(rewards), len(rewards)
    advantages = [r - (total - r) / (n - 1) for r in rewards]
    mean = total / n
    used = [(s, a) for s, a in zip(samples, advantages, strict=True) if s and a != 0.0]
    if not used:
        return None, mean
    gain = torch.stack([a * reply_logprobs(model, prompt, s, soft).mean() for s, a in used]).sum()
    return -gain / n, mean


def adapter_params(model: Any) -> list[tuple[str, Any]]:
    """Every trainable parameter, which must all be the adapter's: the base never trains."""
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    stray = [name for name, _ in named if "lora_" not in name]
    if stray:
        raise ValueError(f"only the adapter trains, and {len(stray)} other parameters would ({stray[0]})")
    if not named:
        raise ValueError("the adapter has no trainable parameter")
    return named


def leading(named: Sequence[tuple[str, Any]], rank: int) -> list[Any]:
    """A copy of the first ``rank`` ranks of every LoRA tensor: the long-term adapter's slots."""
    return [(p[:rank] if "lora_A" in name else p[:, :rank]).detach().clone() for name, p in named
            if "lora_A" in name or "lora_B" in name]


def freeze_leading(named: Sequence[tuple[str, Any]], rank: int) -> list[Any]:
    """Gradient masks that hold the first ``rank`` ranks (the long-term adapter) still: A's rows, B's columns."""
    def rows(grad: Any) -> Any:
        grad = grad.clone()
        grad[:rank] = 0
        return grad

    def columns(grad: Any) -> Any:
        grad = grad.clone()
        grad[:, :rank] = 0
        return grad

    return [p.register_hook(rows if "lora_A" in name else columns) for name, p in named
            if "lora_A" in name or "lora_B" in name]


def load_leading(model: Any, tensors: Mapping[str, Any], rank: int, adapter: str = "default") -> None:
    """The long-term adapter's saved pairs into the first ``rank`` ranks of each live LoRA pair; the
    rest of B is zeroed, so tonight's slot starts empty (the reset)."""
    import torch

    params = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for key, value in tensors.items():
            name = key[: -len(".weight")] + (f".{adapter}.weight" if adapter else "")
            target = params.get(name)
            if target is None or ("lora_A" in key) == ("lora_B" in key):
                raise ValueError(f"the long-term adapter's {key} has no place in tonight's LoRA")
            if "lora_A" in key:
                target[:rank].copy_(value)
            else:
                target[:, :rank].copy_(value)
                target[:, rank:].zero_()
            loaded += 1
    if loaded != sum("lora_" in name for name in params):
        raise ValueError("the long-term adapter doesn't cover tonight's LoRA modules")


def grad_norm(model: Any, params: Sequence[Any], row: Seq, soft: Spliced | None = None) -> float:
    """The norm of one row's SFT gradient on the adapter: the dream filter's importance proxy."""
    for p in params:
        p.grad = None
    sft_loss(model, row, soft).backward()
    norm = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in params if p.grad is not None))
    for p in params:
        p.grad = None
    return norm


def anchor_score(model: Any, anchors: Sequence[AnchorSeq]) -> float:
    """The general gate's rule, smoothed: each choice's summed log-probability as the reply, and the
    log-share the right one takes, averaged over the items."""
    import torch

    scores = []
    with _evaluating(model), torch.no_grad():
        for item in anchors:
            sums = torch.stack([reply_logprobs(model, item.prompt, choice).sum() for choice in item.choices])
            scores.append(float(torch.log_softmax(sums, dim=-1)[item.answer]))
    return sum(scores) / len(scores)


def mean_seed_kl(model: Any, rows: Sequence[SeedSeq], teacher: Teacher, soft: Soft | None = None) -> float:
    """Tau: the seeding KL over ``rows``, averaged, without dropout or a gradient (the trial's measure)."""
    import torch

    if not rows:
        return 0.0
    scores = []
    with _evaluating(model), torch.no_grad():
        for row in rows:
            scores.append(float(seed_kl(model, row, teacher, _tokens(soft, row.state, detach=True),
                                       _tokens(soft, row.state, ref=True))))
    return sum(scores) / len(scores)


def sdft_held(model: Any, rows: Sequence[SdftSeq], plan: Plan, soft: Soft | None = None) -> list[tuple[str, SeedSeq]]:
    """One sample of the student per SDFT-held row (seeded, not distilled), each with its kind: the replies
    the receipt's KL is read on before and after training, the same both times (an empty sample is left out)."""
    held = []
    for row in rows:
        reply = sample(model, row.prompt, SDFT_TOKENS, plan, _tokens(soft, row.state, detach=True))
        if reply:
            held.append((row.kind, _scored(row, reply)))
    return held


def sdft_kl_by_kind(model: Any, held: Sequence[tuple[str, SeedSeq]], teacher: Teacher,
                    soft: Soft | None = None) -> dict[str, float | None]:
    """The seeding KL on the SDFT-held samples, over all of them (``all``) and by kind; None where there are none."""
    groups = {"all": [seq for _, seq in held], **{kind: [seq for k, seq in held if k == kind] for kind in SDFT_KINDS}}
    return {name: mean_seed_kl(model, rows, teacher, soft) if rows else None for name, rows in groups.items()}


def _dream_trials(
    model: Any, batch: Batch, plan: Plan, teacher: Teacher, optimizer: Any, soft: Soft | None = None,
    timer: Callable[[], float] = time.monotonic,
) -> tuple[list[int], dict[str, Any]]:
    """The paper's dream selection inside the one job: hold out ``DREAM_HELD`` dream rows, try each
    group of ``DREAM_GROUP`` for ``DREAM_TRIAL_STEPS`` steps at the main rate, and keep it only if the
    held-out seeding KL falls by ``DREAM_MIN_GAIN`` and the anchor score holds within ``ANCHOR_BAND``.
    Every trial restores the weights, the optimizer's state and the torch RNG exactly, so the main
    block's dropout and on-policy samples don't depend on how many groups were tried; the held-out
    rows never train. The deadline stops the trials as it stops the main loop: checked before tau,
    before the anchors and before every group, and once it passes no further group is kept. With no
    group to try (``dreams <= DREAM_HELD``) nothing is measured: every dream trains as a seed row."""
    import torch

    held, groups = dream_split(batch.seeds, plan.seed)
    if not held and not groups:
        return [], {}
    if not groups:  # four dreams or fewer: there is no held-out set to judge a group against
        return sorted(held), {"dreams": len(held), "trialed": False}
    if not batch.anchors:
        raise ValueError("dream trials need the anchor items they score")
    held_rows = [batch.seeds[i] for i in held]
    params = [p for group in optimizer.param_groups for p in group["params"]]
    started = timer()

    def late() -> bool:
        return timer() > plan.deadline

    def cut(tau: float | None = None) -> tuple[list[int], dict[str, Any]]:
        """The receipt when the deadline stops the trials: nothing after the check was measured."""
        return [], {"dreams": len(held) + sum(len(group) for group in groups), "held": len(held),
                    "groups": [], "kept": 0, "groups_kept": 0, "tau_before": tau, "anchor_before": None,
                    "minutes": round((timer() - started) / 60.0, 4), "trialed": True, "out_of_time": True}

    if late():  # before tau, before the anchors, and before every group below
        return cut()
    tau_before = mean_seed_kl(model, held_rows, teacher, soft)
    if late():
        return cut(tau_before)
    anchor_before = anchor_score(model, batch.anchors)
    out_of_time = False
    model.train()
    kept: list[int] = []
    receipts: list[dict[str, Any]] = []
    for group in groups:
        if late():
            out_of_time = True
            break
        saved = [p.detach().clone() for p in params]
        state = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        for step in range(DREAM_TRIAL_STEPS):
            row = batch.seeds[group[step % len(group)]]  # four steps even from a short last group
            student = _tokens(soft, row.state, detach=True)
            loss = seed_kl(model, row, teacher, student, _tokens(soft, row.state, ref=True))
            (WEIGHTS["seed"] * loss).backward()
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        tau_after = mean_seed_kl(model, held_rows, teacher, soft)
        anchor_after = anchor_score(model, batch.anchors)
        with torch.no_grad():
            for p, value in zip(params, saved, strict=True):
                p.copy_(value)
        optimizer.load_state_dict(state)
        torch.set_rng_state(rng)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for p in params:
            p.grad = None
        keep = dream_kept(tau_before, tau_after, anchor_before, anchor_after)
        receipts.append({"rows": len(group), "tau_before": tau_before, "tau_after": tau_after,
                         "anchor_before": anchor_before, "anchor_after": anchor_after, "kept": keep})
        if keep:
            kept.extend(group)
    minutes = (timer() - started) / 60.0
    return kept, {"dreams": len(held) + sum(len(group) for group in groups), "held": len(held),
                  "groups": receipts, "kept": len(kept), "groups_kept": sum(g["kept"] for g in receipts),
                  "tau_before": tau_before, "anchor_before": anchor_before, "minutes": round(minutes, 4),
                  "trialed": True, "out_of_time": out_of_time}


# ── the loop ─────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self, timer: Callable[[], float]) -> None:
        self._timer = timer
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, part: str) -> Iterator[None]:
        started = self._timer()
        try:
            yield
        finally:
            self.seconds[part] = self.seconds.get(part, 0.0) + self._timer() - started


@dataclass
class _Run:
    model: Any
    plan: Plan
    params: list[Any]
    optimizer: Any
    rate: Any
    teacher: Teacher
    clock: _Clock
    rng: random.Random
    fresh: set[int]
    losses: dict[str, list[float]]
    rewards: list[float]
    soft: Soft | None = None
    timer: Callable[[], float] = time.monotonic
    sampled: int = 0
    sdft_sampled: int = 0
    sdft_late: int = 0
    steps: int = 0
    projector_only: int = 0
    warmup_updates: int = 0
    out_of_time: bool = False
    grads_checked: bool = False

    def tokens(self, state: str | None, *, ref: bool = False, detach: bool = False) -> Spliced | None:
        return _tokens(self.soft, state, ref=ref, detach=detach)

    def late(self) -> bool:
        """Whether the deadline has passed; once it has, no update starts."""
        self.out_of_time = self.out_of_time or self.timer() > self.plan.deadline
        return self.out_of_time

    def loss(self, term: str, row: Any) -> Any:
        if term == "dpo":
            return dpo_loss(self.model, row, self.plan.beta, self.tokens(row.state), self.tokens(row.state, ref=True))
        if term == "seed":  # the student reads the state detached; the teacher, as the night began
            index, seq = row
            student = self.tokens(seq.state, detach=True)
            if index in self.fresh and (fresh := sample(self.model, seq.prompt, FRESH_TOKENS, self.plan, student)):
                seq, self.sampled = replace(seq, reply=fresh), self.sampled + 1
            return seed_kl(self.model, seq, self.teacher, student, self.tokens(seq.state, ref=True))
        if term == SDFT_TERM:  # every visit a fresh sample; past the term's budget or the deadline, none
            _, row = row
            if self.clock.seconds.get(SDFT_TERM, 0.0) > SDFT_BUDGET_S or self.timer() > self.plan.deadline:
                self.sdft_late += 1
                return None
            loss, reply = sdft_kl(self.model, row, self.teacher, self.plan, self.tokens(row.state, detach=True),
                                  self.tokens(row.state, ref=True))
            self.sdft_sampled += bool(reply)
            return loss
        if term == "imitate":
            loss, reward = imitation_loss(self.model, row, self.rng, self.plan, self.tokens(row.state, detach=True))
            if reward is not None:
                self.rewards.append(reward)
            return loss
        return sft_loss(self.model, row, self.tokens(row.state, detach=term == "dream"))  # an imagined state never teaches it

    def backward(self, chunk: Sequence[tuple[str, Any]]) -> None:
        """Each row's weighted loss, its share of the chunk's mean, into the adapter's gradient: the
        term's weight, and on an SFT row or a DPO pair its own surprise weight."""
        for term, row in chunk:
            with self.clock(term):
                loss = self.loss(term, row)
                if loss is not None:
                    surprise = row.weight if term in SURPRISE_TERMS else 1.0
                    (_weight(term) * surprise * loss / len(chunk)).backward()
                    self.losses[term].append(float(loss.detach()))

    def step(self, chunk: Sequence[tuple[str, Any]], optimizer: Any = None, rate: Any = None) -> None:
        """One update on ``chunk``: the main block's optimizer and schedule unless others are given. In the
        main block's first ``warmup_updates`` the adapter's gradients are dropped: the projector learns alone."""
        import torch

        main = optimizer is None
        optimizer, rate = optimizer or self.optimizer, rate or self.rate
        self.backward(chunk)
        if self.steps == 0:
            self.check_grads()
        if main and self.steps < self.warmup_updates:
            for p in self.params:
                p.grad = None  # AdamW skips a tensor with no gradient
            self.projector_only += 1
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], MAX_GRAD_NORM)
        optimizer.step()
        rate.step()
        optimizer.zero_grad(set_to_none=True)
        self.steps += 1

    def check_grads(self) -> None:
        """After the first backward pass: a gradient on anything but the adapter stops the run."""
        stray = [name for name, p in self.model.named_parameters() if p.grad is not None and "lora_" not in name]
        if stray:
            raise ValueError(f"{len(stray)} parameters outside the adapter got a gradient ({stray[0]})")
        self.grads_checked = True


def _weight(term: str) -> float:
    return SDFT_WEIGHT if term == SDFT_TERM else WEIGHTS[term]


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _stream(batch: Batch, rng: random.Random) -> tuple[list[tuple[str, Any]], int]:
    """The main block's rows in one seeded order, and how many seed rows imitate too."""
    long_enough = [seq for seq in batch.seeds if len(seq.reply) >= 2]
    imitate = rng.sample(long_enough, min(IMITATION_ROWS, len(long_enough)))
    items: list[tuple[str, Any]] = [
        *(("sft", row) for row in batch.sft), *(("seed", (i, seq)) for i, seq in enumerate(batch.seeds)), *(("imitate", seq) for seq in imitate),
        *((SDFT_TERM, (i, seq)) for i, seq in enumerate(batch.sdft)),
    ]
    rng.shuffle(items)
    return items, len(imitate)


class _Held:
    """A schedule that leaves the rate where the stretched block set it."""

    def step(self) -> None:
        return None


def _stretched(run: _Run, items: Sequence[tuple[str, Any]], accumulation: int, timer: Callable[[], float]) -> dict[str, Any]:
    """The matched control's main block: fresh seeded orders of ``items`` until ``plan.stretch_s`` is
    used, each step at the rate ``stretch`` gives the share used. No step starts past the budget."""
    budget, started, steps = run.plan.stretch_s, timer(), 0
    while items and timer() - started < budget and not run.late():
        order = list(items)
        run.rng.shuffle(order)
        for chunk in _chunks(order, accumulation):
            if (used := timer() - started) >= budget or run.late():
                break
            for group in run.optimizer.param_groups:  # each group at its own peak (the projector's too)
                group["lr"] = group["initial_lr"] * stretch(used / budget)
            run.step(chunk, rate=_Held())
            steps += 1
    return {"budget_minutes": round(budget / 60.0, 4), "minutes": round((timer() - started) / 60.0, 4), "steps": steps}


def _dpo_block(run: _Run, pairs: Sequence[PairSeq]) -> dict[str, Any]:
    """Today's DPO pass inside the one job: its own AdamW at today's rate, today's accumulation and
    epochs rule (``colab_train.dpo_plan``), its own cosine with a 3% warmup, clipped at 1.0, each epoch
    in a seeded order. Returns its steps, rate and the largest change one step made to any weight."""
    import torch

    if not pairs:
        return {"steps": 0, "lr": run.plan.dpo_lr, "accumulation": 0, "epochs": 0, "max_update": 0.0}
    accumulation, epochs, per_epoch = step_plan(len(pairs), run.plan.min_steps)
    projector = list(run.soft.dpo_params) if run.soft is not None else []
    optimizer = torch.optim.AdamW([*run.params, *projector], lr=run.plan.dpo_lr, weight_decay=0.0)
    rate = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(epochs * per_epoch))
    largest, steps = 0.0, 0
    for _ in range(epochs):
        order = list(pairs)
        run.rng.shuffle(order)
        for chunk in _chunks([("dpo", pair) for pair in order], accumulation):
            if run.late():
                break
            before = [p.detach().clone() for p in run.params]
            run.step(chunk, optimizer, rate)
            largest = max(largest, max(float((p.detach() - b).abs().max()) for p, b in zip(run.params, before, strict=True)))
            steps += 1
    return {"steps": steps, "lr": run.plan.dpo_lr, "accumulation": accumulation, "epochs": epochs,
            "max_update": largest}


def _dream_block(run: _Run, batch: Batch, accumulation: int) -> dict[str, Any]:
    """Rank the dreams by gradient norm, keep the top k plus b at random, train them, and keep the
    block only if the anchor score holds; else the weights go back (the projector's too with day dreams)."""
    import torch

    with run.clock("dream_rank"):
        norms = [grad_norm(run.model, run.params, row, run.tokens(row.state, detach=True)) for row in batch.dreams]
    kept = select_dreams(norms, DREAM_TOP_K, DREAM_RANDOM, run.plan.seed)
    with run.clock("anchor"):
        before = anchor_score(run.model, batch.anchors)
    restorable = ([p for group in run.optimizer.param_groups for p in group["params"]]
                  if run.plan.day_dreams else run.params)  # day dreams: the projector rolls back too
    saved = [p.detach().clone() for p in restorable]
    for chunk in _chunks([("dream", batch.dreams[i]) for i in kept], accumulation):
        run.step(chunk)
    with run.clock("anchor"):
        after = anchor_score(run.model, batch.anchors)
    held = after >= before - ANCHOR_TOLERANCE
    if not held:
        with torch.no_grad():
            for p, value in zip(restorable, saved, strict=True):
                p.copy_(value)
    return {"candidates": len(batch.dreams), "kept": len(kept), "keep_rate": round(len(kept) / len(batch.dreams), 4),
            "block_kept": held, "anchor_before": before, "anchor_after": after,
            "norm_kept_mean": sum(norms[i] for i in kept) / len(kept), "norm_mean": sum(norms) / len(norms),
            "by_seed": by_group(batch.dream_groups or ("",) * len(norms), norms, kept, DREAM_TOP_K)}


def by_group(groups: Sequence[str], norms: Sequence[float], kept: Sequence[int], k: int) -> dict[str, dict[str, Any]]:
    """The dream filter's choices by seed kind: candidates, kept (the top k and the random few), how
    many of the top k by gradient norm (the importance pick alone) and the kind's mean norm."""
    top = set(kept[:k])
    out: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        mine = [i for i, g in enumerate(groups) if g == group]
        chosen = sum(i in kept for i in mine)
        out[group] = {"candidates": len(mine), "kept": chosen, "keep_rate": round(chosen / len(mine), 4),
                      "top_k": sum(i in top for i in mine), "norm_mean": sum(norms[i] for i in mine) / len(mine)}
    return out


def _probe(model: Any, row: SeedSeq | None, teacher: Teacher, soft: Soft | None) -> tuple[Any, float | None]:
    """The teacher's logits on ``row`` (night-start adapter and projector, context on) and the seeding KL
    there, without dropout or a gradient: taken before and after training, the first must not move and
    the second should."""
    import torch

    if row is None:
        return None, None
    start = _tokens(soft, row.state, ref=True)
    with teacher(model):
        logits = reply_logits(model, row.teacher, row.reply, start)
    with _evaluating(model), torch.no_grad():
        return logits, float(seed_kl(model, row, teacher, _tokens(soft, row.state, detach=True), start))


def train_terms(model: Any, batch: Batch, plan: Plan, *, soft: Soft | None = None,
                timer: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Train the adapter (and with ``soft`` the projector) on the one loss; returns the receipt: rows,
    mean loss and minutes per term, the on-policy samples taken, the imitation reward, the dream block's
    outcome, the day dream trials' groups, and whether the deadline cut the loop short. Counts only."""
    import torch

    named = adapter_params(model)
    dreaming = plan.dreams and bool(batch.dreams)
    if dreaming and not batch.anchors:
        raise ValueError("dreams need the anchor items their check scores")
    if plan.stretch_s and (plan.stretch_s < 0 or dreaming or plan.passes or plan.day_dreams):
        raise ValueError("a matched control trains a positive time budget, and no dreams or passes")
    if batch.dream_groups and len(batch.dream_groups) != len(batch.dreams):
        raise ValueError("each dream has one seed kind")
    if batch.sdft and (plan.stretch_s or any(row.kind not in SDFT_KINDS for row in batch.sdft)):
        raise ValueError(f"self-distillation rows ask for {', '.join(SDFT_KINDS)}, and never in a matched control")
    rng = random.Random(plan.seed)
    torch.manual_seed(plan.seed)
    params = [p for _, p in named]
    teacher = Teacher(params)  # the night-start adapter: the long-term one, before anything trains
    groups = [{"params": params, "lr": plan.lr}, *(dict(group) for group in (soft.groups if soft else ()))]
    optimizer = torch.optim.AdamW(groups, lr=plan.lr, weight_decay=0.0)  # no decay: a masked rank stays put
    hooks = freeze_leading(named, plan.long_term_rank) if plan.long_term_rank else []
    held = leading(named, plan.long_term_rank) if plan.long_term_rank else []
    clock = _Clock(timer)
    try:
        trialed, trials = [], {}
        if any(row.group == DREAM_KIND for row in batch.seeds):
            with clock("dream_trials"):  # the trials' seconds are in the receipt, not only their own block
                trialed, trials = _dream_trials(model, batch, plan, teacher, optimizer, soft, timer)
        seeds = batch.seeds
        if trials:  # held-out rows and dropped groups never train; kept groups train as ordinary seeds
            keep = set(trialed)
            seeds = tuple(row for i, row in enumerate(batch.seeds)
                          if row.group != DREAM_KIND or i in keep)
        held_sdft, trained_sdft = sdft_split(batch.sdft, plan.seed)
        items, imitating = _stream(replace(batch, seeds=seeds, sdft=tuple(batch.sdft[i] for i in trained_sdft)), rng)
        fresh = set(rng.sample(range(len(seeds)), math.ceil(ON_POLICY_SHARE * len(seeds))))
        accumulation, epochs, per_epoch = step_plan(len(items), plan.min_steps) if items else (1, 0, 0)
        if items and plan.passes:  # capacity: every row its own step, each pass; the DPO block keeps its own plan
            accumulation, epochs, per_epoch = 1, plan.passes, len(items)
        main = epochs * per_epoch
        dream_steps = math.ceil(min(len(batch.dreams), DREAM_TOP_K + DREAM_RANDOM) / accumulation) if dreaming else 0
        if main + len(batch.pairs) + dream_steps == 0:
            raise ValueError("a night trains at least one row")
        run = _Run(model, plan, params, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(main)),
                   teacher, clock, rng, fresh, {term: [] for term in (*WEIGHTS, *((SDFT_TERM,) if batch.sdft else ()))},
                   [], soft=soft, timer=timer,
                   warmup_updates=int(soft.warmup_share * main) if soft else 0)
        run.out_of_time = bool(trials.get("out_of_time"))  # the deadline cut the trials before the main block
        probe = seeds[0] if seeds else None
        teacher_before, kl_before = _probe(model, probe, teacher, soft)
        with clock("sdft_held") if batch.sdft else nullcontext():  # sampled once, at the night's start
            sdft_samples = (sdft_held(model, [batch.sdft[i] for i in held_sdft], plan, soft)
                            if batch.sdft and not run.late() else [])
            sdft_before = sdft_kl_by_kind(model, sdft_samples, teacher, soft) if sdft_samples else None
        model.train()
        stretched = _stretched(run, items, accumulation, timer) if plan.stretch_s else None
        for chunk in [] if stretched else [c for _ in range(epochs) for c in _chunks(items, accumulation)]:
            if run.late():
                break
            run.step(chunk)
        dpo = _dpo_block(run, batch.pairs)
        dream = _dream_block(run, batch, accumulation) if dreaming and not run.late() else None
    finally:
        for hook in hooks:
            hook.remove()
    teacher_after, kl_after = _probe(model, probe, teacher, soft)
    sdft_after = sdft_kl_by_kind(model, sdft_samples, teacher, soft) if sdft_samples else None
    drift = None if probe is None else float((teacher_after - teacher_before).abs().max())
    moved = max((float((now - was).abs().max()) for now, was in zip(leading(named, plan.long_term_rank), held,
                                                                     strict=True)), default=0.0) if held else None
    rows = {"sft": len(batch.sft), "dpo": len(batch.pairs), "seed": len(seeds), "imitate": imitating,
            "dream": dream["kept"] if dream else 0}
    minutes = {part: round(seconds / 60.0, 4) for part, seconds in
               {**dict.fromkeys(WEIGHTS, 0.0), **run.clock.seconds}.items()}
    return {
        "weights": dict(WEIGHTS), "steps": run.steps, "accumulation": accumulation, "epochs": epochs, "rows": rows,
        "loss": {term: (sum(v) / len(v) if v else None) for term, v in run.losses.items()}, "minutes": minutes,
        "on_policy_samples": run.sampled, "imitation_reward": sum(run.rewards) / len(run.rewards) if run.rewards else None,
        "long_term_rank": plan.long_term_rank, "dpo": dpo, "dreams": dream,
        "grads_adapter_only": run.grads_checked, "teacher_drift": drift, "long_term_drift": moved, "seed_kl_before": kl_before,
        "seed_kl_after": kl_after, "out_of_time": run.out_of_time, **({"stretch": stretched} if stretched else {}),
        **({"projector_only_updates": run.projector_only} if soft else {}),
        **({"dream_trials": trials} if trials else {}),
        **({SDFT_TERM: _sdft_receipt(batch.sdft, held_sdft, trained_sdft, run, len(sdft_samples), sdft_before,
                                     sdft_after)}
           if batch.sdft else {}),
    }


def _sdft_receipt(rows: Sequence[SdftSeq], held: Sequence[int], trained: Sequence[int], run: _Run, sampled: int,
                  before: Mapping[str, float | None] | None, after: Mapping[str, float | None] | None) -> dict[str, Any]:
    """Self-distillation's part of the receipt: rows distilled by kind, the on-policy samples drawn in
    training, the visits the budget or the deadline skipped, and ``seeded_not_distilled``: the SDFT-held
    rows by kind (their seeding rows trained, so this is no held-out reading), their samples, and the
    seeding KL on them before and after training, over all and by kind. Counts only."""
    def by_kind(indices: Sequence[int]) -> dict[str, int]:
        return {kind: sum(rows[i].kind == kind for i in indices) for kind in SDFT_KINDS}

    reading = {"label": SDFT_HELD_LABEL, "rows": by_kind(held), "samples": sampled,
               "kl_before": None if before is None else before["all"], "kl_after": None if after is None else after["all"],
               "kl_by_kind": {kind: {"before": None if before is None else before[kind],
                                     "after": None if after is None else after[kind]} for kind in SDFT_KINDS}}
    return {"weight": SDFT_WEIGHT, "rows": by_kind(trained), "on_policy_samples": run.sdft_sampled,
            "skipped_late": run.sdft_late, "budget_minutes": SDFT_BUDGET_S / 60.0, "seeded_not_distilled": reading}
