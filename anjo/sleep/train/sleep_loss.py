"""``sft_dpo_seed``: the sleep paradigm's one loss, in the night's one training job (2026-10-29).

After Behrouz, Hashemi and Mirrokni, "Language Models Need Sleep" (arXiv 2606.03979). Glimmer is dense,
so the person's LoRA is the expansion slot: only its parameters ever train, and the base stays frozen.
One optimizer and one schedule minimise one weighted sum, with these weights (``WEIGHTS``):

- ``sft`` 1.0: the night's rows as today (her best replies and the critic's Spark Open corrections);
- ``dpo`` 0.5: the (best, worst) pairs, beta 0.1, against the same model with its adapter off. It
  shares the SFT rate (TRL's separate pass ran 40x lower), so it pulls at half weight;
- ``seed`` 1.0, knowledge seeding: per token, over the full vocabulary, the reverse KL from the
  student (her adapter on, the message alone) to the teacher (the same base with the adapter off and
  the consolidated day before the message), on her own replies. A share ``ON_POLICY_SHARE`` of the
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
  (the general gate's items, scored as the gate scores them, smoothed) holds (``ANCHOR_TOLERANCE``).
  Each dream carries its seed's kind (``Batch.dream_groups``: ``boundary``, a state at the edge of a
  decision, or ``random``, the control), and the receipt gives the filter's choices by kind: whether
  the boundary's dreams carry more gradient than a random thread's is the cheap test of targeting.

**The fold** (``fold``, ``fold_folder``): with ``long_term_rank`` set, the job trains a LoRA whose
first ranks hold the person's long-term adapter, frozen by gradient masks, and whose last ranks are
tonight's fresh slot (B at zero: the reset). After training, each module's product B A is cut to the
best rank-R approximation (QR, then an SVD of the small core) and written as a plain LoRA of rank R,
alpha 2R, which the server takes as any other. Scale is 2 before and after (alpha is always 2r).

It is standalone, as the training script is: the standard library at the top and torch inside the
functions. Modal's image carries it beside ``colab_train.py``, which loads it by its path. The host's
tests import it as a module and run every function on a tiny CPU model.
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

WEIGHTS: Final = {"sft": 1.0, "dpo": 0.5, "seed": 1.0, "imitate": 0.25, "dream": 0.5}
ON_POLICY_SHARE: Final = 0.25  # GKD's lambda: the seed rows scored on a fresh student sample
FRESH_TOKENS: Final = 192
IMITATION_ROWS: Final = 4
IMITATION_SAMPLES: Final = 2
IMITATION_TOKENS: Final = 48
IMITATION_Z0: Final = 0.5
PREFIX_SHARE: Final = (0.3, 0.7)
DREAM_TOP_K: Final = 6
DREAM_RANDOM: Final = 2
DREAM_RATE: Final = 0.5  # the dream block's share of the peak learning rate
ANCHOR_TOLERANCE: Final = 0.0  # a dream block is kept only if the anchor score doesn't fall
MAX_GRAD_NORM: Final = 1.0
WARMUP: Final = 0.03
MAX_ACCUMULATION: Final = 16
MAX_EPOCHS: Final = 4
TEMPERATURE: Final = 1.0
_DTYPES: Final = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}

Ids = tuple[int, ...]


@dataclass(frozen=True)
class Seq:
    prompt: Ids
    reply: Ids


@dataclass(frozen=True)
class PairSeq:
    prompt: Ids
    chosen: Ids
    rejected: Ids


@dataclass(frozen=True)
class SeedSeq:
    """The student's prompt (no context), the teacher's (the consolidated day, then the same message), the reply."""

    prompt: Ids
    teacher: Ids
    reply: Ids


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


@dataclass(frozen=True)
class Plan:
    lr: float
    min_steps: int
    beta: float
    seed: int
    eot: int
    pad: int
    long_term_rank: int = 0  # the fold: the leading ranks that hold the long-term adapter, frozen
    dreams: bool = False


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


def step_plan(n: int, min_steps: int) -> tuple[int, int, int]:
    """Accumulation, epochs and steps an epoch, by the SFT rule (``lora.plan_steps``)."""
    accumulation = max(1, min(MAX_ACCUMULATION, n // min_steps))
    per_epoch = math.ceil(n / accumulation)
    epochs = 1 if per_epoch >= min_steps else min(MAX_EPOCHS, math.ceil(min_steps / per_epoch))
    return accumulation, epochs, per_epoch


def schedule(main_steps: int) -> Callable[[int], float]:
    """The rate's factor: a 3% warmup, a cosine over the main steps, then the dream block's share."""
    warm = max(1, math.ceil(WARMUP * main_steps))

    def factor(step: int) -> float:
        if step >= main_steps:
            return DREAM_RATE
        if step < warm:
            return (step + 1) / (warm + 1)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, main_steps - warm)))

    return factor


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


def reply_logits(model: Any, prompt: Ids, reply: Ids) -> Any:
    """The logits that predict each token of ``reply`` after ``prompt``: only those positions are kept."""
    import torch

    ids = torch.tensor([[*prompt, *reply]], device=_device(model))
    return model(input_ids=ids, logits_to_keep=len(reply) + 1).logits[0, :-1].float()


def reply_logprobs(model: Any, prompt: Ids, reply: Ids) -> Any:
    import torch

    logp = torch.log_softmax(reply_logits(model, prompt, reply), dim=-1)
    return logp.gather(-1, torch.tensor(reply, device=logp.device).view(-1, 1)).squeeze(-1)


def sft_loss(model: Any, row: Seq) -> Any:
    return -reply_logprobs(model, row.prompt, row.reply).mean()


def dpo_loss(model: Any, pair: PairSeq, beta: float) -> Any:
    """DPO against the same model with its adapter off, as today's pass scores its reference."""
    import torch

    chosen = reply_logprobs(model, pair.prompt, pair.chosen).sum()
    rejected = reply_logprobs(model, pair.prompt, pair.rejected).sum()
    with torch.no_grad(), model.disable_adapter():
        reference = (reply_logprobs(model, pair.prompt, pair.chosen).sum()
                     - reply_logprobs(model, pair.prompt, pair.rejected).sum())
    return -torch.nn.functional.logsigmoid(beta * ((chosen - rejected) - reference))


def seed_kl(model: Any, row: SeedSeq) -> Any:
    """Knowledge seeding: reverse KL(student || teacher) a reply position, averaged. The teacher is
    the base with the adapter off and the day in context, scored without a gradient; the student is
    her adapter on the message alone."""
    import torch

    with torch.no_grad(), model.disable_adapter():
        teacher = torch.log_softmax(reply_logits(model, row.teacher, row.reply), dim=-1)
    student = torch.log_softmax(reply_logits(model, row.prompt, row.reply), dim=-1)
    return (student.exp() * (student - teacher)).sum(dim=-1).mean()


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


def sample(model: Any, prompt: Ids, max_new: int, plan: Plan) -> Ids:
    """One sample of her adapter (on) after ``prompt``, up to ``max_new`` tokens or her end of turn."""
    import torch

    ids = torch.tensor([list(prompt)], device=_device(model))
    with _evaluating(model), torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=TEMPERATURE,
                             top_p=1.0, top_k=0, max_new_tokens=max_new, eos_token_id=plan.eot, pad_token_id=plan.pad)
    return _until_eot([int(t) for t in out[0, len(prompt):].tolist()], plan.eot)


def imitation_loss(model: Any, row: SeedSeq, rng: random.Random, plan: Plan) -> tuple[Any, float | None]:
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
    samples = [sample(model, prompt, len(target), plan) for _ in range(IMITATION_SAMPLES)]
    rewards = [edit_reward(s, target) for s in samples]
    total, n = sum(rewards), len(rewards)
    advantages = [r - (total - r) / (n - 1) for r in rewards]
    mean = total / n
    used = [(s, a) for s, a in zip(samples, advantages, strict=True) if s and a != 0.0]
    if not used:
        return None, mean
    gain = torch.stack([a * reply_logprobs(model, prompt, s).mean() for s, a in used]).sum()
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


def grad_norm(model: Any, params: Sequence[Any], row: Seq) -> float:
    """The norm of one row's SFT gradient on the adapter: the dream filter's importance proxy."""
    for p in params:
        p.grad = None
    sft_loss(model, row).backward()
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
    clock: _Clock
    rng: random.Random
    fresh: set[int]
    losses: dict[str, list[float]]
    rewards: list[float]
    sampled: int = 0
    steps: int = 0

    def loss(self, term: str, row: Any) -> Any:
        if term == "dpo":
            return dpo_loss(self.model, row, self.plan.beta)
        if term == "seed":
            index, seq = row
            if index in self.fresh and (fresh := sample(self.model, seq.prompt, FRESH_TOKENS, self.plan)):
                seq, self.sampled = replace(seq, reply=fresh), self.sampled + 1
            return seed_kl(self.model, seq)
        if term == "imitate":
            loss, reward = imitation_loss(self.model, row, self.rng, self.plan)
            if reward is not None:
                self.rewards.append(reward)
            return loss
        return sft_loss(self.model, row)  # sft and dream

    def step(self, chunk: Sequence[tuple[str, Any]]) -> None:
        import torch

        for term, row in chunk:
            with self.clock(term):
                loss = self.loss(term, row)
                if loss is not None:
                    (WEIGHTS[term] * loss / len(chunk)).backward()
                    self.losses[term].append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(self.params, MAX_GRAD_NORM)
        self.optimizer.step()
        self.rate.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _stream(batch: Batch, rng: random.Random) -> tuple[list[tuple[str, Any]], int]:
    """The main block's rows in one seeded order, and how many seed rows imitate too."""
    long_enough = [seq for seq in batch.seeds if len(seq.reply) >= 2]
    imitate = rng.sample(long_enough, min(IMITATION_ROWS, len(long_enough)))
    items: list[tuple[str, Any]] = [
        *(("sft", row) for row in batch.sft), *(("dpo", pair) for pair in batch.pairs),
        *(("seed", (i, seq)) for i, seq in enumerate(batch.seeds)), *(("imitate", seq) for seq in imitate),
    ]
    rng.shuffle(items)
    return items, len(imitate)


def _dream_block(run: _Run, batch: Batch, accumulation: int) -> dict[str, Any]:
    """Rank the dreams by gradient norm, keep the top k plus b at random, train them, and keep the
    block only if the anchor score holds; else the adapter goes back to its state before the block."""
    import torch

    with run.clock("dream_rank"):
        norms = [grad_norm(run.model, run.params, row) for row in batch.dreams]
    kept = select_dreams(norms, DREAM_TOP_K, DREAM_RANDOM, run.plan.seed)
    with run.clock("anchor"):
        before = anchor_score(run.model, batch.anchors)
    saved = [p.detach().clone() for p in run.params]
    for chunk in _chunks([("dream", batch.dreams[i]) for i in kept], accumulation):
        run.step(chunk)
    with run.clock("anchor"):
        after = anchor_score(run.model, batch.anchors)
    held = after >= before - ANCHOR_TOLERANCE
    if not held:
        with torch.no_grad():
            for p, value in zip(run.params, saved, strict=True):
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


def train_terms(model: Any, batch: Batch, plan: Plan, *, timer: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Train the adapter on the one loss; returns the receipt: rows, mean loss and minutes per term,
    the on-policy samples taken, the imitation reward, and the dream block's outcome. Counts only."""
    import torch

    named = adapter_params(model)
    dreaming = plan.dreams and bool(batch.dreams)
    if dreaming and not batch.anchors:
        raise ValueError("dreams need the anchor items their check scores")
    if batch.dream_groups and len(batch.dream_groups) != len(batch.dreams):
        raise ValueError("each dream has one seed kind")
    rng = random.Random(plan.seed)
    torch.manual_seed(plan.seed)
    items, imitating = _stream(batch, rng)
    fresh = set(rng.sample(range(len(batch.seeds)), math.ceil(ON_POLICY_SHARE * len(batch.seeds))))
    accumulation, epochs, per_epoch = step_plan(len(items), plan.min_steps) if items else (1, 0, 0)
    main = epochs * per_epoch
    dream_steps = math.ceil(min(len(batch.dreams), DREAM_TOP_K + DREAM_RANDOM) / accumulation) if dreaming else 0
    if main + dream_steps == 0:
        raise ValueError("a night trains at least one row")
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=plan.lr, weight_decay=0.0)  # no decay: a masked rank stays put
    run = _Run(model, plan, params, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(main)),
               _Clock(timer), rng, fresh, {term: [] for term in WEIGHTS}, [])
    hooks = freeze_leading(named, plan.long_term_rank) if plan.long_term_rank else []
    model.train()
    try:
        for _ in range(epochs):
            for chunk in _chunks(items, accumulation):
                run.step(chunk)
        dream = _dream_block(run, batch, accumulation) if dreaming else None
    finally:
        for hook in hooks:
            hook.remove()
    rows = {"sft": len(batch.sft), "dpo": len(batch.pairs), "seed": len(batch.seeds), "imitate": imitating,
            "dream": dream["kept"] if dream else 0}
    minutes = {part: round(seconds / 60.0, 4) for part, seconds in
               {**dict.fromkeys(WEIGHTS, 0.0), **run.clock.seconds}.items()}
    return {
        "weights": dict(WEIGHTS), "steps": run.steps, "accumulation": accumulation, "epochs": epochs, "rows": rows,
        "loss": {term: (sum(v) / len(v) if v else None) for term, v in run.losses.items()}, "minutes": minutes,
        "on_policy_samples": run.sampled, "imitation_reward": sum(run.rewards) / len(run.rewards) if run.rewards else None,
        "long_term_rank": plan.long_term_rank, "dreams": dream,
    }


# ── the fold, and the adapter file ───────────────────────────────────────────


def fold(tensors: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """Each LoRA pair's product cut to its best rank ``min(rank, k)`` approximation, as a plain pair."""
    import torch

    a_names = sorted(name for name in tensors if name.endswith("lora_A.weight"))
    if not a_names or len(tensors) != 2 * len(a_names):
        raise ValueError("a plain LoRA holds lora_A and lora_B pairs and nothing else")
    out: dict[str, Any] = {}
    for a_name in a_names:
        b_name = a_name[: -len("lora_A.weight")] + "lora_B.weight"
        a, b = tensors[a_name].double(), tensors[b_name].double()  # (k, d_in) and (d_out, k)
        keep = min(rank, a.shape[0])
        qb, rb = torch.linalg.qr(b)
        qa, ra = torch.linalg.qr(a.T)
        u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
        m = min(keep, s.numel())
        root = s[:m].sqrt()
        new_b = torch.zeros(b.shape[0], keep, dtype=b.dtype)
        new_a = torch.zeros(keep, a.shape[1], dtype=a.dtype)
        new_b[:, :m] = (qb @ u[:, :m]) * root
        new_a[:m] = root[:, None] * (vh[:m] @ qa.T)
        dtype = tensors[a_name].dtype
        out[a_name], out[b_name] = new_a.to(dtype).contiguous(), new_b.to(dtype).contiguous()
    return out


def read_safetensors(path: Path) -> dict[str, Any]:
    import torch

    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + size])
    data = raw[8 + size:]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] not in _DTYPES:
            raise ValueError(f"{name} is {meta['dtype']}, not a float type an adapter is saved in")
        begin, end = meta["data_offsets"]
        flat = torch.frombuffer(bytearray(data[begin:end]), dtype=getattr(torch, _DTYPES[meta["dtype"]]))
        out[name] = flat.reshape(meta["shape"]).clone()
    return out


def write_safetensors(tensors: Mapping[str, Any], path: Path) -> None:
    import torch

    codes = {name: code for code, name in _DTYPES.items()}
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    blobs, offset = [], 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        blob = tensor.view(-1).view(torch.uint8).numpy().tobytes()  # bfloat16 has no numpy type
        header[name] = {"dtype": codes[str(tensor.dtype).removeprefix("torch.")], "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def fold_folder(folder: Path, rank: int) -> int:
    """Fold the adapter saved in ``folder`` to rank ``min(rank, r)`` in place, as a plain LoRA with
    alpha 2r: the two files, nothing else. Returns the rank it holds."""
    config_path, weights_path = folder / "adapter_config.json", folder / "adapter_model.safetensors"
    meta = json.loads(config_path.read_text(encoding="utf-8"))
    r = meta.get("r")
    if (meta.get("peft_type") != "LORA" or meta.get("use_dora") or meta.get("rank_pattern")
            or meta.get("alpha_pattern") or not isinstance(r, int) or meta.get("lora_alpha") != 2 * r):
        raise ValueError("the fold takes a plain LoRA with alpha 2r")
    tensors = read_safetensors(weights_path)
    if any(t.shape[0 if name.endswith("lora_A.weight") else 1] != r for name, t in tensors.items()):
        raise ValueError("the adapter's tensors don't hold the rank its config names")
    folded, kept = fold(tensors, rank), min(rank, r)
    pending = folder / f".{weights_path.name}.tmp"
    write_safetensors(folded, pending)
    os.replace(pending, weights_path)
    meta.update({"r": kept, "lora_alpha": 2 * kept, "rank_pattern": {}, "alpha_pattern": {}})
    config_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return kept
