"""The thought channel's half of the night's training job: the person's LoRA and the projector that
lets Glimmer read MIND's state, trained together in one run (one system: ``brief_one_system``).

**What it trains.** A night's rows (``train.dataset``) may carry the core state each was made under
(``core``, a ``night.core_snapshot.CoreSnapshot``: the projector's input, never the raw state). With
a recipe ``channel`` block, the training script (``colab_train.main``) hands the run to ``main`` here
instead of TRL: each row's prompt goes to Glimmer as input embeddings, with K soft tokens, the
projector's output for the row's reading, spliced in just before the ``<|eot|>`` that closes the
latest user turn, exactly as the server splices them (``serving.soft_tokens.splice``). The loss is
on the reply and ``<|eot|>`` only, as in the SFT pass; gradients reach the LoRA through the frozen
4-bit base and the projector through the soft tokens. For the first ``projector_warmup`` share of the
updates only the projector learns (the LoRA held still), so a fresh projector learns to speak before
the adapter learns around it. Then, for ``sft_dpo``, a DPO pass on the pair rows, with soft tokens on
both sides; its reference is the model with the adapter switched off and the projector as the run
began (as TRL's ``adapter_off`` reference, plus the channel).

**Contrast rows** (the shared pretraining across generated lives, ``mind.core.reader_data``): a row
with ``contrast`` also carries another moment's state (``contrast.core``) and the reply that state
earns (``contrast.plain``). The run trains reply-given-state on both, and adds a state-conditional
preference in both directions at once: with the row's own state, the raise reply should gain on the
plain one more than with the other state (``-log sigmoid(beta * margin)``, the margin a difference
of differences). The message is the same on all four sequences, so only the soft tokens can move the
margin: a margin near zero means the channel carries nothing. ``val_share`` of the contrast rows are
held out by their ``group`` (the generated life; the row's id without one), so siblings never sit on
both sides, and scored after training with their own state and with the empty reading.

**The pieces compose** (one loss, one budget, one job): ``Channel`` puts the projector on a model
(``soft`` for a reading, ``splice`` and ``batch`` for a sequence as the server sends it, ``logps`` for
the replies' log-probabilities, ``projector_groups`` for an optimizer); ``sft_loss`` and
``contrast_loss`` are the channel's terms; ``train_pass`` is this run's loop over them. Another loss
adds its own terms over ``Channel.logps`` and puts ``projector_groups`` in its optimizer.

**The projector's math** is ``serving.projector_math`` (the host serves it, so it lives there): this
module re-exports it, imported, or loaded by path beside this file's copy on the image (the training
script runs isolated, with no ``anjo`` on its path). numpy, torch, transformers and peft load inside
the functions that need them. The run's width and dtype are the model's own, so the CPU smoke run's
tiny Glimmer trains through the same code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

_MATH: Final = ("ANCHOR_WORDS", "HIDDEN", "MAX_K", "PARAMS", "PROJECTOR_FILE", "PROJECTOR_VERSION", "RMS_EPS",
                "SCALARS", "decode_reading", "features", "forward", "fresh_params", "load_params", "params_sha", "rms",
                "save_params")  # serving.projector_math's names, re-exported here

try:
    from anjo.sleep.serving.projector_math import (
        ANCHOR_WORDS as ANCHOR_WORDS,
        HIDDEN as HIDDEN,
        MAX_K as MAX_K,
        PARAMS as PARAMS,
        PROJECTOR_FILE as PROJECTOR_FILE,
        PROJECTOR_VERSION as PROJECTOR_VERSION,
        RMS_EPS as RMS_EPS,
        SCALARS as SCALARS,
        decode_reading as decode_reading,
        features as features,
        forward as forward,
        fresh_params as fresh_params,
        load_params as load_params,
        params_sha as params_sha,
        rms as rms,
        save_params as save_params,
    )
except ImportError:  # pragma: no cover - the image's isolated training script: loaded by path
    _SPEC = importlib.util.spec_from_file_location(
        "anjo_projector_math", Path(__file__).resolve().parents[1] / "serving" / "projector_math.py")
    if _SPEC is None or _SPEC.loader is None:
        raise
    _LOADED = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_LOADED)
    globals().update({name: getattr(_LOADED, name) for name in _MATH})

EOT: Final = "<|eot|>"
CHANNEL_KEYS: Final = ("k", "d", "width", "arm", "projector_lr", "anchor_lr", "contrast_beta",
                       "contrast_weight", "val_share", "projector_warmup", "max_minutes")
ARMS: Final = ("state", "zero")
# The recipe's channel settings that aren't the projector's shape, its arm or the run's minutes.
DEFAULTS: Final = {"projector_lr": 3e-4, "anchor_lr": 1e-5, "contrast_beta": 0.1, "contrast_weight": 1.0,
                   "val_share": 0.1, "projector_warmup": 0.1}
LOAD_MINUTES: Final = 12.0  # Glimmer's 4-bit load from the weights volume, and the run's checks, about
SAVE_MINUTES: Final = 5.0  # the held-out margins, the adapter and the projector written
LAYERS: Final = ("w1", "b1", "u1", "c1", "w2", "b2")  # the projector's MLP
ANCHORS: Final = ("anchors", "speak", "silence")  # its anchor rows, at their own (small) rate
SCALING: Final = ("x_mean", "x_std", "g_mean", "g_std")  # set once from the readings, never trained
Reading = tuple[Any, Any, Any, float]  # slot_x (K-1, F), gates (K-1,), global_x (2d,), speak


def check_channel(block: object) -> dict[str, Any]:
    """A recipe's ``channel`` block, checked: K tokens (2 to ``MAX_K``), the core's width, the
    projector's width, the arm (``state``, or ``zero``: the ablation, every reading empty), the
    projector's and anchors' learning rates, the contrast's beta and weight, the held-out share, the
    projector-only warmup's share and the minutes the loop may train."""
    if not isinstance(block, dict) or sorted(block) != sorted(CHANNEL_KEYS):
        raise ValueError(f"the channel block holds exactly {', '.join(CHANNEL_KEYS)}")
    for key, low, high in (("k", 2, MAX_K), ("d", 1, 4096), ("width", 8, 4096)):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"channel {key} is an integer from {low} to {high}")
    if block["arm"] not in ARMS:
        raise ValueError(f"channel arm is one of {', '.join(ARMS)}")
    for key, top in (("projector_lr", 1e-2), ("anchor_lr", 1e-2), ("contrast_beta", 10.0),
                     ("contrast_weight", 10.0), ("max_minutes", 600.0)):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= top:
            raise ValueError(f"channel {key} is a number from 0 to {top:g}")
    if not 0 <= block["val_share"] < 1 or not 0 <= block["projector_warmup"] < 1:
        raise ValueError("channel val_share and projector_warmup are in [0, 1)")
    return dict(block)


def block(*, k: int, d: int, width: int, arm: str, exec_s: float, **overrides: float) -> dict[str, Any]:
    """A checked channel block: the projector's shape and arm, ``DEFAULTS`` (or ``overrides``), and the
    minutes the loop may train inside a script that may run ``exec_s`` seconds."""
    minutes = max(1.0, exec_s / 60.0 - LOAD_MINUTES - SAVE_MINUTES)
    return check_channel({**DEFAULTS, **overrides, "k": k, "d": d, "width": width, "arm": arm, "max_minutes": minutes})


# ── The pieces a training loop composes ─────────────────────────────────────


def _embedding_weight(embed: Any) -> Any:
    """The token table under Glimmer's input embedding (a table and a weightless norm)."""
    weight = getattr(embed, "weight", None)
    if weight is not None:
        return weight
    found = [m.weight for m in embed.modules() if type(m).__name__ == "Embedding"]
    if len(found) != 1:
        raise ValueError("the model's input embedding holds no single token table")
    return found[0]


class Channel:
    """The projector on a model: its arrays as torch tensors (the MLP and anchors trainable, the
    feature scaling fixed), the soft tokens for a reading, and sequences spliced and scored as the
    server sends them. Its width and dtype are the model's input embedding's."""

    def __init__(self, model: Any, params: Mapping[str, Any], *, device: Any, eot: int,
                 ref_params: Mapping[str, Any] | None = None) -> None:
        import torch  # type: ignore[import-not-found, unused-ignore]

        self.torch, self.model, self.device, self.eot = torch, model, device, eot
        self.embed = model.get_input_embeddings()
        weight = _embedding_weight(self.embed)
        self.hidden, self.dtype = int(weight.shape[1]), weight.dtype
        self.params = {n: torch.tensor(v, device=device, requires_grad=n not in SCALING) for n, v in params.items()}
        self.ref = {n: torch.tensor(v, device=device) for n, v in (ref_params or params).items()}
        self.ones = torch.ones(self.hidden, 1, device=device)

    def projector_groups(self, *, projector_lr: float, anchor_lr: float) -> list[dict[str, Any]]:
        """The projector's trainable tensors as optimizer groups: the MLP, then the anchors."""
        return [{"params": [self.params[n] for n in LAYERS], "lr": projector_lr},
                {"params": [self.params[n] for n in ANCHORS], "lr": anchor_lr}]

    def arrays(self) -> dict[str, Any]:
        """The projector's arrays now, as numpy copies (for ``save_params``): a CPU tensor's ``numpy``
        shares its memory, and the next update would change them."""
        return {n: v.detach().cpu().numpy().copy() for n, v in self.params.items()}

    def soft(self, reading: Reading, ref: bool = False) -> Any:
        """The K soft tokens ``(K, H)`` for ``reading``, from the projector now or as it began."""
        torch = self.torch
        slot_x, gates, global_x = (torch.as_tensor(a, device=self.device)[None] for a in reading[:3])
        speak = torch.tensor([reading[3]], device=self.device)
        p = self.ref if ref else self.params
        first, rest = forward(p, slot_x, gates, global_x, speak, tanh=torch.tanh, sqrt=torch.sqrt, ones=self.ones)
        return torch.cat([first[:, None], rest], dim=1)[0]

    def splice(self, prompt: Sequence[int], reply: Sequence[int], soft: Any) -> tuple[Any, list[int]]:
        """One sequence's rows ``(T, H)`` in float32, with ``soft`` just before the prompt's last
        ``<|eot|>`` (as ``serving.soft_tokens.splice``), and its labels: the reply's ids only. The rows
        are the model's own input embedding, its token table through its weightless norm, normalized
        once, as the host's are (``soft_tokens.normalize``); ``check_embeds`` holds the model to it."""
        torch = self.torch
        rows = self.embed(torch.tensor([*prompt, *reply], device=self.device)).float()
        cut = len(prompt) - 1 - list(prompt)[::-1].index(self.eot)
        parts = [rows[:cut], *([soft.float()] if soft is not None else []), rows[cut:]]
        return torch.cat(parts), [-100] * (len(prompt) + (0 if soft is None else soft.shape[0])) + list(reply)

    def batch(self, seqs: Sequence[tuple[Sequence[int], Sequence[int], Any]]) -> tuple[Any, Any, Any, int]:
        """The sequences padded into the model's input: embeddings in its own dtype, the attention
        mask, the labels, and the first position whose logits any label needs."""
        torch = self.torch
        spliced = [self.splice(prompt, reply, soft) for prompt, reply, soft in seqs]
        width = max(len(labels) for _, labels in spliced)
        x = torch.zeros(len(spliced), width, self.hidden, device=self.device)
        mask = torch.zeros(len(spliced), width, dtype=torch.long, device=self.device)
        lab = torch.full((len(spliced), width), -100, dtype=torch.long, device=self.device)
        for i, (rows, labels) in enumerate(spliced):
            x[i, :len(labels)], mask[i, :len(labels)] = rows, 1
            lab[i, :len(labels)] = torch.tensor(labels, device=self.device)
        first = min(next(n for n, t in enumerate(labels) if t != -100) for _, labels in spliced)
        return x.to(self.dtype), mask, lab, first - 1

    def logps(self, seqs: Sequence[tuple[Sequence[int], Sequence[int], Any]], grad: bool = True) -> Any:
        """Each sequence's summed log-probability of its reply, soft tokens spliced in. Only the
        positions a label needs reach the head (``logits_to_keep``)."""
        torch = self.torch
        x, mask, lab, start = self.batch(seqs)
        if grad and not x.requires_grad and torch.is_grad_enabled():
            x.requires_grad_(True)
        keep = x.shape[1] - start
        logits = self.model(inputs_embeds=x, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
        target = lab[:, start + 1:]
        picked = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, target.clamp(min=0)[..., None])[..., 0]
        return (picked * (target != -100)).sum(-1)


def sft_loss(ch: Channel, prompt: Sequence[int], reply: Sequence[int], own: Any) -> Any:
    """Reply-given-state: the reply's mean negative log-likelihood with ``own``'s soft tokens."""
    return -ch.logps([(prompt, reply, own)])[0] / len(reply)


def contrast_loss(ch: Channel, prompt: Sequence[int], raised: Sequence[int], plain: Sequence[int], own: Any,
                  other: Any, *, beta: float, weight: float) -> tuple[Any, float]:
    """The contrast row's loss and margin: the raise reply given its own state and the plain reply
    given the other, and the state-conditional preference on ``(lp[raise|own] - lp[plain|own]) -
    (lp[raise|other] - lp[plain|other])``, which pushes both states' sides at once."""
    torch = ch.torch
    lp = ch.logps([(prompt, raised, own), (prompt, plain, own), (prompt, raised, other), (prompt, plain, other)])
    margin = (lp[0] - lp[1]) - (lp[2] - lp[3])
    nll = -lp[0] / len(raised) - lp[3] / len(plain)
    return nll - weight * torch.nn.functional.logsigmoid(beta * margin), float(margin.detach())


def item_loss(ch: Channel, item: Mapping[str, Any], readings: Mapping[str, Reading], *, beta: float,
              weight: float) -> tuple[Any, float | None]:
    """One prepared row's loss (and its margin, for a contrast row)."""
    own = None if item["own"] is None else ch.soft(readings[item["own"]])
    if "plain" not in item:
        return sft_loss(ch, item["prompt"], item["reply"], own), None
    return contrast_loss(ch, item["prompt"], item["reply"], item["plain"], own, ch.soft(readings[item["other"]]),
                         beta=beta, weight=weight)


def train_pass(
    ch: Channel, order: list[dict[str, Any]], readings: Mapping[str, Reading], optimizer: Any, schedule: Any, *,
    lora: Sequence[Any], epochs: int, accumulation: int, beta: float, weight: float, warmup_updates: int,
    deadline: float, clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The channel's loop: ``epochs`` shuffled passes over ``order``, an update every ``accumulation``
    rows. For the first ``warmup_updates`` updates the ``lora`` tensors' gradients are dropped before
    the step, so only the projector learns. It stops at ``deadline``. Returns its counts."""
    steps, seen, losses, margins = 0, 0, list[float](), list[float]()
    params = [p for g in optimizer.param_groups for p in g["params"]]
    for _ in range(epochs):
        random.shuffle(order)
        for n, item in enumerate(order):
            if clock() > deadline:
                break
            loss, margin = item_loss(ch, item, readings, beta=beta, weight=weight)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            if margin is not None:
                margins.append(margin)
            seen += 1
            if (n + 1) % accumulation == 0 or n + 1 == len(order):
                if steps < warmup_updates:
                    for p in lora:
                        p.grad = None  # the LoRA held still: AdamW skips a tensor with no gradient
                ch.torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
    return {"steps": steps, "seen": seen, "losses": losses, "margins": margins}


def check_embeds(ch: Channel, ids: Sequence[int]) -> float:
    """Glimmer on token ids and on the rows ``Channel.splice`` feeds (its input embedding's output)
    as ``inputs_embeds`` must agree: those rows are post-norm, as vLLM's prompt embeddings are, and the
    model's ``inputs_embeds`` path adds no norm of its own. Returns the largest gap."""
    torch = ch.torch
    held = torch.tensor([list(ids)], device=ch.device)
    with torch.no_grad():
        by_ids = ch.model(input_ids=held, use_cache=False).logits.float()
        rows = ch.embed(held).float().to(ch.dtype)
        by_rows = ch.model(inputs_embeds=rows, use_cache=False).logits.float()
    gap = float((by_ids - by_rows).abs().max())
    if gap > 0.5:
        raise ValueError(f"Glimmer reads normalized rows differently from its ids (gap {gap:.3f}); the channel "
                         "would train on embeddings the server never sends")
    return gap


def check_forward(ch: Channel, readings: Sequence[Reading]) -> float:
    """The projector in torch here against the same code on numpy, as the host serves it."""
    import numpy as np

    p = ch.arrays()
    worst = 0.0
    for reading in readings[:4]:
        first, rest = forward(p, reading[0][None], reading[1][None], reading[2][None], np.array([reading[3]], np.float32),
                              tanh=np.tanh, sqrt=np.sqrt, ones=np.ones((ch.hidden, 1), np.float32))
        here = ch.soft(reading).detach().cpu().numpy()
        worst = max(worst, float(np.abs(np.concatenate([first, rest[0]]) - here).max()))
    if worst > 1e-3:
        raise ValueError(f"the projector's torch and numpy outputs differ by {worst:.2e}")
    return worst


# ── The run's inputs ──────────────────────────────────────────────────────────


def _row_reading(row: Mapping[str, Any], key: str = "core") -> Mapping[str, Any] | None:
    core = row.get(key)
    return None if core is None else core["reading"]


def held_out(row: Mapping[str, Any], share: float) -> bool:
    """Whether ``row`` is held out: by its ``group`` (a generated life), else its id."""
    key = str(row.get("group") or row["id"])
    return int(hashlib.sha256(f"val:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def readings_of(rows: Sequence[Mapping[str, Any]], arm: str, k: int, d: int) -> dict[str, Reading]:
    """Every reading the run's rows carry, by its row and role; with the ``zero`` arm every one is
    the empty reading (the zero-tokens ablation, on the same rows)."""
    import numpy as np

    empty: Reading = (np.zeros((k - 1, features(d)), np.float32), np.zeros(k - 1, np.float32),
                      np.zeros(2 * d, np.float32), 0.0)
    found: dict[str, Reading] = {}
    for row in rows:
        for role, reading in (("own", _row_reading(row)), ("other", _row_reading(row.get("contrast") or {}))):
            if reading is None:
                continue
            decoded = decode_reading(reading)
            if decoded[0].shape != empty[0].shape:
                raise ValueError(f"row {row.get('id')}'s reading is for another K or core width")
            found[f"{row['id']}/{role}"] = empty if arm == "zero" else decoded
    found["empty"] = empty
    return found


def anchor_rows(tokenizer: Any, embed: Any, k: int) -> Any:
    """Each anchor word's rows, averaged, then normalized: speak, silence, then each rank's."""
    import numpy as np
    import torch  # type: ignore[import-not-found, unused-ignore]

    rows = []
    weight = _embedding_weight(embed)
    with torch.no_grad():
        for word in ANCHOR_WORDS[: k + 1]:
            ids = torch.tensor(tokenizer(word, add_special_tokens=False)["input_ids"], device=weight.device)
            rows.append(embed(ids).float().mean(0).cpu().numpy())
    raw = np.stack(rows)
    return (raw / np.sqrt((raw * raw).mean(-1, keepdims=True) + RMS_EPS)).astype(np.float32)


def feature_stats(params: dict[str, Any], readings: Sequence[Reading]) -> dict[str, Any]:
    """A fresh projector with its feature scaling from the run's own readings (routed slots only)."""
    import numpy as np

    slots = [r[0][r[1] > 0] for r in readings]
    stacked = np.concatenate([s for s in slots if len(s)] or [np.zeros((1, params["x_mean"].size), np.float32)])
    whole = np.stack([r[2] for r in readings]) if readings else np.zeros((1, params["g_mean"].size), np.float32)
    return {**params, "x_mean": stacked.mean(0), "x_std": stacked.std(0) + 1e-3,
            "g_mean": whole.mean(0), "g_std": whole.std(0) + 1e-3}


# ── The run (GPU VM only) ─────────────────────────────────────────────────────


def _encode(ct: Any, tokenizer: Any, row: dict[str, Any], side: str, template: dict[str, str],
            reply: str | None = None) -> tuple[list[int], list[int]]:  # pragma: no cover - needs the training stack
    """A row's prompt ids (as the server renders them, BOS first, the reply's header last) and one
    side's reply ids (closed by ``<|eot|>``), checked against the chat template's own render."""
    shaped = {**row, side: [{"role": "assistant", "content": reply}]} if reply is not None else row
    plain = ct.plain_row(tokenizer, {k: v for k, v in shaped.items() if k in ("id", "prompt", side, "current_date")},
                         template)
    prompt_ids = list(tokenizer(text=plain["prompt"])["input_ids"])
    reply_ids = [*tokenizer(ct._reply(shaped[side]), add_special_tokens=False)["input_ids"],
                 tokenizer.convert_tokens_to_ids(EOT)]
    ct._check_tokens(tokenizer, shaped, side, prompt_ids, reply_ids, template)
    return prompt_ids, reply_ids


def _items(ct: Any, tokenizer: Any, cfg: dict[str, Any], ch: dict[str, Any], sft_rows: list[dict[str, Any]],
           pair_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:  # pragma: no cover - training stack
    """The rows as token ids: SFT, contrast, held-out contrast and pair items; a row too long is dropped."""
    template, limit = ct.template_kwargs(cfg), cfg["max_seq_length"] - ch["k"]
    found: dict[str, list[dict[str, Any]]] = {"sft": [], "contrast": [], "val": [], "pairs": []}
    for row in sft_rows:
        prompt, reply = _encode(ct, tokenizer, row, "completion", template)
        item: dict[str, Any] = {"id": row["id"], "prompt": prompt, "reply": reply,
                                "own": f"{row['id']}/own" if row.get("core") else None}
        if row.get("contrast"):
            _, plain = _encode(ct, tokenizer, row, "completion", template, reply=row["contrast"]["plain"])
            item.update(plain=plain, other=f"{row['id']}/other" if row["contrast"].get("core") else "empty")
        if len(prompt) + max(len(reply), len(item.get("plain", []))) > limit:
            continue
        kind = "sft" if "plain" not in item else "val" if held_out(row, ch["val_share"]) else "contrast"
        found[kind].append(item)
    for row in pair_rows:
        prompt, chosen = _encode(ct, tokenizer, row, "chosen", template)
        _, rejected = _encode(ct, tokenizer, row, "rejected", template)
        if len(prompt) + max(len(chosen), len(rejected)) <= limit:
            found["pairs"].append({"id": row["id"], "prompt": prompt, "chosen": chosen, "rejected": rejected,
                                   "own": f"{row['id']}/own" if row.get("core") else None})
    return found


def main(ct: Any, cfg: dict[str, Any]) -> int:  # pragma: no cover - runs only on the GPU VM
    """The channel run: ``ct`` is the training script's module (``colab_train``), ``cfg`` its recipe."""
    import numpy as np
    import torch  # type: ignore[import-not-found, unused-ignore]
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoTokenizer, set_seed  # type: ignore[import-not-found, unused-ignore]

    ch = check_channel(cfg["channel"])
    smoke, base = ct._smoke_model(), ct.base_model(os.environ)
    out, prev = Path(os.environ["ANJO_TRAIN_OUT"]), os.environ.get("ANJO_TRAIN_PREV") or None
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    set_seed(cfg["seed"])
    started_all = time.monotonic()
    local = smoke or base
    tokenizer = AutoTokenizer.from_pretrained(local or ct.MODEL, revision=None if local else cfg["revision"])
    ct.check_template(tokenizer, cfg)
    sft_rows, pair_rows = ct.split_rows(ct._rows(Path(os.environ["ANJO_TRAIN_DATASET"])))
    if cfg["objective"] != "sft_dpo":
        pair_rows = []
    readings = readings_of([*sft_rows, *pair_rows], ch["arm"], ch["k"], ch["d"])
    items = _items(ct, tokenizer, cfg, ch, sft_rows, pair_rows)
    ct.check_every_row_kept(cfg, len(sft_rows), len(items["sft"]) + len(items["contrast"]) + len(items["val"]))
    settings = ct.build_settings(cfg)
    model = ct._load_model(cfg, settings, smoke, base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if prev and (Path(prev) / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, prev, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(**settings["lora"]))
    device = torch.device("cpu" if smoke else "cuda:0")
    hidden = int(_embedding_weight(model.get_input_embeddings()).shape[1])
    projector_prev = Path(prev) / PROJECTOR_FILE if prev else None
    if projector_prev is not None and projector_prev.is_file():
        params, meta = load_params(projector_prev.read_bytes())
        shape = (meta.get("k"), meta.get("d"), meta.get("width"), meta.get("hidden", HIDDEN))  # older saves: Glimmer's
        if shape != (ch["k"], ch["d"], ch["width"], hidden):
            raise ValueError("the warm start's projector has another K, core width, projector width or model width")
        fresh = False
    else:
        rows = anchor_rows(tokenizer, model.get_input_embeddings(), ch["k"])
        params = feature_stats(fresh_params(ch["k"], ch["d"], ch["width"], rows, cfg["seed"], hidden=hidden),
                               [v for key, v in readings.items() if key != "empty"])
        fresh = True
    start_sha = params_sha(params)
    run = Channel(model, params, device=device, eot=tokenizer.convert_tokens_to_ids(EOT))
    embeds_gap = check_embeds(run, tokenizer("The camera arrives on Friday.")["input_ids"])
    forward_gap = check_forward(run, [v for key, v in readings.items() if key != "empty"] or [readings["empty"]])
    lora = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": cfg["learning_rate"]},
                                   *run.projector_groups(projector_lr=ch["projector_lr"], anchor_lr=ch["anchor_lr"])],
                                  weight_decay=0.0)
    order = [*items["sft"], *items["contrast"]]
    planned = max(1, cfg["epochs"] * math.ceil(len(order) / cfg["accumulation"]))
    warm = max(1, int(ct.WARMUP * planned))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, planned) / planned)))
    deadline = started_all + 60.0 * ch["max_minutes"]
    model.train()
    done = train_pass(run, order, readings, optimizer, schedule, lora=lora, epochs=cfg["epochs"],
                      accumulation=cfg["accumulation"], beta=ch["contrast_beta"], weight=ch["contrast_weight"],
                      warmup_updates=int(ch["projector_warmup"] * planned), deadline=deadline)
    dpo_steps = _dpo(run, items["pairs"], readings, cfg, deadline) if items["pairs"] and time.monotonic() < deadline else 0
    model.eval()
    held = _validate(run, items["val"], readings) if items["val"] else {}
    model.save_pretrained(str(out))
    ct.record_rendering(out, cfg["rendering"])
    if base is not None:
        ct.record_base_model(out)
    final = run.arrays()
    meta = {"k": ch["k"], "d": ch["d"], "width": ch["width"], "arm": ch["arm"], "trained_from": start_sha}
    (out / PROJECTOR_FILE).write_bytes(save_params(final, meta))
    losses, margins = done["losses"], done["margins"]
    kept = len(order) + len(items["val"])
    receipt = {
        "steps": done["steps"], "epochs": cfg["epochs"], "train_minutes": (time.monotonic() - started_all) / 60.0,
        "examples": len(order), "dropped_truncated": len(sft_rows) - kept,
        "gpu": "cpu" if smoke else torch.cuda.get_device_name(0), "objective": cfg["objective"],
        "dpo_steps": dpo_steps, "pairs": len(items["pairs"]), "pairs_dropped_truncated": len(pair_rows) - len(items["pairs"]),
        "rows_checked": done["seen"], "dpo_reference": "adapter_off+projector_start" if items["pairs"] else "none",
        "rendering": cfg["rendering"],
        "channel": {"k": ch["k"], "arm": ch["arm"], "fresh": fresh, "projector_from": start_sha,
                    "projector": params_sha(final), "contrast_rows": len(items["contrast"]), "held_out": len(items["val"]),
                    "projector_only_updates": min(done["steps"], int(ch["projector_warmup"] * planned)),
                    "train_margin_mean": float(np.mean(margins)) if margins else None,
                    "loss_first": float(np.mean(losses[:20])) if losses else None,
                    "loss_last": float(np.mean(losses[-20:])) if losses else None,
                    "embeds_gap": embeds_gap, "forward_gap": forward_gap, **held},
    }
    (out / ct.RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0


def _dpo(run: Channel, pairs: list[dict[str, Any]], readings: Mapping[str, Reading], cfg: dict[str, Any],
         deadline: float) -> int:  # pragma: no cover - GPU VM only
    """One DPO epoch with the channel on both sides; the reference is the adapter switched off and
    the projector as the run began, scored once before the pass."""
    torch = run.torch
    optimizer = torch.optim.AdamW([p for p in run.model.parameters() if p.requires_grad]
                                  + [run.params[n] for n in LAYERS], lr=cfg["dpo_learning_rate"], weight_decay=0.0)
    refs = []
    with torch.no_grad(), run.model.disable_adapter():
        for pair in pairs:
            ref_soft = None if pair["own"] is None else run.soft(readings[pair["own"]], ref=True)
            lp = run.logps([(pair["prompt"], pair["chosen"], ref_soft), (pair["prompt"], pair["rejected"], ref_soft)],
                           grad=False)
            refs.append(float(lp[0] - lp[1]))
    steps = 0
    for n, (pair, ref) in enumerate(zip(pairs, refs, strict=True)):
        if time.monotonic() > deadline:
            break
        own = None if pair["own"] is None else run.soft(readings[pair["own"]])
        lp = run.logps([(pair["prompt"], pair["chosen"], own), (pair["prompt"], pair["rejected"], own)])
        loss = -torch.nn.functional.logsigmoid(cfg["dpo_beta"] * ((lp[0] - lp[1]) - ref))
        (loss / cfg["accumulation"]).backward()
        if (n + 1) % cfg["accumulation"] == 0 or n + 1 == len(pairs):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    return steps


def _validate(run: Channel, val: list[dict[str, Any]], readings: Mapping[str, Reading]) -> dict[str, Any]:  # pragma: no cover
    """The held-out contrast rows' margins with their own state and with the empty reading in its
    place: if the two match, the channel carries nothing."""
    torch = run.torch
    found: dict[str, list[float]] = {"state": [], "zero": []}
    with torch.no_grad():
        for item in val:
            other = run.soft(readings[item["other"]])
            for arm, own_key in (("state", item["own"]), ("zero", "empty")):
                own = run.soft(readings[own_key or "empty"])
                lp = run.logps([(item["prompt"], item["reply"], own), (item["prompt"], item["plain"], own),
                                (item["prompt"], item["reply"], other), (item["prompt"], item["plain"], other)],
                               grad=False)
                found[arm].append(float((lp[0] - lp[1]) - (lp[2] - lp[3])))
    return {f"val_{arm}_margin": sum(m) / len(m) for arm, m in found.items()} | {
        f"val_{arm}_right": sum(x > 0 for x in m) / len(m) for arm, m in found.items()}
